import os
import cv2
import torch
import numpy as np
from PIL import Image
import clip

# =========================
# CONFIG
# =========================

IMAGE_FOLDER = "datasets/eurosat/2750/AnnualCrop"
OUTPUT_FOLDER = "attention_outputs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================
# LOAD MODEL
# =========================

model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()


# =========================
# PATCH ATTENTION
# =========================

def patch_clip_attention(model):
    for blk in model.visual.transformer.resblocks:
        attn = blk.attn

        def new_forward(q, k, v, attn=attn, **kwargs):
            # q, k, v shape: [seq_len, batch, embed_dim]  (CLIP uses seq-first)
            N, B, C = q.shape
            H = attn.num_heads
            head_dim = C // H

            # Project Q, K, V separately using the combined in_proj_weight
            W_q, W_k, W_v = attn.in_proj_weight.chunk(3, dim=0)
            b_q, b_k, b_v = attn.in_proj_bias.chunk(3, dim=0)

            q = torch.nn.functional.linear(q, W_q, b_q)  # [N, B, C]
            k = torch.nn.functional.linear(k, W_k, b_k)  # [N, B, C]
            v = torch.nn.functional.linear(v, W_v, b_v)  # [N, B, C]

            # Reshape to [B, H, N, head_dim]
            def reshape(t):
                return t.reshape(N, B * H, head_dim).transpose(0, 1).reshape(B, H, N, head_dim)

            q, k, v = reshape(q), reshape(k), reshape(v)

            scale = head_dim ** -0.5
            attn_weights = (q @ k.transpose(-2, -1)) * scale  # [B, H, N, N]
            attn_weights = attn_weights.softmax(dim=-1)
            attn.attn_probs = attn_weights.detach()

            out = attn_weights @ v                                        # [B, H, N, head_dim]
            out = out.reshape(B, H, N, head_dim).permute(2, 0, 1, 3)     # [N, B, H, head_dim]
            out = out.reshape(N, B, C)
            out = torch.nn.functional.linear(out, attn.out_proj.weight, attn.out_proj.bias)
            return out, None   # CLIP expects a tuple (attn_output, attn_weights)

        attn.forward = new_forward


patch_clip_attention(model)


# =========================
# ATTENTION ROLLOUT
# =========================

def get_attention_map(model, image_tensor):
    all_attn = []

    def hook_fn(module, input, output):
        if hasattr(module, "attn_probs"):
            all_attn.append(module.attn_probs.detach().cpu())

    hooks = [blk.attn.register_forward_hook(hook_fn)
             for blk in model.visual.transformer.resblocks]

    with torch.no_grad():
        model.encode_image(image_tensor)

    for h in hooks:
        h.remove()

    if not all_attn:
        raise RuntimeError("No attention captured — patch failed.")

    # Stack → [layers, heads, tokens, tokens]
    attn = torch.stack([a.squeeze(0) for a in all_attn])

    # Use only the LAST 4 layers
    attn = attn[-4:]

    # Keep head MAX (sharper signal)
    attn = attn.max(dim=1).values  # [layers, tokens, tokens]

    num_tokens = attn.size(-1)

    # Residual connection correction
    eye = torch.eye(num_tokens)
    attn = attn + eye.unsqueeze(0)
    attn = attn / attn.sum(dim=-1, keepdim=True)

    # Rollout
    rollout = torch.eye(num_tokens)
    for layer in attn:
        rollout = layer @ rollout

    # CLS token → attention over all patch tokens
    mask = rollout[0, 1:]

    size = int(np.sqrt(mask.shape[0]))  # 7 for ViT-B/32
    mask = mask.reshape(size, size).numpy()

    # Percentile normalization
    low, high = np.percentile(mask, 10), np.percentile(mask, 95)
    mask = np.clip(mask, low, high)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # Power sharpening
    mask = np.power(mask, 1.5)

    return mask


# =========================
# PROCESS & SAVE IMAGE
# =========================

def process_image(img_path, idx):
    img_orig = Image.open(img_path).convert("RGB")
    img_np = np.array(img_orig).astype(np.float32) / 255.0

    H, W = img_np.shape[:2]

    input_tensor = preprocess(img_orig).unsqueeze(0).to(DEVICE)
    mask = get_attention_map(model, input_tensor)

    # Upsample to original size
    mask_up = cv2.resize(mask, (W, H), interpolation=cv2.INTER_CUBIC)
    mask_up = np.clip(mask_up, 0, 1)

    # HOT colormap
    heatmap_raw = cv2.applyColorMap(np.uint8(255 * mask_up), cv2.COLORMAP_HOT)
    heatmap_rgb = heatmap_raw[:, :, ::-1].astype(np.float32) / 255.0

    # Weighted blend
    overlay = 0.55 * heatmap_rgb + 0.45 * img_np
    overlay = np.clip(overlay, 0, 1)

    overlay_bgr = cv2.cvtColor(np.uint8(255 * overlay), cv2.COLOR_RGB2BGR)
    orig_bgr    = cv2.cvtColor(np.uint8(255 * img_np),  cv2.COLOR_RGB2BGR)

    side_by_side = np.concatenate([orig_bgr, heatmap_raw, overlay_bgr], axis=1)

    save_path = os.path.join(OUTPUT_FOLDER, f"annualcrop_{idx:03d}.png")
    cv2.imwrite(save_path, side_by_side)
    print(f"  saved → {save_path}")


# =========================
# RUN
# =========================

image_paths = []
for root, _, files in os.walk(IMAGE_FOLDER):
    for f in sorted(files):
        if f.lower().endswith((".jpg", ".png")):
            image_paths.append(os.path.join(root, f))

image_paths = image_paths[:30]

print(f"Found {len(image_paths)} images. Processing...\n")

for i, path in enumerate(image_paths):
    print(f"[{i+1}/{len(image_paths)}] {os.path.basename(path)}")
    try:
        process_image(path, i)
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n✅ Done — check attention_outputs/")
print("Each saved image is:  [original | heatmap | overlay]  side by side.")