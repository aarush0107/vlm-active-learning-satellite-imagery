import os
import random
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from .AL import AL


class PCB(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, statistics, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.pred = []
        self.confidence = []

        # IMPORTANT: statistics = class distribution or priority
        self.statistics = statistics

        # Lower threshold to avoid missing rare classes
        self.conf_threshold = 0.1

        # XAI output folder
        self.xai_dir = "xai_outputs"
        os.makedirs(self.xai_dir, exist_ok=True)

    def save_xai_stub(self, img_tensor, idx):
        try:
            img = img_tensor.cpu()
            if img.dim() == 4:
                img = img[0]
            img = to_pil_image(img)
            img.save(os.path.join(self.xai_dir, f"query_{idx}.png"))
        except Exception as e:
            print(f"XAI save failed for {idx}: {e}")

    def select(self, n_query, **kwargs):
        print("🔥 PCB SELECT RUNNING 🔥")

        self.pred = []
        self.confidence = []
        self.model.eval()

        num_unlabeled = len(self.U_index)
        assert len(self.unlabeled_set) == num_unlabeled

        cached_images = []

        with torch.no_grad():
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            for batch in unlabeled_loader:
                inputs = batch["img"].to(self.device)
                cached_images.extend(inputs.detach().cpu())

                out, _ = self.model(inputs, get_feature=True)

                probs = torch.nn.functional.softmax(out, dim=1)

                max_inds = torch.argmax(probs, dim=1)
                max_confs, _ = torch.max(probs, dim=1)

                self.pred.append(max_inds.detach().cpu())
                self.confidence.append(max_confs.detach().cpu())

        self.pred = torch.cat(self.pred)
        self.confidence = torch.cat(self.confidence)

        Q_index = []

        while len(Q_index) < n_query:

            # ✅ FIXED: select least represented class
            target_cls = int(torch.argmin(self.statistics))

            # indices belonging to that class
            candidates = (self.pred == target_cls).nonzero().squeeze(dim=1).tolist()

            # filter valid candidates
            valid_candidates = [
                idx for idx in candidates
                if idx not in Q_index and self.confidence[idx].item() > self.conf_threshold
            ]

            if len(valid_candidates) > 0:

                # sort by uncertainty (low confidence first)
                sorted_candidates = sorted(
                    valid_candidates,
                    key=lambda idx: self.confidence[idx].item()
                )

                # ✅ diversity: pick randomly from top uncertain
                top_k = min(5, len(sorted_candidates))
                chosen_idx = random.choice(sorted_candidates[:top_k])

                Q_index.append(chosen_idx)

                # update class statistics (simulate labeling)
                self.statistics[target_cls] += 1

            else:
                # fallback: random but safe
                num = random.randint(0, num_unlabeled - 1)
                while num in Q_index:
                    num = random.randint(0, num_unlabeled - 1)
                Q_index.append(num)

        # Save XAI stubs (optional)
        for idx in Q_index:
            if idx < len(cached_images):
                self.save_xai_stub(cached_images[idx], idx)

        # Save readable indices (IMPORTANT)
        with open("selected_indices.txt", "w") as f:
            for idx in Q_index:
                cls = int(self.pred[idx])
                f.write(f"Index: {self.U_index[idx]} | Class: {cls}\n")

        # map to original indices
        Q_index = [self.U_index[idx] for idx in Q_index]

        return Q_index