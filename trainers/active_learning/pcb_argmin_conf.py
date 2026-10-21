import torch
from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader
import random
from .AL import AL

class PCB(AL):
    """
    Ablation variant: argmin class balancing + confidence filtering.
    No top-K diversity — deterministically picks most uncertain passing candidate.
    """
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, statistics, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device
        self.pred = []
        self.confidence = []
        self.statistics = statistics
        self.conf_threshold = 0.1

        # logging
        self.selection_log = []        # (round, class_selected)
        self.filter_log = []           # (round, total_candidates, valid_candidates)

    def select(self, n_query, round_num=0, **kwargs):
        self.pred = []
        self.confidence = []
        self.model.eval()

        num_unlabeled = len(self.U_index)
        assert len(self.unlabeled_set) == num_unlabeled

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
                out, _ = self.model(inputs, get_feature=True)
                probs = torch.nn.functional.softmax(out, dim=1)
                max_confs, max_inds = torch.max(probs, dim=1)
                self.pred.append(max_inds.detach().cpu())
                self.confidence.append(max_confs.detach().cpu())

        self.pred = torch.cat(self.pred)
        self.confidence = torch.cat(self.confidence)

        Q_index = []

        while len(Q_index) < n_query:
            target_cls = int(torch.argmin(self.statistics))

            candidates = (self.pred == target_cls).nonzero().squeeze(dim=1).tolist()

            # confidence filter applied
            valid_candidates = [
                idx for idx in candidates
                if idx not in Q_index and self.confidence[idx].item() >= self.conf_threshold
            ]

            # log filtering stats
            self.filter_log.append((round_num, len(candidates), len(valid_candidates)))

            if len(valid_candidates) > 0:
                # deterministic: pick single most uncertain, no top-K randomness
                sorted_candidates = sorted(
                    valid_candidates,
                    key=lambda idx: self.confidence[idx].item()
                )
                chosen_idx = sorted_candidates[0]
                Q_index.append(chosen_idx)
                self.statistics[target_cls] += 1
                self.selection_log.append((round_num, target_cls))
            else:
                num = random.randint(0, num_unlabeled - 1)
                while num in Q_index:
                    num = random.randint(0, num_unlabeled - 1)
                Q_index.append(num)

        Q_index = [self.U_index[idx] for idx in Q_index]
        return Q_index