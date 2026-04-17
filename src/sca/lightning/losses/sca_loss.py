import torch
import torch.nn as nn

from sca.losses.pairwise import compute_other_loss
from sca.lightning.losses.retrieval import compute_triplet_loss


class SCALoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.criterion_triplet = nn.TripletMarginLoss(
            margin=cfg.margin,
            p=2,
            reduction="sum",
        )

    def forward(self, feats_ground, feats_aerial, data_dict, triplets_local_indexes):
        other_loss = compute_other_loss(
            feats_ground,
            feats_aerial,
            data_dict,
            positive_thd=self.cfg.train_positives_dist_threshold,
            negative_thd=self.cfg.val_positive_dist_threshold,
            args=self.cfg,
        )

        feats_ground_embed = feats_ground["embedding"].unsqueeze(1)
        feats_aerial_embed = feats_aerial["embedding"]
        features = torch.cat((feats_ground_embed, feats_aerial_embed), dim=1)
        features = features.view(-1, self.cfg.features_dim)
        triplet_loss = compute_triplet_loss(
            self.cfg,
            self.criterion_triplet,
            triplets_local_indexes,
            features,
        )
        total_loss = other_loss + triplet_loss * self.cfg.tripletloss_weight

        return {
            "total": total_loss,
            "triplet": triplet_loss,
            "other": other_loss,
        }
