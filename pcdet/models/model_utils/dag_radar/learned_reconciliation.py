from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gaussian_object_stream import GaussianObjectOutput


def global_compatibility_loss(pair_logits, pair_targets):
    """Average CPR binary cross-entropy over all non-empty candidate pairs."""
    valid = []
    for logits, targets in zip(pair_logits, pair_targets):
        if logits.numel() != targets.numel():
            raise ValueError("CPR logits and targets must contain the same pairs")
        if logits.numel():
            valid.append((logits.reshape(-1), targets.float().reshape(-1)))
    if not valid:
        anchor = next(iter(pair_logits), None)
        return anchor.sum() * 0.0 if anchor is not None else torch.tensor(0.0)
    logits, targets = zip(*valid)
    return F.binary_cross_entropy_with_logits(torch.cat(logits), torch.cat(targets))


def zero_background_quality_targets(iou_targets, rcnn_cls_labels):
    """Keep refined IoU for non-background RoIs and set background to zero."""
    if iou_targets.shape != rcnn_cls_labels.shape:
        raise ValueError("CQR IoU targets and RoI labels must have the same shape")
    return torch.where(
        rcnn_cls_labels > 0,
        iou_targets,
        torch.zeros_like(iou_targets),
    )


def sample_reconciliation_metadata(metadata, sampled_inds):
    """Gather proposal metadata with the exact RoI target-sampling indices."""
    output = {}
    for name, value in metadata.items():
        index = sampled_inds
        while index.dim() < value.dim():
            index = index.unsqueeze(-1)
        index = index.expand(*sampled_inds.shape, *value.shape[2:])
        output[name] = torch.gather(value, 1, index)
    return output


@dataclass
class LearnedReconciledProposals:
    boxes: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    matched: torch.Tensor
    recovered: torch.Tensor
    gaussian_scores: torch.Tensor
    trust: torch.Tensor
    source: torch.Tensor
    original_boxes: torch.Tensor
    pair_logits: list
    pair_dense_indices: list
    pair_gaussian_indices: list
    pair_targets: list
    valid_counts: list


class LearnedCrossRepresentationProposalReconciliation(nn.Module):
    """Learn proposal compatibility and use it for matching and box trust."""

    def __init__(
        self,
        max_proposals: int,
        match_radii: Sequence[float],
        recovery_threshold: float,
        geometry_trust: Sequence[float] = None,
        max_dense_proposals: int = None,
        hidden_dim: int = 64,
        match_threshold: float = 0.5,
        mode: str = "learned",
    ):
        super().__init__()
        if mode not in {"fixed", "learned"}:
            raise ValueError(f"unsupported CPR mode: {mode}")
        self.mode = mode
        self.max_proposals = int(max_proposals)
        self.max_dense_proposals = (
            self.max_proposals
            if max_dense_proposals is None
            else min(int(max_dense_proposals), self.max_proposals)
        )
        self.match_radii = tuple(float(v) for v in match_radii)
        self.geometry_trust = tuple(
            float(v) for v in (
                geometry_trust if geometry_trust is not None
                else [1.0] * len(self.match_radii)
            )
        )
        self.recovery_threshold = float(recovery_threshold)
        self.match_threshold = float(match_threshold)
        self.compatibility_head = None
        if self.mode == "learned":
            self.compatibility_head = nn.Sequential(
                nn.Linear(9, int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Linear(int(hidden_dim), 1),
            )
            nn.init.zeros_(self.compatibility_head[-1].weight)
            nn.init.zeros_(self.compatibility_head[-1].bias)

    @staticmethod
    def build_pair_descriptor(
        dense_boxes,
        dense_scores,
        gaussian_boxes,
        gaussian_scores,
        radii,
    ):
        radii = radii.reshape(-1, 1).clamp_min(1e-3)
        center = (dense_boxes[:, :2] - gaussian_boxes[:, :2]) / radii
        size = torch.log(
            dense_boxes[:, 3:6].clamp_min(1e-3)
            / gaussian_boxes[:, 3:6].clamp_min(1e-3)
        )
        heading = dense_boxes[:, 6] - gaussian_boxes[:, 6]
        return torch.cat(
            [
                center,
                size,
                torch.sin(heading).unsqueeze(-1),
                torch.cos(heading).unsqueeze(-1),
                dense_scores.reshape(-1, 1),
                gaussian_scores.reshape(-1, 1),
            ],
            dim=-1,
        )

    @staticmethod
    def compatibility_loss(pair_logits, pair_targets):
        if pair_logits.numel() == 0:
            return pair_logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            pair_logits.reshape(-1), pair_targets.float().reshape(-1)
        )

    @staticmethod
    def _assign_box_centers_to_gt(boxes, labels, gt_boxes):
        assignment = labels.new_full((len(boxes),), -1)
        if len(boxes) == 0 or len(gt_boxes) == 0:
            return assignment
        for gt_index, gt in enumerate(gt_boxes):
            if gt.abs().sum() == 0:
                continue
            class_mask = labels == gt[-1].long()
            delta = boxes[:, :2] - gt[:2]
            cosine, sine = torch.cos(gt[6]), torch.sin(gt[6])
            local_x = delta[:, 0] * cosine + delta[:, 1] * sine
            local_y = -delta[:, 0] * sine + delta[:, 1] * cosine
            local_z = boxes[:, 2] - gt[2]
            inside = (
                (local_x.abs() <= gt[3].clamp_min(1e-3) / 2)
                & (local_y.abs() <= gt[4].clamp_min(1e-3) / 2)
                & (local_z.abs() <= gt[5].clamp_min(1e-3) / 2)
                & class_mask
            )
            assignment[(assignment < 0) & inside] = gt_index
        return assignment

    def build_pair_targets(
        self, dense_boxes, gaussian_boxes, dense_labels, gaussian_labels,
        gt_boxes, pair_dense_indices, pair_gaussian_indices,
    ):
        if pair_dense_indices.numel() == 0:
            return dense_boxes.new_empty(0)
        dense_owner = self._assign_box_centers_to_gt(
            dense_boxes, dense_labels, gt_boxes
        )[pair_dense_indices]
        gaussian_owner = self._assign_box_centers_to_gt(
            gaussian_boxes, gaussian_labels, gt_boxes
        )[pair_gaussian_indices]
        return ((dense_owner >= 0) & (dense_owner == gaussian_owner)).float()

    @staticmethod
    def _blend_geometry(native_box, native_score, gaussian_box, gaussian_score, trust):
        denominator = (native_score + gaussian_score).clamp_min(1e-6)
        blended = (native_box * native_score + gaussian_box * gaussian_score) / denominator
        blended = blended.clone()
        blended[6] = torch.atan2(
            torch.sin(native_box[6]) * native_score
            + torch.sin(gaussian_box[6]) * gaussian_score,
            torch.cos(native_box[6]) * native_score
            + torch.cos(gaussian_box[6]) * gaussian_score,
        )
        reconciled = native_box + trust * (blended - native_box)
        heading_delta = torch.atan2(
            torch.sin(blended[6] - native_box[6]),
            torch.cos(blended[6] - native_box[6]),
        )
        heading = native_box[6] + trust * heading_delta
        reconciled[6] = torch.atan2(torch.sin(heading), torch.cos(heading))
        return reconciled

    def _candidate_pairs(self, boxes, scores, labels, g_boxes, g_scores, g_labels):
        dense_ids = []
        gaussian_ids = []
        radii = []
        for dense_index in range(len(boxes)):
            class_index = int(labels[dense_index]) - 1
            if not 0 <= class_index < len(self.match_radii):
                continue
            candidates = torch.where(g_labels == labels[dense_index])[0]
            if candidates.numel() == 0:
                continue
            distance = torch.linalg.vector_norm(
                g_boxes[candidates, :2] - boxes[dense_index, :2], dim=-1
            )
            for candidate_index in candidates[distance <= self.match_radii[class_index]]:
                dense_ids.append(dense_index)
                gaussian_ids.append(int(candidate_index))
                radii.append(self.match_radii[class_index])
        if not dense_ids:
            empty = labels.new_empty(0)
            return empty, empty, scores.new_empty(0), scores.new_empty((0, 9))
        dense_ids = labels.new_tensor(dense_ids)
        gaussian_ids = labels.new_tensor(gaussian_ids)
        descriptor = self.build_pair_descriptor(
            boxes[dense_ids], scores[dense_ids], g_boxes[gaussian_ids],
            g_scores[gaussian_ids], scores.new_tensor(radii),
        )
        normalized_distance = torch.linalg.vector_norm(descriptor[:, :2], dim=-1)
        prior_probability = (
            1.0 - (1.0 - self.match_threshold) * normalized_distance
        ).clamp(self.match_threshold, 1.0 - 1e-4)
        prior_logit = torch.logit(prior_probability)
        pair_logits = prior_logit
        if self.compatibility_head is not None:
            pair_logits = pair_logits + self.compatibility_head(descriptor).squeeze(-1)
        return dense_ids, gaussian_ids, pair_logits, descriptor

    def _greedy_matches(self, dense_ids, gaussian_ids, pair_logits):
        if pair_logits.numel() == 0:
            return []
        probabilities = pair_logits.sigmoid()
        matches = []
        used_gaussian = set()
        for dense_index in torch.unique(dense_ids, sorted=True):
            candidates = torch.where(dense_ids == dense_index)[0]
            candidates = sorted(
                candidates.tolist(), key=lambda i: float(probabilities[i]), reverse=True
            )
            pair_index = next(
                (i for i in candidates if int(gaussian_ids[i]) not in used_gaussian),
                None,
            )
            if pair_index is None or probabilities[pair_index] < self.match_threshold:
                continue
            matches.append(pair_index)
            used_gaussian.add(int(gaussian_ids[pair_index]))
        return matches

    def forward(self, native_boxes, native_logits, native_labels, gaussian: GaussianObjectOutput, gt_boxes=None):
        batch_size, _, channels = native_boxes.shape
        boxes = native_boxes.new_zeros((batch_size, self.max_proposals, channels))
        original_boxes = boxes.clone()
        logits = native_logits.new_zeros((batch_size, self.max_proposals))
        labels = native_labels.new_zeros((batch_size, self.max_proposals))
        matched = torch.zeros((batch_size, self.max_proposals), device=boxes.device, dtype=torch.bool)
        recovered = torch.zeros_like(matched)
        gaussian_scores = logits.new_zeros(logits.shape)
        trust = logits.new_zeros(logits.shape)
        source = labels.new_zeros(labels.shape)
        pair_logits_all, pair_dense_all, pair_gaussian_all = [], [], []
        pair_targets_all, valid_counts = [], []

        for batch_index in range(batch_size):
            native_valid = torch.any(native_boxes[batch_index] != 0, dim=-1)
            base_boxes = native_boxes[batch_index, native_valid]
            base_logits = native_logits[batch_index, native_valid]
            base_scores = base_logits.sigmoid()
            base_labels = native_labels[batch_index, native_valid].long()
            dense_order = torch.argsort(base_logits, descending=True)
            base_boxes = base_boxes[dense_order]
            base_logits = base_logits[dense_order]
            base_scores = base_scores[dense_order]
            base_labels = base_labels[dense_order]
            gaussian_valid = gaussian.labels[batch_index] > 0
            g_boxes = gaussian.boxes[batch_index, gaussian_valid]
            g_scores = gaussian.scores[batch_index, gaussian_valid].clamp(
                1e-4, 1.0 - 1e-4
            )
            g_labels = gaussian.labels[batch_index, gaussian_valid].long()

            base_count = min(len(base_boxes), self.max_dense_proposals)
            boxes[batch_index, :base_count] = base_boxes[:base_count]
            original_boxes[batch_index, :base_count] = base_boxes[:base_count]
            logits[batch_index, :base_count] = base_logits[:base_count]
            labels[batch_index, :base_count] = base_labels[:base_count]

            dense_ids, gaussian_ids, pair_logits, descriptor = self._candidate_pairs(
                base_boxes[:base_count], base_scores[:base_count], base_labels[:base_count],
                g_boxes, g_scores, g_labels,
            )
            pair_logits_all.append(pair_logits)
            pair_dense_all.append(dense_ids)
            pair_gaussian_all.append(gaussian_ids)
            if gt_boxes is None:
                pair_targets_all.append(pair_logits.new_empty(0))
            else:
                pair_targets_all.append(self.build_pair_targets(
                    base_boxes[:base_count], g_boxes, base_labels[:base_count],
                    g_labels, gt_boxes[batch_index], dense_ids, gaussian_ids,
                ))
            used = torch.zeros(len(g_boxes), device=boxes.device, dtype=torch.bool)
            explained = torch.zeros_like(used)
            if gaussian_ids.numel():
                explained[gaussian_ids] = True
            for pair_index in self._greedy_matches(dense_ids, gaussian_ids, pair_logits):
                dense_index = int(dense_ids[pair_index])
                gaussian_index = int(gaussian_ids[pair_index])
                q = pair_logits[pair_index].sigmoid()
                class_index = int(base_labels[dense_index]) - 1
                if self.mode == "fixed":
                    rho = q.new_tensor(self.geometry_trust[class_index])
                else:
                    prior_q = (
                        1.0
                        - (1.0 - self.match_threshold)
                        * torch.linalg.vector_norm(descriptor[pair_index, :2])
                    ).clamp(self.match_threshold, 1.0 - 1e-4)
                    rho = (
                        self.geometry_trust[class_index] + 0.1 * (q - prior_q)
                    ).clamp(0, 1)
                boxes[batch_index, dense_index] = self._blend_geometry(
                    base_boxes[dense_index], base_scores[dense_index],
                    g_boxes[gaussian_index], g_scores[gaussian_index], rho,
                )
                matched[batch_index, dense_index] = True
                source[batch_index, dense_index] = 1
                gaussian_scores[batch_index, dense_index] = g_scores[gaussian_index]
                trust[batch_index, dense_index] = rho
                used[gaussian_index] = True

            output_index = base_count
            for gaussian_index in torch.argsort(g_scores, descending=True):
                if output_index >= self.max_proposals:
                    break
                if (
                    used[gaussian_index]
                    or explained[gaussian_index]
                    or g_scores[gaussian_index] < self.recovery_threshold
                ):
                    continue
                boxes[batch_index, output_index] = g_boxes[gaussian_index]
                original_boxes[batch_index, output_index] = g_boxes[gaussian_index]
                logits[batch_index, output_index] = torch.logit(g_scores[gaussian_index])
                labels[batch_index, output_index] = g_labels[gaussian_index]
                recovered[batch_index, output_index] = True
                source[batch_index, output_index] = 2
                gaussian_scores[batch_index, output_index] = g_scores[gaussian_index]
                output_index += 1
            valid_counts.append(output_index)

        return LearnedReconciledProposals(
            boxes, logits, labels, matched, recovered, gaussian_scores, trust,
            source, original_boxes, pair_logits_all, pair_dense_all,
            pair_gaussian_all, pair_targets_all, valid_counts,
        )


class CrossRepresentationQualityRanking(nn.Module):
    """Predict a bounded post-RoI score residual from proposal relations."""

    def __init__(
        self,
        num_classes,
        hidden_dim=64,
        class_embed_dim=8,
        max_logit_residual=1.0,
        use_relation_features=True,
    ):
        super().__init__()
        self.max_logit_residual = float(max_logit_residual)
        self.use_relation_features = bool(use_relation_features)
        self.class_embedding = nn.Embedding(int(num_classes) + 1, int(class_embed_dim))
        self.source_embedding = (
            nn.Embedding(3, int(class_embed_dim))
            if self.use_relation_features
            else None
        )
        input_dim = (
            2 + 12 + 2 * int(class_embed_dim)
            if self.use_relation_features
            else 1 + int(class_embed_dim)
        )
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)), nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    @staticmethod
    def _box_delta(source, target):
        radius = torch.linalg.vector_norm(source[..., :2], dim=-1, keepdim=True).clamp_min(1.0)
        center = (target[..., :2] - source[..., :2]) / radius
        size = torch.log(target[..., 3:6].clamp_min(1e-3) / source[..., 3:6].clamp_min(1e-3))
        heading = torch.sin(target[..., 6] - source[..., 6]).unsqueeze(-1)
        return torch.cat([center, size, heading], dim=-1)

    def forward(self, roi_logits, trust, source, labels, original_boxes, reconciled_boxes, refined_boxes):
        native_logits = roi_logits.detach()
        if self.use_relation_features:
            features = torch.cat(
                [
                    native_logits,
                    trust.detach().unsqueeze(-1),
                    self._box_delta(original_boxes, reconciled_boxes).detach(),
                    self._box_delta(reconciled_boxes, refined_boxes).detach(),
                    self.source_embedding(source.long().clamp(0, 2)),
                    self.class_embedding(labels.long().clamp(0, self.class_embedding.num_embeddings - 1)),
                ],
                dim=-1,
            )
        else:
            features = torch.cat(
                [
                    native_logits,
                    self.class_embedding(
                        labels.long().clamp(
                            0, self.class_embedding.num_embeddings - 1
                        )
                    ),
                ],
                dim=-1,
            )
        residual = self.max_logit_residual * torch.tanh(
            self.output_layer(self.feature_layer(features))
        )
        return native_logits + residual

    @staticmethod
    def get_loss(corrected_logits, iou_targets, labels, rank_margin_eps=0.05):
        logits = corrected_logits.squeeze(-1)
        quality = F.binary_cross_entropy_with_logits(logits, iou_targets.detach().float())
        pair_terms = []
        for batch_index in range(logits.shape[0]):
            for class_label in torch.unique(labels[batch_index]):
                if int(class_label) <= 0:
                    continue
                indices = torch.where(labels[batch_index] == class_label)[0]
                if indices.numel() < 2:
                    continue
                targets = iou_targets[batch_index, indices].detach()
                difference = targets[:, None] - targets[None, :]
                pair_i, pair_j = torch.where(difference > float(rank_margin_eps))
                if pair_i.numel():
                    pair_terms.append(F.softplus(-(
                        logits[batch_index, indices[pair_i]]
                        - logits[batch_index, indices[pair_j]]
                    )))
        ranking = (
            torch.cat(pair_terms).mean()
            if pair_terms
            else logits.sum() * 0.0
        )
        return quality, ranking
