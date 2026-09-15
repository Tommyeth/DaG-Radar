from dataclasses import dataclass
from typing import Sequence

import torch

from .gaussian_object_stream import GaussianObjectOutput


@dataclass
class ReconciledProposals:
    boxes: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    matched: torch.Tensor
    recovered: torch.Tensor
    gaussian_scores: torch.Tensor
    valid_counts: list


class CrossRepresentationProposalReconciliation:
    """Own, recover, and geometrically reconcile dual-representation proposals."""

    def __init__(
        self,
        max_proposals: int,
        match_radii: Sequence[float],
        geometry_trust: Sequence[float],
        recovery_threshold: float,
        enable_geometry: bool = True,
    ):
        self.max_proposals = int(max_proposals)
        self.match_radii = tuple(float(value) for value in match_radii)
        self.geometry_trust = tuple(float(value) for value in geometry_trust)
        self.recovery_threshold = float(recovery_threshold)
        self.enable_geometry = bool(enable_geometry)

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
        reconciled = native_box + float(trust) * (blended - native_box)
        heading_delta = torch.atan2(
            torch.sin(blended[6] - native_box[6]),
            torch.cos(blended[6] - native_box[6]),
        )
        heading = native_box[6] + float(trust) * heading_delta
        reconciled[6] = torch.atan2(torch.sin(heading), torch.cos(heading))
        return reconciled

    def __call__(self, native_boxes, native_logits, native_labels, gaussian):
        batch_size, _, channels = native_boxes.shape
        boxes = native_boxes.new_zeros((batch_size, self.max_proposals, channels))
        logits = native_logits.new_zeros((batch_size, self.max_proposals))
        labels = native_labels.new_zeros((batch_size, self.max_proposals))
        matched = torch.zeros(
            (batch_size, self.max_proposals),
            device=native_boxes.device,
            dtype=torch.bool,
        )
        recovered = torch.zeros_like(matched)
        gaussian_scores = native_logits.new_zeros((batch_size, self.max_proposals))
        valid_counts = []
        radii = native_boxes.new_tensor(self.match_radii)

        for batch_index in range(batch_size):
            native_valid = torch.any(native_boxes[batch_index] != 0, dim=-1)
            base_boxes = native_boxes[batch_index, native_valid]
            base_logits = native_logits[batch_index, native_valid]
            base_labels = native_labels[batch_index, native_valid].long()
            dense_order = torch.argsort(base_logits, descending=True)
            base_boxes = base_boxes[dense_order]
            base_logits = base_logits[dense_order]
            base_labels = base_labels[dense_order]
            gaussian_valid = gaussian.labels[batch_index] > 0
            candidate_boxes = gaussian.boxes[batch_index, gaussian_valid]
            candidate_scores = gaussian.scores[batch_index, gaussian_valid].clamp(
                1e-4, 1.0 - 1e-4
            )
            candidate_labels = gaussian.labels[batch_index, gaussian_valid].long()
            used = torch.zeros(
                len(candidate_boxes), device=native_boxes.device, dtype=torch.bool
            )

            base_count = min(len(base_boxes), self.max_proposals)
            boxes[batch_index, :base_count] = base_boxes[:base_count]
            logits[batch_index, :base_count] = base_logits[:base_count]
            labels[batch_index, :base_count] = base_labels[:base_count]

            for base_index in range(base_count):
                class_index = int(base_labels[base_index]) - 1
                if not 0 <= class_index < len(self.match_radii):
                    continue
                candidates = torch.where(
                    (candidate_labels == base_labels[base_index]) & ~used
                )[0]
                if len(candidates) == 0:
                    continue
                distances = torch.linalg.vector_norm(
                    candidate_boxes[candidates, :2] - base_boxes[base_index, :2],
                    dim=-1,
                )
                nearest_offset = distances.argmin()
                if distances[nearest_offset] > radii[class_index]:
                    continue
                candidate_index = candidates[nearest_offset]
                used[candidate_index] = True
                score = candidate_scores[candidate_index]
                if self.enable_geometry:
                    boxes[batch_index, base_index] = self._blend_geometry(
                        base_boxes[base_index],
                        base_logits[base_index].sigmoid(),
                        candidate_boxes[candidate_index],
                        score,
                        self.geometry_trust[class_index],
                    )
                matched[batch_index, base_index] = True
                gaussian_scores[batch_index, base_index] = score

            output_index = base_count
            for candidate_index in torch.argsort(candidate_scores, descending=True):
                if output_index >= self.max_proposals:
                    break
                if used[candidate_index]:
                    continue
                class_index = int(candidate_labels[candidate_index]) - 1
                if not 0 <= class_index < len(self.match_radii):
                    continue
                score = candidate_scores[candidate_index]
                if score < self.recovery_threshold:
                    continue
                boxes[batch_index, output_index] = candidate_boxes[candidate_index]
                logits[batch_index, output_index] = torch.logit(
                    score.clamp(1e-4, 1.0 - 1e-4)
                )
                labels[batch_index, output_index] = candidate_labels[candidate_index]
                recovered[batch_index, output_index] = True
                gaussian_scores[batch_index, output_index] = score
                output_index += 1
            valid_counts.append(output_index)

        return ReconciledProposals(
            boxes=boxes,
            logits=logits,
            labels=labels,
            matched=matched,
            recovered=recovered,
            gaussian_scores=gaussian_scores,
            valid_counts=valid_counts,
        )
