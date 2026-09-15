from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional, Sequence

import torch
import torch.nn as nn


VOD_RETURN_MEAN = (0.0, 0.0, 0.0, -12.339, -2.818, 0.037, 0.0)
VOD_RETURN_STD = (1.0, 1.0, 1.0, 13.240, 1.926, 1.929, 1.0)
TJ4D_RETURN_MEAN = (0.0, 0.0, 0.0, -6.569, 86.196, 10.526, -1.093, -0.347)
TJ4D_RETURN_STD = (1.0, 1.0, 1.0, 1.496, 70.192, 3.643, 20.438, 3.902)


@dataclass
class GaussianObjectOutput:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    training_loss: Optional[torch.Tensor] = None


def normalize_and_filter_radar_returns(
    batch_points: torch.Tensor,
    batch_size: int,
    point_cloud_range: Sequence[float],
    mean: Sequence[float],
    std: Sequence[float],
    input_layout: str = "vod",
):
    ranges = batch_points.new_tensor(point_cloud_range)
    mean = batch_points.new_tensor(mean)
    std = batch_points.new_tensor(std)
    scenes = []
    for batch_index in range(int(batch_size)):
        points = batch_points[batch_points[:, 0].long() == batch_index, 1:]
        valid = ((points[:, :3] >= ranges[:3]) & (points[:, :3] <= ranges[3:6])).all(
            dim=-1
        )
        points = points[valid]
        if input_layout == "tj4d_compact":
            if points.shape[-1] != 5:
                raise ValueError(
                    "TJ4DRadSet compact input expects x,y,z,v_r,SNR"
                )
            xyz = points[:, :3]
            planar_range = torch.linalg.vector_norm(xyz[:, :2], dim=-1)
            range_3d = torch.linalg.vector_norm(xyz, dim=-1)
            alpha = torch.atan2(xyz[:, 1], xyz[:, 0])
            beta = torch.atan2(xyz[:, 2], planar_range.clamp_min(1e-6))
            points = torch.cat(
                (
                    xyz,
                    points[:, 3:4],
                    range_3d.unsqueeze(-1),
                    points[:, 4:5],
                    alpha.unsqueeze(-1),
                    beta.unsqueeze(-1),
                ),
                dim=-1,
            )
        elif input_layout != "vod":
            raise ValueError(f"unsupported Gaussian input layout: {input_layout}")
        if points.shape[-1] != mean.numel() or mean.shape != std.shape:
            raise ValueError(
                "Gaussian input width and normalization statistics must match"
            )
        scenes.append((points - mean) / std)
    return scenes


def normalize_and_filter_vod_returns(
    batch_points: torch.Tensor,
    batch_size: int,
    point_cloud_range: Sequence[float],
):
    return normalize_and_filter_radar_returns(
        batch_points=batch_points,
        batch_size=batch_size,
        point_cloud_range=point_cloud_range,
        mean=VOD_RETURN_MEAN,
        std=VOD_RETURN_STD,
        input_layout="vod",
    )


class GaussianObjectStream(nn.Module):
    """Generate object-centric Gaussian hypotheses from the shared radar frame."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: Optional[str],
        point_cloud_range: Sequence[float],
        max_proposals: int = 128,
        return_mean: Sequence[float] = VOD_RETURN_MEAN,
        return_std: Sequence[float] = VOD_RETURN_STD,
        input_layout: str = "vod",
    ):
        super().__init__()
        from mmcv import Config
        from mmcv.runner import load_checkpoint
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        from mmdet3d.models import build_detector

        project_root = Path(__file__).resolve().parents[4]
        radar_gaussian_root = project_root / "third_party" / "RadarGaussianDet3D"
        if str(radar_gaussian_root) not in sys.path:
            sys.path.insert(0, str(radar_gaussian_root))
        import plugin.RadarGaussianDet3D  # noqa: F401

        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = project_root / config_path
        config = Config.fromfile(str(config_path))
        self.encoder = build_detector(
            config.model,
            train_cfg=config.get("train_cfg"),
            test_cfg=config.get("test_cfg"),
        )
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.is_absolute():
                checkpoint_path = project_root / checkpoint_path
            load_checkpoint(
                self.encoder,
                str(checkpoint_path),
                map_location="cpu",
                strict=True,
                logger=None,
            )
        self.box_type = LiDARInstance3DBoxes
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        self.max_proposals = int(max_proposals)
        self.return_mean = tuple(float(value) for value in return_mean)
        self.return_std = tuple(float(value) for value in return_std)
        self.input_layout = str(input_layout)

    def extract_bev(self, batch_points: torch.Tensor, batch_size: int):
        """Encode a radar batch into the trainable Point-Gaussian BEV tensor."""
        scenes = normalize_and_filter_radar_returns(
            batch_points=batch_points,
            batch_size=batch_size,
            point_cloud_range=self.point_cloud_range,
            mean=getattr(self, "return_mean", VOD_RETURN_MEAN),
            std=getattr(self, "return_std", VOD_RETURN_STD),
            input_layout=getattr(self, "input_layout", "vod"),
        )
        return self.encoder.extract_pts_feat(scenes)

    def _convert_ground_truth(self, gt_boxes: torch.Tensor):
        if gt_boxes is None:
            raise ValueError("Point-Gaussian training requires ground-truth boxes")
        if gt_boxes.ndim != 3 or gt_boxes.shape[-1] < 8:
            raise ValueError(
                "ground-truth boxes must have shape [B, N, >=8] with labels last"
            )
        boxes_3d = []
        labels_3d = []
        for scene in gt_boxes:
            labels = scene[:, -1].long()
            valid = (labels > 0) & (scene[:, 3:6] > 0).all(dim=-1)
            boxes = scene[valid, :7].clone()
            if boxes.numel():
                boxes[:, 2] -= boxes[:, 5] * 0.5
            boxes_3d.append(self.box_type(boxes, box_dim=7))
            labels_3d.append(labels[valid] - 1)
        return boxes_3d, labels_3d

    def training_loss(self, outputs, gt_boxes: torch.Tensor):
        boxes_3d, labels_3d = self._convert_ground_truth(gt_boxes)
        loss_dict = self.encoder.pts_bbox_head.loss(
            boxes_3d, labels_3d, outputs
        )
        terms = []
        for value in loss_dict.values():
            values = value if isinstance(value, (list, tuple)) else (value,)
            terms.extend(item.mean() for item in values)
        if not terms:
            raise RuntimeError("Point-Gaussian head returned no training losses")
        return torch.stack(terms).sum()

    def forward(
        self,
        batch_points: torch.Tensor,
        batch_size: int,
        gt_boxes: Optional[torch.Tensor] = None,
    ):
        bev = self.extract_bev(batch_points, batch_size)
        outputs = self.encoder.pts_bbox_head(
            [bev], img_metas=[{} for _ in range(int(batch_size))]
        )
        with torch.no_grad():
            predictions = self.encoder.pts_bbox_head.get_bboxes(
                outputs,
                [{"box_type_3d": self.box_type} for _ in range(int(batch_size))],
            )
        training_loss = (
            self.training_loss(outputs, gt_boxes) if self.training else None
        )
        device, dtype = batch_points.device, batch_points.dtype
        boxes = torch.zeros(
            (batch_size, self.max_proposals, 7), device=device, dtype=dtype
        )
        scores = torch.zeros(
            (batch_size, self.max_proposals), device=device, dtype=dtype
        )
        labels = torch.zeros(
            (batch_size, self.max_proposals), device=device, dtype=torch.long
        )
        for batch_index, (scene_boxes, scene_scores, scene_labels) in enumerate(
            predictions
        ):
            scene_boxes = scene_boxes.tensor.to(device=device, dtype=dtype).clone()
            scene_boxes[:, 2] += scene_boxes[:, 5] * 0.5
            count = min(len(scene_boxes), self.max_proposals)
            boxes[batch_index, :count] = scene_boxes[:count]
            scores[batch_index, :count] = scene_scores[:count].to(dtype)
            labels[batch_index, :count] = scene_labels[:count].long() + 1
        return GaussianObjectOutput(
            boxes=boxes,
            scores=scores,
            labels=labels,
            training_loss=training_loss,
        )
