from .MAFF_Net import MAFF_Net
from ..model_utils.dag_radar import (
    CrossRepresentationQualityRanking,
    CrossRepresentationProposalReconciliation,
    GaussianBEVFeatureFusion,
    GaussianObjectStream,
    LearnedCrossRepresentationProposalReconciliation,
    global_compatibility_loss,
    sample_reconciliation_metadata,
    validate_dag_radar_variant,
    zero_background_quality_targets,
)
from ...ops.iou3d_nms import iou3d_nms_utils
import torch


class DaGRadar(MAFF_Net):
    """Dual-representation radar detector with dense and Gaussian object views."""

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        cfg = model_cfg.DUAL_REPRESENTATION
        self.enable_gaussian_object_stream = bool(
            cfg.ENABLE_GAUSSIAN_OBJECT_STREAM
        )
        self.fusion_mode = str(cfg.get("FUSION_MODE", "proposal"))
        if self.fusion_mode not in {"proposal", "feature"}:
            raise ValueError(f"unsupported DaG-Radar fusion mode: {self.fusion_mode}")
        if self.fusion_mode == "feature" and not self.enable_gaussian_object_stream:
            raise ValueError("feature fusion requires the Gaussian object stream")
        self.gaussian_object_stream = None
        if self.enable_gaussian_object_stream:
            stream_cfg = cfg.GAUSSIAN_OBJECT_STREAM
            self.gaussian_object_stream = GaussianObjectStream(
                config_path=stream_cfg.CONFIG_PATH,
                checkpoint_path=stream_cfg.CHECKPOINT_PATH,
                point_cloud_range=tuple(
                    model_cfg.get("POINT_CLOUD_RANGE", dataset.point_cloud_range)
                ),
                max_proposals=int(cfg.MAX_PROPOSALS),
                return_mean=tuple(
                    stream_cfg.get(
                        "RETURN_MEAN",
                        (0.0, 0.0, 0.0, -12.339, -2.818, 0.037, 0.0),
                    )
                ),
                return_std=tuple(
                    stream_cfg.get(
                        "RETURN_STD",
                        (1.0, 1.0, 1.0, 13.240, 1.926, 1.929, 1.0),
                    )
                ),
                input_layout=str(stream_cfg.get("INPUT_LAYOUT", "vod")),
            )
        self.gaussian_feature_fusion = None
        if self.fusion_mode == "feature":
            feature_cfg = cfg.FEATURE_FUSION
            self.gaussian_feature_fusion = GaussianBEVFeatureFusion(
                dense_channels=int(feature_cfg.DENSE_CHANNELS),
                gaussian_channels=int(feature_cfg.GAUSSIAN_CHANNELS),
                out_channels=int(feature_cfg.OUT_CHANNELS),
            )
        self.proposal_reconciliation = CrossRepresentationProposalReconciliation(
            max_proposals=int(cfg.MAX_PROPOSALS),
            match_radii=tuple(cfg.MATCH_RADII),
            geometry_trust=tuple(cfg.GEOMETRY_TRUST),
            recovery_threshold=float(cfg.RECOVERY_THRESHOLD),
            enable_geometry=bool(cfg.ENABLE_GEOMETRY_RECONCILIATION),
        )
        learned_cpr_cfg = cfg.get("LEARNED_CPR", None)
        self.enable_learned_cpr = bool(
            learned_cpr_cfg is not None and learned_cpr_cfg.ENABLED
        )
        if self.enable_learned_cpr:
            self.cpr_mode = str(learned_cpr_cfg.get("MODE", "learned"))
            self.learned_proposal_reconciliation = (
                LearnedCrossRepresentationProposalReconciliation(
                    max_proposals=int(cfg.MAX_PROPOSALS),
                    max_dense_proposals=int(
                        learned_cpr_cfg.get("MAX_DENSE_PROPOSALS", cfg.MAX_PROPOSALS)
                    ),
                    match_radii=tuple(cfg.MATCH_RADII),
                    geometry_trust=tuple(cfg.GEOMETRY_TRUST),
                    recovery_threshold=float(cfg.RECOVERY_THRESHOLD),
                    hidden_dim=int(learned_cpr_cfg.HIDDEN_DIM),
                    match_threshold=float(learned_cpr_cfg.MATCH_THRESHOLD),
                    mode=self.cpr_mode,
                )
            )
            self.learned_cpr_loss_weight = float(learned_cpr_cfg.LOSS_WEIGHT)

        learned_cqr_cfg = cfg.get("LEARNED_CQR", None)
        self.enable_learned_cqr = bool(
            learned_cqr_cfg is not None and learned_cqr_cfg.ENABLED
        )
        validate_dag_radar_variant(
            fusion_mode=self.fusion_mode,
            enable_gaussian=self.enable_gaussian_object_stream,
            enable_cpr=self.enable_learned_cpr,
            enable_cqr=self.enable_learned_cqr,
        )
        if self.enable_learned_cqr:
            self.quality_ranking = CrossRepresentationQualityRanking(
                num_classes=num_class,
                hidden_dim=int(learned_cqr_cfg.HIDDEN_DIM),
                class_embed_dim=int(learned_cqr_cfg.CLASS_EMBED_DIM),
                max_logit_residual=float(learned_cqr_cfg.MAX_LOGIT_RESIDUAL),
                use_relation_features=bool(
                    learned_cqr_cfg.get("USE_RELATION_FEATURES", True)
                ),
            )
            self.learned_cqr_quality_weight = float(
                learned_cqr_cfg.QUALITY_LOSS_WEIGHT
            )
            self.learned_cqr_rank_weight = float(
                learned_cqr_cfg.RANK_LOSS_WEIGHT
            )
            self.learned_cqr_rank_eps = float(learned_cqr_cfg.RANK_MARGIN_EPS)
        self.dag_learned_active_calls = 0

    def _encode_dense_context(self, batch_dict):
        batch_dict = self.vfe(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.image_backbone(batch_dict)
        batch_dict = self.fuser(batch_dict)
        if self.fusion_mode == "feature":
            gaussian_bev = self.gaussian_object_stream.extract_bev(
                batch_dict["points"], batch_dict["batch_size"]
            )
            batch_dict["spatial_features"] = self.gaussian_feature_fusion(
                batch_dict["spatial_features"], gaussian_bev
            )
        batch_dict = self.backbone_2d(batch_dict)
        return self.dense_head(batch_dict)

    def _reconcile_proposals(self, batch_dict):
        native_boxes = batch_dict["rois"].clone()
        native_logits = batch_dict["roi_scores"].clone()
        native_labels = batch_dict["roi_labels"].clone()
        gaussian = self.gaussian_object_stream(
            batch_dict["points"],
            batch_dict["batch_size"],
            gt_boxes=batch_dict.get("gt_boxes") if self.training else None,
        )
        batch_dict["dag_pgs_loss"] = gaussian.training_loss
        heuristic = self.proposal_reconciliation(
            native_boxes, native_logits, native_labels,
            gaussian,
        )
        learned = None
        if self.enable_learned_cpr:
            learned = self.learned_proposal_reconciliation(
                native_boxes, native_logits, native_labels, gaussian,
                gt_boxes=batch_dict.get("gt_boxes") if self.training else None,
            )
        use_learned = learned is not None
        reconciled = learned if use_learned else heuristic
        if use_learned:
            self.dag_learned_active_calls += 1
        batch_dict["rois"] = reconciled.boxes
        batch_dict["roi_scores"] = reconciled.logits
        batch_dict["roi_labels"] = reconciled.labels
        batch_dict["roi_valid_num"] = reconciled.valid_counts
        batch_dict["dag_reconciliation"] = reconciled
        batch_dict["dag_learned_reconciliation"] = learned
        if use_learned:
            metadata = {
                "original_boxes": learned.original_boxes,
                "reconciled_boxes": learned.boxes,
                "trust": learned.trust,
                "source": learned.source,
                "gaussian_scores": learned.gaussian_scores,
            }
        else:
            source = heuristic.labels.new_zeros(heuristic.labels.shape)
            source[heuristic.matched] = 1
            source[heuristic.recovered] = 2
            trust = heuristic.logits.new_zeros(heuristic.logits.shape)
            for class_index, class_trust in enumerate(
                self.proposal_reconciliation.geometry_trust, start=1
            ):
                trust[(heuristic.labels == class_index) & heuristic.matched] = class_trust
            original_boxes = heuristic.boxes.clone()
            for batch_index in range(native_boxes.shape[0]):
                valid = torch.any(native_boxes[batch_index] != 0, dim=-1)
                count = min(int(valid.sum()), original_boxes.shape[1])
                original_boxes[batch_index, :count] = native_boxes[batch_index, valid][:count]
            metadata = {
                "original_boxes": original_boxes,
                "reconciled_boxes": heuristic.boxes,
                "trust": trust,
                "source": source,
                "gaussian_scores": heuristic.gaussian_scores,
            }
        batch_dict["dag_reconciliation_metadata"] = metadata
        return batch_dict

    def _refine_objects(self, batch_dict):
        batch_dict = self.pfe(batch_dict)
        batch_dict = self.point_head(batch_dict)
        return self.roi_head(batch_dict)

    def _learned_training_losses(self, batch_dict):
        zero = next(self.parameters()).sum() * 0.0
        cpr_loss = zero
        learned = batch_dict.get("dag_learned_reconciliation")
        if (
            self.enable_learned_cpr
            and self.cpr_mode == "learned"
            and learned is not None
        ):
            cpr_loss = global_compatibility_loss(
                learned.pair_logits, learned.pair_targets
            )

        quality_loss = zero
        rank_loss = zero
        if self.enable_learned_cqr:
            targets = self.roi_head.forward_ret_dict
            batch_size = batch_dict["batch_size"]
            roi_logits = targets["rcnn_cls"].view(batch_size, -1, 1)
            _, refined_boxes = self.roi_head.generate_predicted_boxes(
                batch_size=batch_size,
                rois=targets["rois"],
                cls_preds=targets["rcnn_cls"],
                box_preds=targets["rcnn_reg"],
            )
            metadata = batch_dict["dag_sampled_metadata"]
            corrected = self.quality_ranking(
                roi_logits, metadata["trust"], metadata["source"],
                targets["roi_labels"], metadata["original_boxes"],
                metadata["reconciled_boxes"], refined_boxes,
            )
            with torch.no_grad():
                assigned_gt = targets["gt_of_rois_src"][..., :7]
                iou_targets = []
                for batch_index in range(batch_size):
                    matrix = iou3d_nms_utils.boxes_iou3d_gpu(
                        refined_boxes[batch_index], assigned_gt[batch_index]
                    )
                    iou_targets.append(torch.diagonal(matrix))
                iou_targets = torch.stack(iou_targets).clamp(0, 1)
                iou_targets = zero_background_quality_targets(
                    iou_targets, targets["rcnn_cls_labels"]
                )
            quality_loss, rank_loss = self.quality_ranking.get_loss(
                corrected, iou_targets, targets["roi_labels"],
                rank_margin_eps=self.learned_cqr_rank_eps,
            )
        return cpr_loss, quality_loss, rank_loss

    def _apply_learned_quality_ranking(self, batch_dict):
        metadata = batch_dict["dag_reconciliation_metadata"]
        base_logits = batch_dict["batch_cls_preds"]
        batch_dict["batch_cls_preds"] = self.quality_ranking(
            base_logits, metadata["trust"], metadata["source"],
            batch_dict["roi_labels"], metadata["original_boxes"],
            metadata["reconciled_boxes"], batch_dict["batch_box_preds"],
        )
        self.dag_learned_active_calls += 1
        return batch_dict

    def forward(self, batch_dict):
        if not self.enable_gaussian_object_stream:
            return super().forward(batch_dict)

        batch_dict = self._encode_dense_context(batch_dict)
        batch_dict = self.roi_head.proposal_layer(
            batch_dict,
            nms_config=self.roi_head.model_cfg.NMS_CONFIG[
                "TRAIN" if self.training else "TEST"
            ],
        )
        if self.fusion_mode == "feature":
            if self.training:
                targets_dict = self.roi_head.assign_targets(batch_dict)
                batch_dict["rois"] = targets_dict["rois"]
                batch_dict["roi_scores"] = targets_dict["roi_scores"]
                batch_dict["roi_labels"] = targets_dict["roi_labels"]
                batch_dict["roi_targets_dict"] = targets_dict
                if "roi_valid_num" in batch_dict:
                    count = targets_dict["rois"].shape[1]
                    batch_dict["roi_valid_num"] = [
                        count
                    ] * batch_dict["batch_size"]
            batch_dict = self._refine_objects(batch_dict)
            if self.training:
                loss, tb_dict, disp_dict = self.get_training_loss()
                return {"loss": loss}, tb_dict, disp_dict
            return self.post_processing(batch_dict)

        batch_dict = self._reconcile_proposals(batch_dict)

        if self.training:
            targets_dict = self.roi_head.assign_targets(batch_dict)
            batch_dict["dag_sampled_metadata"] = sample_reconciliation_metadata(
                batch_dict["dag_reconciliation_metadata"],
                targets_dict["sampled_inds"],
            )
            batch_dict["rois"] = targets_dict["rois"]
            batch_dict["roi_scores"] = targets_dict["roi_scores"]
            batch_dict["roi_labels"] = targets_dict["roi_labels"]
            batch_dict["roi_targets_dict"] = targets_dict
            if "roi_valid_num" in batch_dict:
                count = targets_dict["rois"].shape[1]
                batch_dict["roi_valid_num"] = [count] * batch_dict["batch_size"]

        batch_dict = self._refine_objects(batch_dict)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            cpr_loss, quality_loss, rank_loss = self._learned_training_losses(batch_dict)
            pgs_loss = batch_dict["dag_pgs_loss"]
            if pgs_loss is None:
                raise RuntimeError("Point-Gaussian training loss is unavailable")
            loss = (
                loss
                + pgs_loss
                + (
                    self.learned_cpr_loss_weight * cpr_loss
                    if self.enable_learned_cpr and self.cpr_mode == "learned"
                    else 0
                )
                + (self.learned_cqr_quality_weight * quality_loss if self.enable_learned_cqr else 0)
                + (self.learned_cqr_rank_weight * rank_loss if self.enable_learned_cqr else 0)
            )
            tb_dict["dag_cpr_loss"] = float(cpr_loss.detach())
            tb_dict["dag_cqr_quality_loss"] = float(quality_loss.detach())
            tb_dict["dag_cqr_rank_loss"] = float(rank_loss.detach())
            tb_dict["dag_pgs_loss"] = float(pgs_loss.detach())
            return {"loss": loss}, tb_dict, disp_dict

        if self.enable_learned_cqr:
            batch_dict = self._apply_learned_quality_ranking(batch_dict)
        return self.post_processing(batch_dict)
