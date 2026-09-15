def validate_dag_radar_variant(
    fusion_mode,
    enable_gaussian,
    enable_cpr,
    enable_cqr,
):
    """Reject method combinations that do not correspond to a defined variant."""
    fusion_mode = str(fusion_mode)
    enable_gaussian = bool(enable_gaussian)
    enable_cpr = bool(enable_cpr)
    enable_cqr = bool(enable_cqr)

    if fusion_mode not in {"proposal", "feature"}:
        raise ValueError(f"unsupported DaG-Radar fusion mode: {fusion_mode}")
    if fusion_mode == "feature" and not enable_gaussian:
        raise ValueError("feature fusion requires the Gaussian object stream")
    if fusion_mode == "feature" and (enable_cpr or enable_cqr):
        raise ValueError("feature fusion cannot be combined with CPR or CQR")
    if enable_cqr and not enable_cpr:
        raise ValueError("CQR requires an active learned CPR path")

