# DaG-Radar

Official PyTorch implementation of **DaG-Radar: Dual-Representation 3D Object
Detection with Dense Context and Gaussian Object Modeling**.

DaG-Radar is a radar-only 3-D object detector that maintains dense-context and
point-Gaussian object hypotheses until the proposal stage. It reconciles the
two representations before shared RoI refinement and then uses their relation
to improve localization-quality ranking.

## Method Overview

The model contains five main components:

1. **Dense Context Stream (DCS)** generates dense BEV features and object
   proposals from pillar and cluster representations.
2. **Point-Gaussian Stream (PGS)** represents radar returns with Gaussian
   primitives and produces complementary object proposals.
3. **Cross-Representation Proposal Reconciliation (CPR)** performs class-aware
   matching, geometry reconciliation, and missed-object recovery.
4. **Shared RoI Refinement** refines the reconciled proposals using point and
   BEV features.
5. **Cross-Representation Quality Ranking (CQR)** corrects proposal scores from
   provenance, reconciliation trust, and two-stage geometric evolution.

## Repository Structure

```text
DaG-Radar-core/
|-- configs/                         # Main VoD and TJ4DRadSet configurations
|-- pcdet/
|   |-- models/detectors/DaGRadar.py # Detector integration and forward path
|   `-- models/model_utils/dag_radar/
|       |-- gaussian_object_stream.py
|       |-- learned_reconciliation.py
|       |-- proposal_reconciliation.py
|       `-- feature_fusion.py
|-- third_party/RadarGaussianDet3D/  # Point-Gaussian stream dependency
|-- tools/                           # Training, evaluation, and data utilities
|-- requirements.txt
`-- setup.py
```

## Environment

The reference environment uses:

- Python 3.9
- PyTorch 1.12.0 with CUDA 11.3
- OpenPCDet-compatible CUDA extensions
- MMCV 1.6.0
- MMDetection 2.25.0
- MMSegmentation 0.26.0
- MMDetection3D 1.0.0rc3

Install the Python dependencies and build the detector extensions:

```bash
pip install -r requirements.txt
python setup.py develop
```

Build the Gaussian rasterization extension when using the point-Gaussian
stream:

```bash
cd third_party/RadarGaussianDet3D/plugin/RadarGaussianDet3D/ops/diff-gaussian-rasterization-bev
python setup.py install
cd ../../../../../..
```

## Data Preparation

Download the View-of-Delft (VoD) or TJ4DRadSet dataset from its official source.
Datasets and model weights are not included in this source release. The default
dataset locations are defined in:

```text
tools/cfgs/dataset_configs/vod_dataset_radar.yaml
tools/cfgs/dataset_configs/TJ4DRadSet_dataset_radar.yaml
```

Update these paths for the local environment before training or evaluation.

## Main Configurations

The paper-aligned main configurations are:

```text
configs/paper/DaGRadar_vod_cpr_cqr.yaml
configs/paper/DaGRadar_tj4d_cpr_cqr.yaml
```

The standard OpenPCDet entry points are used for training and evaluation:

```bash
python tools/train.py --cfg_file configs/paper/DaGRadar_vod_cpr_cqr.yaml --seed 12

python tools/test.py \
  --cfg_file configs/paper/DaGRadar_vod_cpr_cqr.yaml \
  --ckpt /path/to/checkpoint.pth
```

Use seeds `12`, `24`, and `666` for the reported multi-seed protocol. The
TJ4DRadSet experiment uses the corresponding configuration listed above.

## Core Implementation

- Detector orchestration and losses:
  `pcdet/models/detectors/DaGRadar.py`
- CPR compatibility, matching, recovery, and CQR:
  `pcdet/models/model_utils/dag_radar/learned_reconciliation.py`
- Point-Gaussian stream adapter:
  `pcdet/models/model_utils/dag_radar/gaussian_object_stream.py`
- Deterministic proposal reconciliation:
  `pcdet/models/model_utils/dag_radar/proposal_reconciliation.py`

## Acknowledgements

This implementation builds on OpenPCDet and incorporates components adapted
from MAFF-Net and RadarGaussianDet3D. We thank the authors of these projects for
making their research code available.
