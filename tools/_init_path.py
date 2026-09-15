import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADAR_GAUSSIAN_ROOT = PROJECT_ROOT / "third_party" / "RadarGaussianDet3D"

for path in (PROJECT_ROOT, RADAR_GAUSSIAN_ROOT):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)
