"""Compatibility guard for MAFF-Net runs that do not use shared memory.

The upstream OpenPCDet utilities import :mod:`SharedArray` unconditionally,
although VoD evaluation leaves every shared-memory option disabled.  The
binary wheel available in this environment targets a newer NumPy C API.  This
small guard keeps the unused import path available without changing NumPy or
the existing detection environment.
"""


def _disabled(*_args, **_kwargs):
    raise RuntimeError(
        "SharedArray support is disabled for this isolated MAFF-Net run. "
        "Set USE_SHARED_MEMORY=False."
    )


create = _disabled
attach = _disabled
delete = _disabled
