"""Path shim: make the vendored NVlabs/GatedDeltaNet-2 importable.

We only reuse the pure-recurrent GDN-2 token mixer + its Triton chunk kernel.
The vendored package's __init__.py has been neutralized so importing
`lit_gpt.gdn2` does NOT pull in flash-attn / SWA / lightning.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_GDN_ROOT = os.path.normpath(os.path.join(_HERE, "..", "GatedDeltaNet-2"))
if _GDN_ROOT not in sys.path:
    sys.path.insert(0, _GDN_ROOT)

from lit_gpt.gdn2 import GatedDeltaNet2  # noqa: E402
from lit_gpt.gdn2_ops.chunk_gdn2 import chunk_gdn2  # noqa: E402

__all__ = ["GatedDeltaNet2", "chunk_gdn2", "_GDN_ROOT"]
