"""
Geometry tools — spatial relationships in the model's activation space.

Group 17: 5 tools that work in the full native dimensionality (e.g. 2560
for Gemma 3-4B). All computations use actual angles in degrees, not lossy
2D projections. PCA projections are optional and flagged as lossy.

Importing this package triggers @mcp.tool() registration for all geometry tools.
"""

from . import direction_angles  # noqa: F401
from . import feature_dimensionality  # noqa: F401
from . import residual_trajectory  # noqa: F401
from . import subspace_decomposition  # noqa: F401
from . import token_space  # noqa: F401
