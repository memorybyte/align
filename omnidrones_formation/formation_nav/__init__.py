"""
FormationNav for OmniDrones: take-off -> formation -> waypoint navigation through static /
dynamic obstacles -> hold, plus the paper's FC-LSTM-FC MAPPO.

`env` (the Isaac Sim environment) is not imported here because it needs a running Isaac Sim
app; import `formation_nav.env` explicitly after `init_simulation_app`.
"""

from .core import (
    FORMATIONS,
    PHASE_FORM,
    PHASE_HOLD,
    PHASE_NAV,
    SCENARIOS,
    STAT_KEYS,
    FormationNavConfig,
    FormationNavCore,
    formation_template,
    procrustes_error,
)
from .mappo_lstm import MAPPOLSTM, MAPPOLSTMConfig
