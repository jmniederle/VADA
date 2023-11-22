import warnings

from .util import *
try:
    from .resquiggle import resquiggle_read_normalized, seq_to_signal
except ModuleNotFoundError:
    warnings.warn("Failed to import Tombo, resquiggling not possible", ImportWarning)
