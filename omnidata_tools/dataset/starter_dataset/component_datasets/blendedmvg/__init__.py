import os
from ..splits import get_splits, get_all_spaces
###############################################################################
# Split info:
# Exports:
#   e.g. flat_split_to_spaces['tiny-train'] -> List[str] building names
###############################################################################
split_file = os.path.join(os.path.dirname(__file__), 'train_val_test_blendedMVG.csv')
split_to_spaces = get_splits(split_file)
subset_to_spaces = {'debug': ['000000000000000000000009'], 'fullplus': get_all_spaces(split_to_spaces) }

