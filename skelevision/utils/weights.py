from collections import OrderedDict
from typing import TYPE_CHECKING
from typing import OrderedDict as T_ORDERED_DICT

if TYPE_CHECKING:
    from torch import Tensor


_LEGACY_FEATURES_TO_LAYER_CONVERSION = {
    "features.0": "layer1.0",
    "features.1": "layer1.1",
    ###
    "features.4": "layer2.0",
    "features.5": "layer2.1",
    ###
    "features.8": "layer3.0",
    "features.9": "layer3.1",
    ###
    "features.11": "layer4.0",
    "features.12": "layer4.1",
    ###
    "features.14": "layer5.0",
    "features.15": "layer5.1",
}


T_STATEDICT = T_ORDERED_DICT[str, "Tensor"]


def rename_legacy_backbone_names(state_dict: T_STATEDICT) -> T_STATEDICT:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("backbone.features"):
            fnum = k.split(".").pop(2)
            fstr = f"features.{fnum}"
            lstr = _LEGACY_FEATURES_TO_LAYER_CONVERSION[fstr]
            k = k.replace(fstr, lstr)
        new_state_dict[k] = v
    return new_state_dict


def rename_lightning_ckpt_state_dict_names(state_dict: T_STATEDICT) -> T_STATEDICT:
    PREFIX = "model."
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        assert k.startswith(PREFIX), k
        k = k.replace(PREFIX, "", 1)
        new_state_dict[k] = v
    return new_state_dict
