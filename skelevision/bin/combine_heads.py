from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import torch
import yaml

from skelevision.utils.paths import resolve_path
from skelevision.utils.weights import T_STATEDICT
from skelevision.utils.weights import rename_legacy_backbone_names
from skelevision.utils.weights import rename_lightning_ckpt_state_dict_names


def _is_batch_norm_param(key: str) -> bool:
    return (
        key.endswith(".running_mean")
        or key.endswith(".running_var")
        or key.endswith(".num_batches_tracked")
    )


def load_state_dict(cfg_path: Path) -> T_STATEDICT:
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    assert cfg["MODEL_BUILDER"]["FREEZE_BACKBONE"] is True
    weights_path = resolve_path(cfg["MODEL_BUILDER"]["WEIGHTS_PATH"])
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))

    if weights_path.suffix == ".ckpt":
        state_dict = state_dict.pop("state_dict")
        return rename_lightning_ckpt_state_dict_names(state_dict)
    else:
        return rename_legacy_backbone_names(state_dict)


def cli() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("pretrained_keypoints_dir", type=Path)
    return parser


def main():
    args = cli().parse_args()

    new_state_dict = OrderedDict()
    rpn_state_dict = load_state_dict(args.pretrained_keypoints_dir / "tuning.yaml")
    kpt_state_dict = load_state_dict(args.pretrained_keypoints_dir / "model.yaml")

    # 1. Add RPN head weights
    while len(rpn_state_dict):
        k, v = rpn_state_dict.popitem(last=False)
        if k.startswith("backbone."):
            v_ = kpt_state_dict.pop(k)
            if _is_batch_norm_param(k):
                v = v_  # keep updated batchnorm parameter
            else:
                assert torch.all(v == v_), k  # ensure same backbone weights
        assert k not in kpt_state_dict, k
        new_state_dict[k] = v

        # 1.5 Add missing num_batches_tracked parameter
        num_batches_tracked_key = k.replace(".running_var", ".num_batches_tracked")
        if (
            k.endswith(".running_var")
            and (not k.startswith("rpn_head"))
            and (num_batches_tracked_key not in rpn_state_dict)
        ):
            new_state_dict[num_batches_tracked_key] = kpt_state_dict.pop(
                num_batches_tracked_key
            )

    # 2. Add keypoint head weights
    while len(kpt_state_dict):
        k, v = kpt_state_dict.popitem(last=False)
        assert k.startswith("keypoint_head."), k
        new_state_dict[k] = v

    output_path = args.pretrained_keypoints_dir / "combined_heads.pth"
    torch.save(new_state_dict, output_path)
    print(f"Saved at {output_path}")


if __name__ == "__main__":
    main()
