import argparse
from collections import OrderedDict

import torch


def _strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return OrderedDict(
            (k.replace("module.", "", 1), v) for k, v in state_dict.items()
        )
    return state_dict


def convert(input_path, output_path):
    legacy = torch.load(input_path, map_location="cpu")
    state_dict = OrderedDict()
    for key, value in _strip_module_prefix(legacy["model_state_dict"]).items():
        state_dict[f"model.{key}"] = value
    for key, value in _strip_module_prefix(legacy["modelq_state_dict"]).items():
        state_dict[f"modelq.{key}"] = value

    lightning_ckpt = {
        "state_dict": state_dict,
        "epoch": int(legacy.get("epoch_num", 0)),
        "global_step": 0,
        "optimizer_states": [],
        "lr_schedulers": [],
        "callbacks": {},
        "hyper_parameters": {},
    }
    torch.save(lightning_ckpt, output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert legacy SCA checkpoint weights.")
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
