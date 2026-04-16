import logging
from collections import OrderedDict

import torch


def get_flops(model, input_shape=(480, 640)):
    return None


def _strip_module_prefix(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        return OrderedDict(
            (k.replace("module.", "", 1), v) for k, v in state_dict.items()
        )
    return state_dict


def resume_model(args, model, strict=True):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=strict)
    logging.debug("Loaded model checkpoint: %s", args.resume)
    return model
