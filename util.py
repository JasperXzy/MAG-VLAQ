
import torch
import shutil
import logging
from collections import OrderedDict
from os.path import join

# import datasets_ws


def get_flops(model, input_shape=(480, 640)):
    """Return the FLOPs as a string, such as '22.33 GFLOPs'"""
    # assert len(input_shape) == 2, f"input_shape should have len==2, but it's {input_shape}"
    # module_info = torchscan.crawl_module(model, (3, input_shape[0], input_shape[1]))
    # output = torchscan.utils.format_info(module_info)
    # return re.findall("Floating Point Operations on forward: (.*)\n", output)[0]
    return None


def save_checkpoint(args, state, is_best, filename):
    model_path = join(args.save_dir, filename)
    torch.save(state, model_path)
    if is_best:
        shutil.copyfile(model_path, join(args.save_dir, "best_model.pth"))


def _strip_module_prefix(state_dict):
    """Strip 'module.' prefix added by DataParallel/DDP from state_dict keys."""
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = OrderedDict({k.replace('module.', '', 1): v for (k, v) in state_dict.items()})
    return state_dict


def resume_model(args, model):
    checkpoint = torch.load(args.resume, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        # The pre-trained models that we provide in the README do not have 'state_dict' in the keys as
        # the checkpoint is directly the state dict
        state_dict = checkpoint
    # Strip "module." prefix from DataParallel/DDP-saved checkpoints
    state_dict = _strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)
    return model


def resume_train(args, model, modelq, optimizer=None, strict=False):
    """Load model, optimizer, and other training parameters.
    Handles state_dicts saved from both plain and DDP-wrapped models."""
    logging.debug(f"Loading checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=args.device)
    start_epoch_num = checkpoint["epoch_num"]

    model_sd = _strip_module_prefix(checkpoint["model_state_dict"])
    modelq_sd = _strip_module_prefix(checkpoint["modelq_state_dict"])
    model.load_state_dict(model_sd, strict=strict)
    modelq.load_state_dict(modelq_sd, strict=strict)

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    best_r5 = checkpoint["best_r5"]
    not_improved_num = checkpoint["not_improved_num"]
    logging.debug(f"Loaded checkpoint: start_epoch_num = {start_epoch_num}, "
                  f"current_best_R@5 = {best_r5:.1f}")
    if args.resume.endswith("last_model.pth"):  # Copy best model to current save_dir
        shutil.copy(args.resume.replace("last_model.pth", "best_model.pth"), args.save_dir)
    return model, modelq, optimizer, best_r5, start_epoch_num, not_improved_num
