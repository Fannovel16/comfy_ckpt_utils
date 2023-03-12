try:
    import safetensors.torch
    safetensors_support = True
except:
    safetensors_support = False
import torch
import os
from pathlib import Path
import model_management

from .utils import get_state_dicts, unpatch_models, weighted_sum, get_diff, add_diff

class SaveCheckpoint:
    def __init__(self):
        self.output_dir = Path(os.path.dirname(
            os.path.realpath(__file__)), "../../models/checkpoints")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "my-model"}),
                "model": ("MODEL", ),
                "clip": ("CLIP", ),
                "vae": ("VAE", ),
                "save_as": ([".ckpt"] + [".safetensors"] if safetensors_support else [], {"default": ".ckpt"})
            }
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save_checkpoint"

    CATEGORY = "checkpoint"

    def save_checkpoint(self, name, model, clip, vae, save_as):
        model_sd, clip_sd, vae_sd = get_state_dicts(model, clip, vae)

        state_dict = {**model_sd}
        state_dict_parts = {
            "cond_stage_model": clip_sd,
            "first_stage_model": vae_sd
        }
        for part_key, part_value in state_dict_parts.items():
            for k, v in part_value.items():
                state_dict[f"{part_key}.{k}"] = v

        if save_as == ".ckpt":
            filename = f"{name}.ckpt"
            torch.save({"state_dict": state_dict}, Path(self.output_dir, filename).resolve())
        else:
            filename = f"{name}.safetensors"
            safetensors.torch.save_file({"state_dict": state_dict}, Path(self.output_dir, filename).resolve())

        del state_dict
        model_management.unload_model()
        unpatch_models(model, clip)

        return {"ui": {"model": filename}}





NODE_CLASS_MAPPINGS = {
    "SaveCheckpoint": SaveCheckpoint,

}
