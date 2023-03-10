try:
    import safetensors.torch
    safetensors_support = True
except:
    safetensors_support = False
import torch
import os

class SaveCheckpoint:
    def __init__(self):
        self.output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models/checkpoint")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "my-merged-model"}),
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
        model_state_dict = model.model.state_dict() #Yes. "model.model"
        clip_state_dict = clip.patcher.patch_model().state_dict()
        vae_state_dict = vae.first_stage_model.state_dict()

        state_dict = {**model_state_dict, **clip_state_dict, **vae_state_dict}
        if save_as == ".ckpt":
            filename = f"{name}.ckpt"
            torch.save({"state_dict": state_dict}, os.path.join(self.output_dir, filename))
        else:
            filename = f"{name}.safetensors"
            safetensors.torch.save_file({"state_dict": state_dict}, os.path.join(self.output_dir, filename))

        clip.patcher.unpatch_model()
        model.unpatch_model()
        return { "ui": { "model": filename} }

NODE_CLASS_MAPPINGS = {
    "SaveCheckpoint": SaveCheckpoint
}
