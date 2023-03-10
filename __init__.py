try:
    import safetensors.torch
    safetensors_support = True
except:
    safetensors_support = False
import torch


class SaveCheckpoint:
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
        model_state_dict = model.patch_model().state_dict()
        clip_state_dict = clip.patcher.patch_model().state_dict()
        vae_state_dict = vae.first_stage_model.state_dict()

        state_dict = {**model_state_dict, **clip_state_dict, **vae_state_dict}
        if save_as == ".ckpt":
            torch.save({"state_dict": state_dict}, f"{name}.ckpt")
        else:
            safetensors.torch.save_file({"state_dict": state_dict}, f"{name}.ckpt")

        clip.patcher.unpatch_model()
        model.unpatch_model()

NODE_CLASS_MAPPINGS = {
    "SaveCheckpoint": SaveCheckpoint
}
