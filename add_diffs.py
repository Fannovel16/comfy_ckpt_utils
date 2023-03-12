from .utils import get_state_dicts, add_diff

class FullAddDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelA": ("MODEL", ),
                "modelB": ("MODEL", ),
                "modelC": ("MODEL", ),
                "clipA": ("CLIP", ),
                "clipB": ("CLIP", ),
                "clipC": ("CLIP", ),
                "multipler": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    RETURN_TYPES = ("MODEL","CLIP")
    FUNCTION = "mad"

    CATEGORY = "checkpoint"
    def mad(self, modelA, modelB, modelC, clipA, clipB, clipC, multipler):
        modelA_sd, modelB_sd, modelC_sd, clipA_sd, clipB_sd, clipC_sd = get_state_dicts(modelA, modelB, modelC, clipA, clipB, clipC)

        result_model = modelA.clone()
        result_model.model.load_state_dict(
            add_diff(modelA_sd, modelB_sd, modelC_sd, multipler)
        )

        result_clip = clipA.clone()
        result_clip.cond_stage_model.load_state_dict(
            add_diff(clipA_sd, clipB_sd, clipC_sd)
        )

        return (result_model,result_clip)

class ModelAddDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelA": ("MODEL", ),
                "modelB": ("MODEL", ),
                "modelC": ("MODEL", ),
                "multipler": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "mad"

    CATEGORY = "checkpoint"
    def mad(self, modelA, modelB, modelC, multipler):
        modelA_sd, modelB_sd, modelC_sd = get_state_dicts(modelA, modelB, modelC)
        result_model = modelA.clone()
        result_model.model.load_state_dict(
            add_diff(modelA_sd, modelB_sd, modelC_sd, multipler)
        )

        return (result_model,)

class CLIPAddDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clipA": ("CLIP", ),
                "clipB": ("CLIP", ),
                "clipC": ("CLIP", ),
                "multipler": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "mad"

    CATEGORY = "checkpoint"
    def mad(self, clipA, clipB, clipC, multipler):
        clipA_sd, clipB_sd, clipC_sd = get_state_dicts(clipA, clipB, clipC)
        result_clip = clipA.clone()
        result_clip.cond_stage_model.load_state_dict(
            add_diff(clipA_sd, clipB_sd, clipC_sd, multipler)
        )
        return (result_clip,)
