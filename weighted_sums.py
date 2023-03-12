from .utils import get_state_dicts, weighted_sum

class FullWeightedSum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelA": ("MODEL", ),
                "modelB": ("MODEL", ),
                "clipA": ("CLIP", ),
                "clipB": ("CLIP", ),
                "multipler": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "fws"
    CATEGORY = "checkpoint"

    def fws(self, modelA, modelB, clipA, clipB, multipler):
        modelA_sd, modelB_sd, clipA_sd, clipB_sd= get_state_dicts(modelA, modelB, clipA, clipB)
        result_model = modelA.clone()
        result_model.model.load_state_dict(weighted_sum(modelA_sd, modelB_sd, multipler))

        result_clip = clipA.clone()
        result_clip.cond_stage_model.load_state_dict(weighted_sum(clipA_sd, clipB_sd, multipler))
        
        return (result_model, result_clip)


class ModelWeightedSum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "modelA": ("MODEL", ),
                "modelB": ("MODEL", ),
                "multipler": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "mws"
    CATEGORY = "checkpoint"
    
    def mws(self, modelA, modelB, multipler):
        modelA_sd, modelB_sd= get_state_dicts(modelA, modelB)
        
        result_model = modelA.clone()
        result_model.model.load_state_dict(weighted_sum(modelA_sd, modelB_sd, multipler))
        return (result_model,)

class CLIPWeightedSum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clipA": ("CLIP", ),
                "clipB": ("CLIP", ),
                "multipler": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "cws"

    CATEGORY = "checkpoint"
    def cws(self, clipA, clipB, multipler):
        clipA_sd, clipB_sd = get_state_dicts(clipA, clipB)
        
        result_clip = clipA.clone()
        result_clip.cond_stage_model.load_state_dict(weighted_sum(clipA_sd, clipB_sd, multipler))
        return (result_clip,)