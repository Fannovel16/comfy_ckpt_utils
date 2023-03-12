is_model = lambda model_component: hasattr(model_component, "model")
is_clip = lambda model_component: hasattr(model_component, "cond_stage_model")
is_vae = lambda model_component: hasattr(model_component, "first_stage_model")

def get_state_dicts(*model_components):
    state_dicts = []
    for model_component in model_components:
        if is_model(model_component):
            model_component.unpatch_model()  # Prevent two-time patch
            model_component.patch_model()
            state_dicts.append(model_component.model.state_dict())
        if is_clip(model_component):
            model_component.patcher.unpatch_model()
            model_component.patcher.patch_model()
            state_dicts.append(model_component.cond_stage_model.state_dict())
        if is_vae(model_component):
            state_dicts.append(model_component.first_stage_model.state_dict())
    return tuple(state_dicts)


def unpatch_models(model=None, clip=None):
    if model is not None:
        model.unpatch_model()
    if clip is not None:
        clip.unpatch_model()

def weighted_sum(sd_A, sd_B, multipler, copy=True):
    if copy: sd_A = sd_A.copy()
    for key in sd_A:
        if 'model' in key and key in sd_B:
            sd_A[key] = sd_A[key] * (1 - multipler) + sd_B[key] * multipler
    return sd_A

def add_diff(sd_A, sd_B, sd_C, multipler, copy=True):
    if copy: sd_A = sd_A.copy()
    for key in sd_A:
        if 'model' in key and key in sd_B and key in sd_C:
            sd_A[key] = sd_A[key] + (sd_B[key] - sd_C[key]) * multipler
    return sd_A