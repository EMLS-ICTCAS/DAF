def get_model(model_name, args):
    name = model_name.lower()
    if name == "daf":
        from models.daf import Learner
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Learner(args)
