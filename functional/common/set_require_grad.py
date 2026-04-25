

def set_requires_grad(model, flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)