def get_device(model):
  return next(model.parameters()).get_device()
