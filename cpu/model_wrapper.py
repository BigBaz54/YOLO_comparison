class ModelWrapper:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.prediction_time = -1
    
    def __repr__(self):
        return f"<Model {self.name}>"
    
    def __str__(self):
        return self.name
    
    def __call__(self, *args, **kwargs):
        time = time.time()
        result = self.model(*args, **kwargs)
        self.prediction_time = time.time() - time
        return result
