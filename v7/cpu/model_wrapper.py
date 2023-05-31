import time

class ModelWrapper:
    def __init__(self, model, name, size=640):
        self.model = model
        self.name = name
        self.size = size
        self.detection_time = -1
    
    def __repr__(self):
        return f"<Model {self.name}>"
    
    def __str__(self):
        return self.name
    
    def __call__(self, *args, **kwargs):
        t = time.time()
        result = self.model(*args, **kwargs)
        self.detection_time = time.time() - t
        return result
    
    def eval(self):
        self.model.eval()
