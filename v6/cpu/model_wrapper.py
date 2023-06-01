import time
import torch


class ModelWrapper:

    @torch.no_grad()
    def __init__(self, inferer, name, size=640):
        self.inferer = inferer
        self.name = name
        self.size = size
        self.detection_time = -1
    
    def __repr__(self):
        return f"<Model {self.name}>"
    
    def __str__(self):
        return self.name
    
    def __call__(self, *args, **kwargs):
        t = time.time()
        self.inferer.run(*args, **kwargs)
        self.detection_time = time.time() - t
