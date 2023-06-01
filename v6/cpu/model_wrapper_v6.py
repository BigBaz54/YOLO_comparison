import time
import torch
import os


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
    
    def __call__(self, save_dir=os.path.join('..', '..', 'runs', 'v6')):
        t = time.time()
        self.inferer.infer(0.25, 0.45, None, False, 1000, save_dir, False, True, False, False, False)
        self.detection_time = time.time() - t
