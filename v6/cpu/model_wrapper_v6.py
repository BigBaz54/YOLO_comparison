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
    
    def __call__(self, conf=0.5, iou=0.45, save_dir=None, save_img=True):
        t1 = time.time()
        self.inferer.infer(conf, iou, None, False, 1000, save_dir, False, save_img, False, False, False)
        self.detection_time = time.time() - t1
        return self.inferer.detections
