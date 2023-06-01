import torch
import os
import sys

sys.path.append(os.path.join('v6', 'yolov6-main'))
from yolov6.core.inferer import Inferer
os.chdir(os.path.join('v6', 'yolov6-main'))
@torch.no_grad()
def get_inferer():
    return Inferer(os.path.join('..', '..', 'img'), False, 0, os.path.join('..', 'models', 'yolov6s.pt'), 0, os.path.join('data', 'coco.yaml'), 640, False)

inferer = get_inferer()
inferer.infer(0.25, 0.45, None, False, 1000, os.path.join('..', '..', 'runs', 'v6'), False, True, False, False, False)
# infer.run(os.path.join('..', 'models', 'yolov6s.pt'), os.path.join('..', '..', 'img'), yaml=os.path.join('data', 'coco.yaml'))