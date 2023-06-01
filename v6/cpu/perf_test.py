from model_wrapper import ModelWrapper

import torch
import os
import sys

sys.path.append(os.path.join('v6', 'yolov6-main'))
from yolov6.core.inferer import Inferer


os.chdir(os.path.join('v6', 'yolov6-main'))

@torch.no_grad()
def get_inferer(weights, size=640):
    return Inferer(os.path.join('..', '..', 'img'), False, 0, weights, 0, os.path.join('data', 'coco.yaml'), size, False)

def load_models():
    models = []

    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6n.pt')), 'yolov6n'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6s.pt')), 'yolov6s'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6m.pt')), 'yolov6m'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6l.pt')), 'yolov6l'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6n6.pt'), size=640), 'yolov6n6', size=640))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6s6.pt'), size=640), 'yolov6s6', size=640))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6m6.pt'), size=640), 'yolov6m6', size=640))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6l6.pt'), size=640), 'yolov6l6', size=640))

    return models


if __name__=="__main__":
    models = load_models()
    img_nb = len(os.listdir(os.path.join("..", "..", "img")))
    print(f'\n\n>>> Run inference on {img_nb} images <<<\n')
    for model in models:
        model()
        print(f'{model.name} ({model.size}x{model.size}) - {round(model.detection_time, 3)}s - {round(img_nb/model.detection_time, 3)} FPS')
        # result.save()