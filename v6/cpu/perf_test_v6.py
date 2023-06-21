from model_wrapper_v6 import ModelWrapper

import torch
import os
import sys
import platform
import GPUtil


sys.path.append(os.path.join('v6', 'yolov6_main'))
try:
    from yolov6.core.inferer import Inferer
except ModuleNotFoundError:
    print("Please run the file from the root of the repository.")
    exit(1)




@torch.no_grad()
def get_inferer(weights, size=640):
    return Inferer(os.path.join('..', '..', 'img', 'coco'), False, 0, weights, 0, os.path.join('data', 'coco.yaml'), size, False)

def load_models():
    models = []

    os.chdir(os.path.join('v6', 'yolov6_main')) 
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6n.pt')), 'yolov6n'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6s.pt')), 'yolov6s'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6m.pt')), 'yolov6m'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6l.pt')), 'yolov6l'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6n6.pt'), size=640), 'yolov6n6', size=640))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6s6.pt'), size=640), 'yolov6s6', size=640))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6m6.pt'), size=640), 'yolov6m6', size=640))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6l6.pt'), size=640), 'yolov6l6', size=640))
    os.chdir(os.path.join('..', '..'))

    return models

def perf_test(models, confidence=0.5):
    img_nb = len(os.listdir(os.path.join('img', 'coco')))
    print(f'\n\nCPU: {platform.processor()}')
    print(f'GPUs: {[gpu.name for gpu in GPUtil.getGPUs()]}')
    print(f'\n>>>>> YOLOv6 : Run inference on {img_nb} images <<<<<\n')
    os.chdir(os.path.join('v6', 'yolov6_main'))
    for model in models:
        r = model()
        print(r)
        print(f'{f"{model.name} " + f"({model.size}x{model.size})":>25} - {round(model.detection_time, 3):>7}s - {round(img_nb/model.detection_time, 3):>6} FPS')
        # result.save()
    os.chdir(os.path.join('..', '..'))

if __name__=="__main__":
    models = load_models()
    perf_test(models)