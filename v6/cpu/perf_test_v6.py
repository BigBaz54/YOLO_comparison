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
def get_inferer(weights, file_path, size=640):
    return Inferer(file_path, False, 0, weights, 0, os.path.join('data', 'coco.yaml'), size, False)

def load_models(file_path, size):
    models = []

    os.chdir(os.path.join('v6', 'yolov6_main'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6n.pt'), file_path, size), 'yolov6n'))
    # models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6s.pt'), file_path), 'yolov6s'))
    # models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6m.pt'), file_path), 'yolov6m'))
    # models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6l.pt'), file_path), 'yolov6l'))
    # models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6n6.pt'), file_path, size=640), 'yolov6n6', size=640))
    # models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6s6.pt'), file_path, size=640), 'yolov6s6', size=640))
    # models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6m6.pt'), file_path, size=640), 'yolov6m6', size=640))
    # models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6l6.pt'), file_path, size=640), 'yolov6l6', size=640))
    os.chdir(os.path.join('..', '..'))

    return models

def perf_test(file_path, size, confidence=0.5):
    models = load_models(file_path, size)

    # if file_path is a directory, count the number of images in it
    if os.path.isdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), file_path)):
        img_nb = len(os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), file_path)))
    else:
        img_nb = 1

    print(f'\n\nCPU: {platform.processor()}')
    print(f'GPUs: {[gpu.name for gpu in GPUtil.getGPUs()]}')
    print(f'\n>>>>> YOLOv6 : Run inference on {img_nb} images <<<<<\n')
    os.chdir(os.path.join('v6', 'yolov6_main'))
    for model in models:
        r = model(save_dir=os.path.join('..', '..', 'runs', 'v6', model.name), conf=confidence)
        print(f'{f"{model.name} " + f"({model.size}x{model.size})":>25} - {round(model.detection_time, 3):>7}s - {round(img_nb/model.detection_time, 3):>6} FPS')
    os.chdir(os.path.join('..', '..'))

def perf_test_vid(file_path, size, confidence=0.5):
    models = load_models(os.path.join('..', '..', file_path), size)

    print(f'\n\nCPU: {platform.processor()}')
    print(f'GPUs: {[gpu.name for gpu in GPUtil.getGPUs()]}')
    os.chdir(os.path.join('v6', 'yolov6_main'))
    for model in models:
        print(f"{model.name} is running...")
        detections = model(save_dir=os.path.join('..', '..', 'vid', 'results'), conf=confidence)
        print(f'{f"{model.name} " + f"({model.size}x{model.size})":>25} - {round(model.detection_time, 3):>7}s - {round(len(model.inferer.detections)/model.detection_time, 3):>6} FPS')
    os.chdir(os.path.join('..', '..'))

if __name__=="__main__":
    # perf_test(os.path.join('img', 'coco'), 640)
    perf_test_vid(os.path.join('vid', 'test_voiture2.mp4'), 640)