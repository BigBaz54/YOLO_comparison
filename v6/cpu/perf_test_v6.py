import torch
import os
import sys
import platform
import GPUtil

from model_wrapper_v6 import ModelWrapper
sys.path.append(os.path.join('.'))
from perf_detection import get_metrics


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
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6n.pt'), file_path, size), 'yolov6n', size))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6s.pt'), file_path, size), 'yolov6s', size))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6m.pt'), file_path), 'yolov6m'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6l.pt'), file_path), 'yolov6l'))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6n6.pt'), file_path, size), 'yolov6n6', size))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6s6.pt'), file_path, size), 'yolov6s6', size))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6m6.pt'), file_path, size), 'yolov6m6', size))
    models.append(ModelWrapper(get_inferer(os.path.join('..', 'models', 'yolov6l6.pt'), file_path, size), 'yolov6l6', size))
    os.chdir(os.path.join('..', '..'))

    return models

def perf_test_img(size, confidence=0.5):
    models = load_models(os.path.join('..', '..', os.path.join('img', 'coco')), size)

    # if file_path is a directory, count the number of images in it
    if os.path.isdir(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', os.path.join('img', 'coco')))):
        img_nb = len(os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', os.path.join('img', 'coco'))))
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

def perf_test_vid(video_name, size, confidence=0.5):
    models = load_models(os.path.join('..', '..', 'vid', video_name), size)

    print(f'\n\nCPU: {platform.processor()}')
    print(f'GPUs: {[gpu.name for gpu in GPUtil.getGPUs()]}')
    os.chdir(os.path.join('v6', 'yolov6_main'))
    for model in models:
        print(f"\n{model.name} is running...")
        detections = model(save_dir=os.path.join('..', '..', 'vid', 'results'), conf=confidence)
        max_dim = max(model.inferer.img_width, model.inferer.img_height)
        formatted_detections = []
        for detection in detections:
            frame_detections = []
            for d in detection:
                frame_detections.append({
                    'class_id': d[5],
                    'confidence': d[4],
                    'left': d[0]*(max_dim/model.inferer.img_width)/model.size,
                    'top': d[1]*(max_dim/model.inferer.img_height)/model.size,
                    'right': d[2]*(max_dim/model.inferer.img_width)/model.size,
                    'bottom': d[3]*(max_dim/model.inferer.img_height)/model.size
                })
            formatted_detections.append(frame_detections.copy())
        gt_file = os.path.join('..', '..', 'vid', video_name).replace('.mp4', 'start.txt')
        metrics = get_metrics(gt_file, formatted_detections)
        print(f'{f"{model.name} " + f"({model.size}x{model.size})":>25} - Recall: {round(metrics["recall"]*100, 2)}% - Precision: {round(metrics["precision"]*100, 2)}% - F1: {round(metrics["f1"]*100, 2)}% - {round(len(model.inferer.detections)/model.detection_time, 3):>6} FPS')
    os.chdir(os.path.join('..', '..'))

if __name__=="__main__":
    # perf_test_img(160)
    perf_test_vid('correctbbox.mp4', 640)