import sys
import os
sys.path.append(os.path.join('.'))
try:
    from v5.cpu.model_wrapper_v5 import ModelWrapper
except ModuleNotFoundError:
    print("Please run the file from the root of the repository.")
    exit(1)


import torch
import os
import cv2
import platform
import GPUtil


def load_models():
    models = []
    
    os.chdir(os.path.join('v5', 'models'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True), 'yolov5n'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True), 'yolov5s'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True), 'yolov5m'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True), 'yolov5l'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True), 'yolov5x'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5n6', pretrained=True), 'yolov5n6', size=640))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True), 'yolov5s6', size=640))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained=True), 'yolov5m6', size=640))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True), 'yolov5l6', size=640))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True), 'yolov5x6', size=640))
    os.chdir(os.path.join('..', '..'))

    for model in models:
        model.eval()
    return models

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image()
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def image_preprocess(image, target_size):
    img = cv2.imread(image)
    img = image_resize(img, target_size, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def perf_test(models):
    sizes = [640, 1280]

    imgs = [os.path.join('img', 'coco', img) for img in os.listdir(os.path.join('img', 'coco')) if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg')]
    imgs_by_size = {}
    for size in sizes:
        imgs_preprocessed = [image_preprocess(img, size) for img in imgs]
        imgs_by_size[size] = imgs_preprocessed
    print(f'\n\nCPU: {platform.processor()}')
    print(f'GPUs: {[gpu.name for gpu in GPUtil.getGPUs()]}')
    print(f'\n>>>>> YOLOv5 : Run inference on {len(imgs)} images <<<<<\n')
    for model in models:
        imgs_copy = [img.copy() for img in imgs_by_size[model.size]]
        result = model(imgs_copy, size=model.size)
        print(f'{f"{model.name} " + f"({model.size}x{model.size})":>25} - {round(model.detection_time, 3):>7}s - {round(len(imgs)/model.detection_time, 3):>6} FPS')
        # result.save()

if __name__=="__main__":
    models = load_models()
    perf_test(models)