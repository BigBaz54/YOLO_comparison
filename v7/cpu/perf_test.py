from model_wrapper import ModelWrapper

import torch
import os
import cv2
import sys
sys.path.append(os.path.join('v7', 'yolov7-main'))
import hubconf


def load_models():
    models = []

    models.append(ModelWrapper(hubconf.custom(os.path.join('v7', 'models', 'yolov7.pt')), 'yolov7'))

    for model in models:
        model.eval()
    return models

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
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


if __name__=="__main__":
    models = load_models()
    SIZE = 640

    imgs = [os.path.join('img', img) for img in os.listdir('img') if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg')]
    imgs_preprocessed = [image_preprocess(img, SIZE) for img in imgs]

    print(f'\n\n>>> Run inference on {len(imgs)} images <<<\n')
    for model in models:
        imgs_copy = [img.copy() for img in imgs_preprocessed]
        result = model(imgs_copy, size=SIZE)
        print(f'{model.name} - {round(model.detection_time, 3)}s - {round(len(imgs)/model.detection_time, 3)} FPS')
        # result.save()