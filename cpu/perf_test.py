import torch
import os
import cv2

from model_wrapper import ModelWrapper


def detect(model, img):
    # convert colors
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # convert to tensor
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    # # send to gpu
    # img = img.cuda()

    with torch.no_grad():
        # detect
        detections = model(img, augment=False)[0]

    # convert to numpy array
    detections = detections.cpu().numpy()
    return detections

def load_models():
    os.chdir(os.path.join('models', 'v5'))
    models = []
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True), 'yolov5s'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True), 'yolov5m'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True), 'yolov5l'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True), 'yolov5x'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5s6', pretrained=True), 'yolov5s6'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained=True), 'yolov5m6'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True), 'yolov5l6'))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True), 'yolov5x6'))
    for model in models:
        model.eval()
    os.chdir(os.path.join('..', '..'))
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