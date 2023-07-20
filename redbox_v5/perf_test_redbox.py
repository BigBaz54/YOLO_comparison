import sys
import os
sys.path.append(os.path.join('.'))
try:
    from redbox_v5.model_wrapper_redbox import ModelWrapper
except ModuleNotFoundError:
    print("Please run the file from the root of the repository.")
    exit(1)

from perf_detection import get_metrics
import torch
import os
import cv2
import platform
import GPUtil


def load_models(size):
    models = []
    
    if size == 'all':
        models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('redbox_v5', 'models', 'v5s160_fit_within.pt')), 'v5s160', 160))
        models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('redbox_v5', 'models', 'v5s320_fit_within.pt')), 'v5s320', 320))
        models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('redbox_v5', 'models', 'v5s640.pt')), 'v5s640', 640))
    else:
        models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('redbox_v5', 'models', 'v5s160_fit_within.pt')), 'v5s160', size))
        # models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('redbox_v5', 'models', 'v5s320_fit_within.pt')), 'v5s320', size))
        # models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('redbox_v5', 'models', 'v5s640.pt')), 'v5s640', size))

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

def perf_test_img(size):
    models = load_models(size)

    imgs = [os.path.join('img', 'redbox', img) for img in os.listdir(os.path.join('img', 'redbox')) if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg')]
    imgs_by_size = {}
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

def perf_test_vid(video_name, size, confidence=0.5, max_frames=None):
    models = load_models(size)
    all_detections = {model.name: [] for model in models}
    vid_path = os.path.join('vid', video_name)
    video = cv2.VideoCapture(vid_path)
    img_width, img_height = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    # print('FPS :', video_fps)

    if not os.path.exists(os.path.join('vid', 'results')):
        os.makedirs(os.path.join('vid', 'results'))
    output_names = {}

    # Chosing available names for the results
    for model in models:
        if os.path.exists(os.path.join('vid', 'results', f'{video_name[:-4]}_{model.name}.mp4')):
            n=1
            while os.path.exists(os.path.join('vid', 'results', f'{video_name[:-4]}_{model.name}_{n}.mp4')):
                n+=1
            output_names[model.name] = f'{video_name[:-4]}_{model.name}_{n}.mp4'
        else:
            output_names[model.name] = f'{video_name[:-4]}_{model.name}.mp4'
    
    # Creating the video writers
    results = {model.name: cv2.VideoWriter(os.path.join('vid', 'results', output_names[model.name]), cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (int(img_width), int(img_height))) for model in models}
    
    # Initializing the variables
    max_dim = max(img_width, img_height)
    frame_done = 0
    total_detection_time = {model.name: 0 for model in models}
    time_step = 1/video_fps
    time = 0 # To sync with the start of the video

    # Starting the detection
    print(f'\n\nCPU: {platform.processor()}')
    print(f'GPUs: {[gpu.name for gpu in GPUtil.getGPUs()]}')
    while (video.isOpened() and ((max_frames is None) or (frame_done < max_frames))):
        ret, frame = video.read()
        if not ret:
            break

        for model in models:
            # Detecting
            frame_preprocessed = cv2.cvtColor(image_resize(frame, model.size, model.size), cv2.COLOR_BGR2RGB)
            result = model(frame_preprocessed, size=model.size)
            detections = [pred for pred in result.pred[0] if pred[4] > confidence]
            this_frame_detections = []
            for detection in detections:
                det_l = detection.tolist()
                this_frame_detections.append({'class_id': int(det_l[5]), 'confidence': det_l[4], 'left': det_l[0]*(max_dim/img_width)/model.size, 'top': det_l[1]*(max_dim/img_height)/model.size, 'right': det_l[2]*(max_dim/img_width)/model.size, 'bottom': det_l[3]*(max_dim/img_height)/model.size})
            all_detections[model.name].append(this_frame_detections)

            # Updating stats
            total_detection_time[model.name] += model.detection_time

            # Drawing the boxes
            frame_with_boxes = frame.copy()
            for box in detections:
                left = (float(box[0])*max_dim)/model.size
                top = (float(box[1])*max_dim)/model.size
                right = (float(box[2])*max_dim)/model.size
                bottom = (float(box[3])*max_dim)/model.size
                cv2.rectangle(frame_with_boxes, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Writing the frame
            results[model.name].write(frame_with_boxes)

        # Updating the time
        frame_done += 1
        time = round(time + time_step, 6)

    # Printing the stats
    print(f'\nStats for {video_name}:')
    for model in models:
        metrics = get_metrics(vid_path.replace('.mp4', 'start.txt'), all_detections[model.name])
        print(f'{model.name} ({model.size}x{model.size}) - Recall: {round(metrics["recall"]*100, 2)}% - Precision: {round(metrics["precision"]*100, 2)}% - F1: {round(metrics["f1"]*100, 2)}% - FPS: {round(min(frame_total, max_frames or frame_total)/total_detection_time[model.name], 2)} - output: {output_names[model.name]}')
        
    # Releasing the video and the writers
    video.release()
    for model in models:
        results[model.name].release()

if __name__=="__main__":
    # perf_test_img(640)
    perf_test_vid('correctbbox.mp4', 'all')
    # perf_test_vid('correctbbox.mp4', 160)