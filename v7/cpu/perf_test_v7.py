from model_wrapper_v7 import ModelWrapper

import wget
import torch
import os
import cv2
import sys
sys.path.append(os.path.join('v7', 'yolov7_main'))
try:
    import hubconf
except ModuleNotFoundError:
    print("Please run the file from the root of the repository.")
    exit(1)
from models.experimental import attempt_load
import platform
import GPUtil


def load_models():
    models = []

    os.chdir(os.path.join('v7', 'models'))
    for name in ['yolov7.pt', 'yolov7x.pt', 'yolov7-e6.pt', 'yolov7-e6e.pt', 'yolov7-d6.pt', 'yolov7-w6.pt', 'yolov7-w6-pose.pt', 'yolov7-tiny.pt']:
        if not os.path.exists(name):
            print(f'\nDownloading {name}...')
            wget.download(f'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{name}', name)
    models.append(ModelWrapper(hubconf.custom('yolov7-tiny.pt'), 'yolov7-tiny', size=640))
    models.append(ModelWrapper(hubconf.custom('yolov7.pt'), 'yolov7'))
    models.append(ModelWrapper(hubconf.custom('yolov7x.pt'), 'yolov7x'))
    models.append(ModelWrapper(hubconf.custom('yolov7-e6.pt'), 'yolov7-e6', size=640))
    models.append(ModelWrapper(hubconf.custom('yolov7-e6e.pt'), 'yolov7-e6e', size=640))
    models.append(ModelWrapper(hubconf.custom('yolov7-d6.pt'), 'yolov7-d6', size=640))
    models.append(ModelWrapper(hubconf.custom('yolov7-w6.pt'), 'yolov7-w6', size=640))
    # models.append(ModelWrapper(attempt_load('yolov7-w6-pose.pt'), map_location=torch.device('cpu')), 'yolov7w6', size=640))
    os.chdir(os.path.join('..', '..'))

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

def perf_test(models):
    sizes = [160, 640, 1280]

    imgs = [os.path.join('img', 'coco', img) for img in os.listdir(os.path.join('img', 'coco')) if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg')]
    imgs_by_size = {}
    for size in sizes:
        imgs_preprocessed = [image_preprocess(img, size) for img in imgs]
        imgs_by_size[size] = imgs_preprocessed

    print(f'\n\nCPU: {platform.processor()}')
    print(f'GPUs: {[gpu.name for gpu in GPUtil.getGPUs()]}')
    print(f'\n>>>>> YOLOv7 : Run inference on {len(imgs)} images <<<<<\n')
    for model in models:
        imgs_copy = [img.copy() for img in imgs_by_size[model.size]]
        result = model(imgs_copy, size=model.size)
        print(f'{f"{model.name} " + f"({model.size}x{model.size})":>25} - {round(model.detection_time, 3):>7}s - {round(len(imgs)/model.detection_time, 3):>6} FPS')
        # result.save()

def get_nb_objects_evolution(video_name):
    nb_objects_changes = []
    with open(os.path.join('vid', f'{video_name[:-4]}.txt'), 'r') as f:
        for line in f.readlines()[1:]:
            if 'plus' in line:
                nb_objects_changes.append((float(line.split(': ')[-1].strip().replace(',', '.')[:-1]), -1))
            else:
                nb_objects_changes.append((float(line.split(': ')[-1].strip().replace(',', '.')[:-1]), 1))
    nb_objects_evolution = []
    nb_objects = 0
    changes_processed = 0
    while changes_processed < len(nb_objects_changes):
        current_time = nb_objects_changes[changes_processed][0]
        objects_change_at_current_time = [changes[1] for changes in nb_objects_changes if changes[0] == current_time]
        nb_objects += sum(objects_change_at_current_time)
        nb_objects_evolution.append((current_time, nb_objects))
        changes_processed += len(objects_change_at_current_time)
    
    return nb_objects_evolution

def perf_test_vid(models, video_name, confidence=0.5, max_frames=None):
    vid_path = os.path.join('vid', video_name)
    video = cv2.VideoCapture(vid_path)
    img_width, img_height = video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    print('FPS :', video_fps)

    # Get the truth from the txt file
    nb_objects_evolution = get_nb_objects_evolution(video_name)
    nb_objects = 0

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
    frame_done = 0
    total_errors = {model.name: 0 for model in models}
    total_detection_time = {model.name: 0 for model in models}
    total_objects = 0
    time_step = 1/video_fps
    time = 0 # To sync with the start of the video

    # Starting the detection
    for model in models:
        print(f"{model.name} is running...")
        
    while (video.isOpened() and ((max_frames is None) or (frame_done < max_frames))):
        # Getting the number of objects at the current time
        if (len(nb_objects_evolution) > 0) and (time > nb_objects_evolution[0][0]):
            nb_objects = nb_objects_evolution[0][1]
            nb_objects_evolution.pop(0)
        total_objects += nb_objects

        ret, frame = video.read()
        if not ret:
            break

        for model in models:
            # Detecting
            frame_preprocessed = cv2.cvtColor(image_resize(frame, model.size, model.size), cv2.COLOR_BGR2RGB)
            result = model(frame_preprocessed, size=model.size)
            detections = [pred for pred in result.pred[0] if pred[4] > confidence]
            nb_detections = len(detections)

            # Updating stats
            total_errors[model.name] += abs(nb_detections - nb_objects)
            # print(f'{model.name} - detections : {nb_detections}/{nb_objects}')
            total_detection_time[model.name] += model.detection_time

            # Drawing the boxes
            frame_with_boxes = frame.copy()
            for box in detections:
                max_dim = max(img_width, img_height)
                left = (float(box[0])*max_dim)/model.size
                top = (float(box[1])*max_dim)/model.size
                right = (float(box[2])*max_dim)/model.size
                bottom = (float(box[3])*max_dim)/model.size
                cv2.rectangle(frame_with_boxes, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

            # Writing the frame
            results[model.name].write(frame_with_boxes)

        # Updating the time
        frame_done += 1
        # print(f'Frame {frame_done}/{frame_total} - {time}s')
        time = round(time + time_step, 6)

    # Printing the stats
    print(f'\nStats for {video_name}:')
    for model in models:
        print(f'{model.name} - Accuracy: {round((1 - total_errors[model.name]/total_objects)*100, 2)}% - FPS: {round(min(frame_total, max_frames or frame_total)/total_detection_time[model.name], 2)} - output: {output_names[model.name]}')
        
    # Releasing the video and the writers
    video.release()
    for model in models:
        results[model.name].release()


if __name__=="__main__":
    models = load_models()
    # perf_test(models)
    perf_test_vid(models, 'cam07.mp4')