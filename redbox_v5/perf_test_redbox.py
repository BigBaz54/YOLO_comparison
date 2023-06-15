import sys
import os
sys.path.append(os.path.join('.'))
try:
    from redbox_v5.model_wrapper_redbox import ModelWrapper
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
    
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('redbox_v5', 'models', 'v5s160_fit_within.pt')), 'v5s160', size=160))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('redbox_v5', 'models', 'v5s320_fit_within.pt')), 'v5s320', size=320))
    models.append(ModelWrapper(torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('redbox_v5', 'models', 'v5s640.pt')), 'v5s640', size=640))

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

def perf_test_img(models):
    sizes = [160, 320, 640]

    imgs = [os.path.join('img', 'redbox', img) for img in os.listdir(os.path.join('img', 'redbox')) if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg')]
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

def get_nb_objects_evolution(video_name):
    nb_objects_changes = []
    with open(os.path.join('vid', f'{video_name[:-4]}.txt'), 'r') as f:
        for line in f:
            if 'plus' in line:
                nb_objects_changes.append((float(line.split(': ')[-1].strip().replace(',', '.')[:-1]), -1))
            else:
                nb_objects_changes.append((float(line.split(': ')[-1].strip().replace(',', '.')[:-1]), 1))
    nb_objects_evolution = []
    nb_objects = 0
    for i in range(len(nb_objects_changes)):
        nb_objects += nb_objects_changes[i][1]
        nb_objects_evolution.append((nb_objects_changes[i][0], nb_objects))
    # Remove duplicates and keeps the last one
    while (sorted(list(set([evo[0] for evo in nb_objects_evolution]))) != [evo[0] for evo in nb_objects_evolution]):
        for i in range(len(nb_objects_evolution)):
            if nb_objects_evolution[i][0] in [evo[0] for evo in nb_objects_evolution[i:]]:
                nb_objects_evolution.pop(i)
                break
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
    results = {}

    # Chosing available names for the results and creating the video writers
    for model in models:
        if os.path.exists(os.path.join('vid', 'results', f'{video_name[:-4]}_{model.name}.mp4')):
            n=1
            while os.path.exists(os.path.join('vid', 'results', f'{video_name[:-4]}_{model.name}_{n}.mp4')):
                n+=1
            results[model.name] = cv2.VideoWriter(os.path.join('vid', 'results', f'{video_name[:-4]}_{model.name}_{n}.mp4'), cv2.VideoWriter_fourcc(*'avc1'), video.get(cv2.CAP_PROP_FPS) , (int(img_width), int(img_height)))
        else:
            results[model.name] = cv2.VideoWriter(os.path.join('vid', 'results', f'{video_name[:-4]}_{model.name}.mp4'), cv2.VideoWriter_fourcc(*'avc1'), video.get(cv2.CAP_PROP_FPS) , (int(img_width), int(img_height)))
    
    # Initializing the variables
    frame_done = 0
    total_errors = {model.name: 0 for model in models}
    total_objects = 0
    time_step = frame_total/video_fps/1000
    time = 0.494677 # To sync with the start of the video

    # Starting the detection
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
            print(f'{model.name} - detections : {nb_detections}/{nb_objects}')

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
        print(f'Frame {frame_done}/{frame_total} - {time}s')
        time = round(time + time_step, 6)

    # Printing the stats
    print('\nStats :')
    for model in models:
        print(f'{model.name} - Accuracy : {round((1 - total_errors[model.name]/total_objects)*100, 2)}% - FPS : {round(min(frame_total, max_frames)/total_detection_time[model.name], 2)}')
        
    # Releasing the video and the writers
    video.release()
    for model in models:
        results[model.name].release()

if __name__=="__main__":
    models = load_models()
    # perf_test_img(models)
    perf_test_vid(models, 'cam05.mp4')
    # print(get_nb_objects_evolution('cam05.mp4'))