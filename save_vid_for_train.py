import os
import cv2

def save_vid_for_train(video_path, save_path, result_name, sample_rate=1):
    # saves the frames of a video and the corresponding labels in the YOLOv5 format
    video = cv2.VideoCapture(video_path)
    nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ground_truth_frames = parse_gt(video_path.replace('.mp4', 'start.txt'))[-nb_frames:]
    frame_count = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            cv2.imwrite(os.path.join(save_path, f'{result_name}_{frame_count}.jpg'), frame)
            with open(os.path.join(save_path, f'{result_name}_{frame_count}.txt'), 'w') as f:
                for bbox in ground_truth_frames[frame_count]:
                    f.write(f'{bbox["class_id"]} {bbox["left"]} {bbox["top"]} {bbox["right"]} {bbox["bottom"]}\n')
        frame_count += 1
    video.release()

def parse_gt(gt_file):
    gt_by_frame = []
    current_frame = []
    with open(gt_file, 'r') as f:
        for line in f.readlines()[1:]:
            if "Temps" in line:
                gt_by_frame.append(current_frame.copy())
                current_frame = []
            else:
                if "cube" in line.lower():
                    class_id = 0
                elif "voiture" in line.lower():
                    class_id = 2
                else:
                    class_id = 0
                coords = line.split(':')[1]
                coords = coords.split(',')
                coords = [coords.replace('(', '').replace(')', '').replace(' ', '') for coords in coords]
                coords = [float(coords) for coords in coords]
                current_frame.append({
                    'class_id': class_id,
                    'left': coords[0],
                    'top': 1 - coords[1],
                    'right': coords[2],
                    'bottom': 1 - coords[3]
                })
    return gt_by_frame


if __name__ == '__main__':
    video_path = 'vid/cube.mp4'
    save_path = 'vid/train/'
    result_name = 'cube_full'
    save_vid_for_train(video_path, save_path, result_name, sample_rate=1)