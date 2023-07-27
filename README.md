# YOLO_comparison

This repository allows to compare the inference time and the detection performance of different YOLO versions. I used all the pretrained models from [YOLOv5](https://github.com/ultralytics/yolov5), [YOLOv6](https://github.com/meituan/YOLOv6/tree/main) and [YOLOv7](https://github.com/WongKinYiu/yolov7).

The detection performance metrics are recall, precision and F1 score.

## Requirements

You can install all the requirements with the following command:

```bash
pip install -r requirements.txt
```

## Usage

You need to run the files `perf_test_*.py` to get the inference time and the detection performance of the different YOLO versions. You must run them from the root directory of this repository.

To get performance on a video file, you must provide the name of a video that is in the `vid` folder. To get the detection metrics, there must be a file in the same folder with the same name but with the extension `.txt` instead of `.mp4` and with the suffix `start` added at the end. Feel free to use the provided video and text files and to add your own following the same naming convention.

The ground truth file must be a text file with the same format as the ones provided in the `vid` folder. Be aware that the coordinates of the bounding boxes must be in the format `(left, top),(right, bottom)` and the the y axis is not inverted (the origin is at the bottom left corner).

This project has been designed to work with this [Unity project]() that generates the ground truth files automatically.


## Screenshots
![image](https://github.com/BigBaz54/YOLO_comparison/assets/96493391/240946a4-3daa-40e1-9f61-44a3cd20cbb2)

![image](https://github.com/BigBaz54/YOLO_comparison/assets/96493391/17b83b4d-bce1-49fa-919d-d6d70425670e)
