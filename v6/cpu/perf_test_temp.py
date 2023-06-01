import os

import sys
sys.path.append(os.path.join('v6', 'yolov6-main'))
import tools.infer as infer
os.chdir(os.path.join('v6', 'yolov6-main'))
infer.run(os.path.join('..', 'models', 'yolov6s.pt'), os.path.join('..', '..', 'img'), yaml=os.path.join('data', 'coco.yaml'))