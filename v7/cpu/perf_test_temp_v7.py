import os

import sys
sys.path.append(os.path.join('v7', 'yolov7_main'))
import hubconf

model = hubconf.custom(os.path.join('v7', 'models', 'yolov7.pt'))