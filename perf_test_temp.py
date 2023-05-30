
import sys
sys.path.append('yolov7')
import os
import hubconf

model = hubconf.custom(os.path.join('models', 'v7', 'yolov7.pt'))