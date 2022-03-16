"""
  Name: create_mask_valid.py
  Desc: Create valid masks for each image.
  
"""

import os
import sys
import numpy as np
from PIL import Image
# Import remaining packages
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from load_settings import settings

basepath = settings.MODEL_PATH
TASK_NAME = 'mask_valid'

def main():

    depthz_path = os.path.join(basepath, 'depth_zbuffer')

    for depth_img_file in os.listdir(depthz_path):
      depth_img = np.array(Image.open(os.path.join(depthz_path, depth_img_file)))
      mask_valid = 255 * (1 - 1 * (depth_img==65535))
      mask_valid = np.array(mask_valid, dtype=np.uint8)
      mask_valid_img = Image.fromarray(mask_valid)
      save_path = os.path.join(basepath, TASK_NAME, depth_img_file[:-17] + 'mask_valid.png')
      mask_valid_img.save(save_path)
      print(save_path)




if __name__ == "__main__":
  main()
