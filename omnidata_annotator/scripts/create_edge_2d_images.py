"""
  Name: create_edge_2d_images.py
  Desc: Creates and saves 2D edge images for each point. The edges are computed from the RGB
    images by using a Canny edge detector.

  Requires (to be run):
    - generate_points.py
    - create_rgb_images.py 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import scipy
from PIL import Image
import numpy as np
from skimage import feature as feature
from skimage import io as io
from skimage import img_as_uint
import cv2
import warnings

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from load_settings import settings
import io_utils
from nonblender_utils import Profiler
import nonblender_utils

basepath = settings.MODEL_PATH
TASK_NAME = 'edge_texture'
INPUT_NAME, INPUT_TYPE = 'rgb', 'L'


def main():
    global logger
    logger = io_utils.create_logger(__name__)

    point_infos = io_utils.load_saved_points_of_interest(basepath)

    n_images = io_utils.get_number_imgs(point_infos)

    if not os.path.exists(os.path.join(basepath, TASK_NAME)):
        os.mkdir(os.path.join(basepath, TASK_NAME))

    image_number = 1
    with Profiler("Render", logger=logger) as pflr:
        for point_number, point_info in enumerate(point_infos):
            for view_num, view_of_point in enumerate(point_info):

                rgb_img = io_utils.get_file_name_for(os.path.join(basepath, INPUT_NAME), view_of_point['point_uuid'],
                                                     view_of_point['view_id'], #view_num
                                                     view_of_point['camera_uuid'], INPUT_NAME,
                                                     settings.PREFERRED_IMG_EXT.lower())

                rgb_img = scipy.misc.imread(rgb_img, INPUT_TYPE) / 255.

                if settings.CANNY_RGB_MIN_THRESH and settings.CANNY_RGB_MAX_THRESH:
                    edge_img = feature.canny(
                        rgb_img,
                        sigma=settings.CANNY_RGB_BLUR_SIGMA,
                        low_threshold=settings.CANNY_RGB_MIN_THRESH,
                        high_threshold=settings.CANNY_RGB_MAX_THRESH,
                        use_quantiles=settings.CANNY_RGB_USE_QUANTILES)
                else:
                    edge_img = nonblender_utils.canny_no_nonmax(
                        rgb_img,
                        sigma=settings.CANNY_RGB_BLUR_SIGMA)
                edge_fpath = io_utils.get_file_name_for(os.path.join(basepath, TASK_NAME), view_of_point['point_uuid'],
                                                        view_of_point['view_id'], #view_num
                                                        view_of_point['camera_uuid'], TASK_NAME,
                                                        settings.PREFERRED_IMG_EXT.lower())

                edge_img = img_as_uint(edge_img)  # convert to 16-bit image
                edge_img = edge_img.astype(np.uint16)

                with warnings.catch_warnings():  # ignore 'low contrast image' warning
                    warnings.simplefilter('ignore', UserWarning)
                    cv2.imwrite(edge_fpath, edge_img)

                pflr.step('finished img {}/{}'.format(image_number, n_images))
                image_number += 1


if __name__ == "__main__":
    with Profiler(os.path.dirname(os.path.basename(__file__))):
        main()
