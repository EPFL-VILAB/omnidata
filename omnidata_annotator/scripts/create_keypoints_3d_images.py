"""
  Name: create_keypoints_3d_images.py
  Desc: Creates and saves interest images for each point/view. An interest image contains 
    the 'soft' output of a keypoint detector, in this case NARF 
    (paper: https://pdfs.semanticscholar.org/e070/1662a370622a1cdbb7c6a83bbede3d0e6c23.pdf)
    Specifically, the interest image contains intensity values for each pixel before a 
    hard yes/no decision is made as to whether that pixel is a keypoint. This will probably
    provide richer signals to a learning model and that is why we use the interest image 
    instead of just keypoints. 

    The actual keypoint code is implemented in narf_interest_image.cpp and this file will
    run many versions concurrently through a threadpool. 

    Narf was chosen based on experiments run here: 
        http://www.pointclouds.org/blog/gsoc12/gballin/tests.php
    as well as based on the results of the original paper. It achieves generally better 
    repeatability compared to other methods along with a much faster runtime. 

  Requires (to be run):
    - generate_points.py
    - create_depth_zbuffer_images.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import math
from multiprocessing.dummy import Pool  # use threads
from subprocess import call
import scipy
from scipy import misc
import sys
import threading

# Import from project
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from load_settings import settings
import io_utils
from nonblender_utils import Profiler
import nonblender_utils

basepath = settings.MODEL_PATH
scriptpath = os.path.dirname(os.path.realpath(__file__))

TASK_NAME = 'keypoints3d'
INPUT_NAME, INPUT_TYPE = 'depth_zbuffer', 'L'


def process_view(view_num_view_of_point):
    try:
        view_num, view_of_point = view_num_view_of_point

        depth_fpath = io_utils.get_file_name_for(os.path.join(basepath, INPUT_NAME),
                                                view_of_point['point_uuid'], view_num,
                                                view_of_point['camera_uuid'], INPUT_NAME,
                                                settings.PREFERRED_IMG_EXT.lower())
        keypoint_fpath = io_utils.get_file_name_for(os.path.join(basepath, TASK_NAME),
                                                    view_of_point['point_uuid'], view_num,
                                                    view_of_point['camera_uuid'], TASK_NAME,
                                                    settings.PREFERRED_IMG_EXT.lower())
       
        
        focal_length = nonblender_utils.get_focal_length(
            view_of_point['field_of_view_rads'],
            view_of_point['resolution'])

        command = "{}/narf_interest_image.bin {} {} -d {} -f {} -r {} -s {} -v 0".format(
            scriptpath, depth_fpath, keypoint_fpath,
            settings.DEPTH_ZBUFFER_SENSITIVITY, focal_length, view_of_point['resolution'],
            settings.KEYPOINT_SUPPORT_SIZE)
        call(command.split(), shell=False, cwd=scriptpath)
        # lowpass_filter( keypoint_fpath )
    except Exception as e:
        print('!!!!!!!!!!!!!!!! ', str(e))

def lowpass_filter(input_file):
    img = cv2.imread(input_file, -1)
    img = cv2.GaussianBlur(img, (settings.KEYPOINT_BLUR_RADIUS, settings.KEYPOINT_BLUR_RADIUS), 0)
    cv2.imwrite(input_file, img)


def main():
    global logger
    logger = io_utils.create_logger(__name__)

    point_infos = io_utils.load_saved_points_of_interest(basepath)

    n_images = io_utils.get_number_imgs(point_infos)

    if not os.path.exists(os.path.join(basepath, TASK_NAME)):
        os.mkdir(os.path.join(basepath, TASK_NAME))

    p = Pool(settings.MAX_CONCURRENT_PROCESSES)
    image_number = 1
    views_of_point = []
    with Profiler("Render", logger=logger) as pflr:
        # Add all the view dicts to queue
        for point_number, point_info in enumerate(point_infos):
            for view_num, view_of_point in enumerate(point_info):
                point, view = view_of_point['point_uuid'], view_of_point['view_id']
                views_of_point.append((view_of_point['view_id'], view_of_point))
        # Run them from the threadpool
        for output in p.imap(process_view, views_of_point):
            pflr.step('finished img {}/{}'.format(image_number, n_images))
            image_number += 1


if __name__ == "__main__":
    with Profiler(os.path.dirname(os.path.basename(__file__))):
        main()
