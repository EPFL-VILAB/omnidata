"""
  Name: create_segmentation_2d_images.py
  Desc: Creates and saves 2D segmentation images for each point. We use normalized cuts for 
        segmenting images into perceptually similar groups.

  Requires (to be run):
    - generate_points.py
    - create_rgb_images.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from multiprocessing import Pool

# Import from project
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from load_settings import settings
import io_utils
from create_images_nonblender_utils import parallel_for_each_view
from nonblender_utils import Profiler
import nonblender_utils

import numpy as np
import scipy
import imageio
from skimage import data, io, segmentation, color
from skimage.future import graph
from PIL import Image
import traceback

basepath = settings.MODEL_PATH
TASK_NAME = 'segment_unsup2d'
INPUT_NAME, INPUT_TYPE = 'rgb', 'L'


def segment2d(img):
    labels1 = segmentation.felzenszwalb(
        img,
        scale=settings.SEGMENTATION_2D_SCALE,
        sigma=settings.SEGMENTATION_2D_BLUR)
    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(
        labels1, g,
        thresh=settings.SEGMENTATION_2D_CUT_THRESH,
        num_cuts=10,
        in_place=True,
        max_edge=settings.SEGMENTATION_2D_SELF_EDGE_WEIGHT)
    return labels2


def process_view(view_num_view_of_point):
    view_num, view_of_point = view_num_view_of_point

    rgb_fpath = io_utils.get_file_name_for(os.path.join(basepath, INPUT_NAME), view_of_point['point_uuid'], view_num,
                                        view_of_point['camera_uuid'], INPUT_NAME, settings.PREFERRED_IMG_EXT.lower())

    rgb_image = scipy.misc.imread(rgb_fpath)[:,:,:3]
    
    try:
        label_img = segment2d(rgb_image)
    except:
        traceback.print_exc()
        label_img = np.ones_like(rgb_image[..., 0])

    segment_fpath = io_utils.get_file_name_for(os.path.join(basepath, TASK_NAME), view_of_point['point_uuid'], view_num,
                                            view_of_point['camera_uuid'], TASK_NAME,
                                            settings.PREFERRED_IMG_EXT.lower())
    
    # scipy.misc.imsave(segment_fpath, label_img)
    imageio.imwrite(segment_fpath, label_img)


def main():
    io_utils.safe_make_output_folder(basepath, TASK_NAME)
    parallel_for_each_view(process_view)


if __name__ == "__main__":
    with Profiler(os.path.dirname(os.path.basename(__file__))):
        main()
