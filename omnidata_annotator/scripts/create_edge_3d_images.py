"""
  Name: create_edge_images.py
  Desc: Creates and saves edge images for each point. The edges are computed from the depth 
  zbuffer images by using a Canny edge detector.

  Requires (to be run):
    - generate_points.py
    - create_depth_zbuffer_images.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import scipy
from scipy import misc
import skimage
from skimage import filters
from skimage import io as io
from skimage import img_as_uint
import cv2
import numpy as np
import warnings

# Import from project
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from load_settings import settings
import io_utils
from nonblender_utils import Profiler
import nonblender_utils


basepath = settings.MODEL_PATH
TASK_NAME = 'edge_occlusion'
INPUT_NAME, INPUT_TYPE = 'depth_zbuffer', 'L'


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

                depth_fpath = io_utils.get_file_name_for(os.path.join(basepath, INPUT_NAME),
                                                         view_of_point['point_uuid'], view_of_point['view_id'], #view_num,
                                                         view_of_point['camera_uuid'], INPUT_NAME,
                                                         settings.PREFERRED_IMG_EXT.lower())
                depth_img = scipy.misc.imread(depth_fpath, INPUT_TYPE)

                mask = depth_img < 2 ** 16 - 500
                input_img = np.sqrt(depth_img) / np.sqrt(float(2 ** 16))

                def fsmooth(x):
                    return skimage.filters.gaussian(x, 1., mode='constant')

                smooth_with_function_and_mask(input_img, fsmooth, mask)
                edge_img = skimage.filters.sobel(input_img, mask=mask)

                edge_img = img_as_uint(edge_img)    # convert to 16-bit image
                edge_img = edge_img.astype(np.uint16)

                if settings.EDGE_3D_THRESH:
                    edge_img = 1.0 * (edge_img > settings.EDGE_3D_THRESH)

                edge_fpath = io_utils.get_file_name_for(os.path.join(basepath, TASK_NAME), view_of_point['point_uuid'],
                                                        view_of_point['view_id'], #view_num,
                                                        view_of_point['camera_uuid'], TASK_NAME,
                                                        settings.PREFERRED_IMG_EXT.lower())

                with warnings.catch_warnings():  # ignore 'low contrast image' warning
                    warnings.simplefilter('ignore', UserWarning)
                    cv2.imwrite(edge_fpath, edge_img)

                pflr.step('finished img {}/{}'.format(image_number, n_images))
                image_number += 1


def smooth_with_function_and_mask(image, function, mask):
    """Smooth an image with a linear function, ignoring masked pixels
    Parameters
    ----------
    image : array
        Image you want to smooth.
    function : callable
        A function that does image smoothing.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.
    Notes
    ------
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """
    bleed_over = function(mask.astype(float))
    masked_image = np.zeros(image.shape, image.dtype)
    masked_image[mask] = image[mask]
    smoothed_image = function(masked_image)
    output_image = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image


if __name__ == "__main__":
    with Profiler(os.path.dirname(os.path.basename(__file__))):
        main()
