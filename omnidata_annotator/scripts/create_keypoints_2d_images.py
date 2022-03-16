"""
  Name: create_keypoints_2d_images.py
  Desc: Creates and saves interest images for each point/view. 2D keypoint detection 
  encourages the network to identify locally important regions of an image, We use the 
  output of SURF (before nonmax suppression) as our ground-truth.

  Requires (to be run):
    - generate_points.py
    - create_rgb_images.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
from multiprocessing.dummy import Pool  # use threads
from subprocess import call
import scipy
from scipy import misc
from skimage import io as io
import sys
import threading

import skimage.feature.blob as skblob

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_laplace
import itertools as itt
import math
from math import sqrt, hypot, log

from PIL import Image, ImageFile
from skimage.feature.util import img_as_float
from skimage.transform import integral_image
from skimage.feature._hessian_det_appx import _hessian_matrix_det
from skimage.feature.peak import peak_local_max
from skimage import img_as_uint
import warnings

# Import from project
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from load_settings import settings
import io_utils
from nonblender_utils import Profiler
import nonblender_utils

basepath = settings.MODEL_PATH
scriptpath = os.path.dirname(os.path.realpath(__file__))

TASK_NAME = 'keypoints2d'
INPUT_NAME, INPUT_TYPE = 'rgb', 'L'

# ImageFile.LOAD_TRUNCATED_IMAGES = True


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

    image_number = 1
    with Profiler("Render", logger=logger) as pflr:
        # Add all the view dicts to queue
        for point_number, point_info in enumerate(point_infos):
            for view_num, view_of_point in enumerate(point_info):

                rgb_fpath = io_utils.get_file_name_for(os.path.join(basepath, INPUT_NAME),
                                                       view_of_point['point_uuid'], view_of_point['view_id'], #view_num,
                                                       view_of_point['camera_uuid'], INPUT_NAME,
                                                       settings.PREFERRED_IMG_EXT.lower())

                keypoint_fpath = io_utils.get_file_name_for(os.path.join(basepath, TASK_NAME),
                                                            view_of_point['point_uuid'], view_of_point['view_id'], #view_num,
                                                            view_of_point['camera_uuid'], TASK_NAME.replace("_", ""),
                                                            settings.PREFERRED_IMG_EXT.lower())

                rgb_img = scipy.misc.imread(rgb_fpath, INPUT_TYPE) / 255.
                

                keypoint_img = _blob_doh(rgb_img)
                keypoint_img = img_as_uint(keypoint_img)  # convert to 16-bit image
                keypoint_img = keypoint_img.astype(np.uint16)

                with warnings.catch_warnings():  # ignore 'low contrast image' warning
                    warnings.simplefilter('ignore', UserWarning)
                    cv2.imwrite(keypoint_fpath, keypoint_img)

                pflr.step('finished img {}/{}'.format(image_number, n_images))
                image_number += 1


def _blob_doh(image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01,
              overlap=.5, log_scale=False):
    """Finds blobs in the given grayscale image.
    Blobs are found using the Determinant of Hessian method [1]_. For each blob
    found, the method returns its coordinates and the standard deviation
    of the Gaussian Kernel used for the Hessian matrix whose determinant
    detected the blob. Determinant of Hessians is approximated using [2]_.
    Parameters
    ----------
    image : ndarray
        Input grayscale image.Blobs can either be light on dark or vice versa.
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this low to detect smaller blobs.
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel used to compute
        Hessian matrix. Keep this high to detect larger blobs.
    num_sigma : int, optional
        The number of intermediate values of standard deviations to consider
        between `min_sigma` and `max_sigma`.
    threshold : float, optional.
        The absolute lower bound for scale space maxima. Local maxima smaller
        than thresh are ignored. Reduce this to detect less prominent blobs.
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a
        fraction greater than `threshold`, the smaller blob is eliminated.
    log_scale : bool, optional
        If set intermediate values of standard deviations are interpolated
        using a logarithmic scale to the base `10`. If not, linear
        interpolation is used.
    Returns
    -------
    A : (n, 3) ndarray
        A 2d array with each row representing 3 values, ``(y,x,sigma)``
        where ``(y,x)`` are coordinates of the blob and ``sigma`` is the
        standard deviation of the Gaussian kernel of the Hessian Matrix whose
        determinant detected the blob.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Blob_detection#The_determinant_of_the_Hessian
    .. [2] Herbert Bay, Andreas Ess, Tinne Tuytelaars, Luc Van Gool,
           "SURF: Speeded Up Robust Features"
           ftp://ftp.vision.ee.ethz.ch/publications/articles/eth_biwi_00517.pdf
    Examples
    --------
    >>> from skimage import data, feature
    >>> img = data.coins()
    >>> feature.blob_doh(img)
    array([[ 270.        ,  363.        ,   30.        ],
           [ 265.        ,  113.        ,   23.55555556],
           [ 262.        ,  243.        ,   23.55555556],
           [ 260.        ,  173.        ,   30.        ],
           [ 197.        ,  153.        ,   20.33333333],
           [ 197.        ,   44.        ,   20.33333333],
           [ 195.        ,  100.        ,   23.55555556],
           [ 193.        ,  275.        ,   23.55555556],
           [ 192.        ,  212.        ,   23.55555556],
           [ 185.        ,  348.        ,   30.        ],
           [ 156.        ,  302.        ,   30.        ],
           [ 126.        ,  153.        ,   20.33333333],
           [ 126.        ,  101.        ,   20.33333333],
           [ 124.        ,  336.        ,   20.33333333],
           [ 123.        ,  205.        ,   20.33333333],
           [ 123.        ,   44.        ,   23.55555556],
           [ 121.        ,  271.        ,   30.        ]])
    Notes
    -----
    The radius of each blob is approximately `sigma`.
    Computation of Determinant of Hessians is independent of the standard
    deviation. Therefore detecting larger blobs won't take more time. In
    methods line :py:meth:`blob_dog` and :py:meth:`blob_log` the computation
    of Gaussians for larger `sigma` takes more time. The downside is that
    this method can't be used for detecting blobs of radius less than `3px`
    due to the box filters used in the approximation of Hessian Determinant.
    """

    # skblob.assert_nD(image, 2)

    image = img_as_float(image)
    image = integral_image(image)

    if log_scale:
        start, stop = log(min_sigma, 10), log(max_sigma, 10)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    hessian_images = [_hessian_matrix_det(image.astype(np.float), s) for s in sigma_list]
    image_cube = np.dstack(hessian_images)
    return np.max(image_cube, axis=2)




if __name__ == "__main__":
    with Profiler(os.path.dirname(os.path.basename(__file__))):
        main()
