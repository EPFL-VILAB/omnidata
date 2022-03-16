"""
  Name: create_segmentation_25d_images.py
  Desc: Creates and saves 2.5D segmentation for each image. Segmentation 2.5D uses the same algorithm 
        as 2D, but the labels are computed jointly from the edge image, depth image, and surface normals
        image. 2.5D segmentation incorporates information about the scene geometry that is not directly 
        present in the RGB image but that is readily inferred by humans.

  Requires (to be run):
    - generate_points.py
    - create_depth_zbuffer_images.py
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
from profiler import Profiler

import math
import numpy as np
import scipy
import skimage
from skimage import data, io, segmentation, color
from skimage.future import graph
import traceback

basepath = settings.MODEL_PATH
TASK_NAME = 'segment_unsup25d'
INPUT_NAME, INPUT_TYPE = 'depth_zbuffer', 'L'


def main():
    io_utils.safe_make_output_folder(basepath, TASK_NAME)
    parallel_for_each_view(process_view)


def segment25d(img):
    # Superpixel based on rgbd
    labels1 = segmentation.felzenszwalb(
        img,
        scale=settings.SEGMENTATION_25D_SCALE,
        sigma=0.,
        min_size=200)

    # Rag is based only on input channels
    g = rag_mean_color(
        img,
        labels1,
        mode='similarity')

    labels2 = graph.cut_normalized(
        labels1, g,
        thresh=settings.SEGMENTATION_25D_CUT_THRESH,
        num_cuts=100,
        in_place=True,
        max_edge=settings.SEGMENTATION_25D_SELF_EDGE_WEIGHT)
    return labels2


def process_view(view_num_view_of_point):
    def fsmooth1(x):
        return skimage.filters.gaussian(x, 1., mode='constant')

    def fsmooth2(x):
        return skimage.filters.gaussian(x, 2., mode='constant')

    view_num, view_of_point = view_num_view_of_point

    depth_fpath = io_utils.get_file_name_for(os.path.join(basepath, 'depth_zbuffer'), view_of_point['point_uuid'], view_num,
                                             view_of_point['camera_uuid'], 'depth_zbuffer', settings.PREFERRED_IMG_EXT.lower())
    depth_img = scipy.misc.imread(depth_fpath, "L")

    edge_fpath = io_utils.get_file_name_for(os.path.join(basepath, 'edge_occlusion'), view_of_point['point_uuid'], view_num,
                                            view_of_point['camera_uuid'], 'edge_occlusion', settings.PREFERRED_IMG_EXT.lower())

    normal_fpath = io_utils.get_file_name_for(os.path.join(basepath, 'normal'), view_of_point['point_uuid'], view_num,
                                              view_of_point['camera_uuid'], 'normal',
                                              settings.PREFERRED_IMG_EXT.lower())

    # Process the images and concat before combining
    img_depth = np.log(depth_img) / 16.
    img_edge = scipy.misc.imread(edge_fpath) / 255. / 255.
    img_normal = scipy.misc.imread(normal_fpath) / 255.  # / 2**16.

    mask = depth_img < 2 ** 16 - 2
    mask3 = np.stack([mask, mask, mask], axis=2)

    img_normal = smooth_with_function_and_mask(img_normal, fsmooth2, mask3)
    img_depth = smooth_with_function_and_mask(img_depth, fsmooth1, mask)

    img = np.concatenate([img_depth[:, :, np.newaxis], img_normal], axis=2)
    img = np.concatenate([img, img_edge[:, :, np.newaxis]], axis=2)
    img_ = np.copy(img)
    img_[:, :, 0] *= settings.SEGMENTATION_25D_DEPTH_WEIGHT
    img_[:, :, 1:4] *= settings.SEGMENTATION_25D_NORMAL_WEIGHT
    img_[:, :, 4] *= settings.SEGMENTATION_25D_EDGE_WEIGHT

    try:
        label_img = segment25d(img_)
    except:
        traceback.print_exc()
        label_img = np.ones_like(depth_img)

    segment_fpath = io_utils.get_file_name_for(os.path.join(basepath, TASK_NAME), view_of_point['point_uuid'], view_num,
                                               view_of_point['camera_uuid'], TASK_NAME,
                                               settings.PREFERRED_IMG_EXT.lower())

    scipy.misc.imsave(segment_fpath, label_img)


def rag_mean_color(image, labels, connectivity=2, mode='distance',
                   sigma=255.0):
    """Compute the Region Adjacency Graph using mean colors.
    Given an image and its initial segmentation, this method constructs the
    corresponding Region Adjacency Graph (RAG). Each node in the RAG
    represents a set of pixels within `image` with the same label in `labels`.
    The weight between two adjacent regions represents how similar or
    dissimilar two regions are depending on the `mode` parameter.
    Parameters
    ----------
    image : ndarray, shape(M, N, [..., P,] 3)
        Input image.
    labels : ndarray, shape(M, N, [..., P])
        The labelled image. This should have one dimension less than
        `image`. If `image` has dimensions `(M, N, 3)` `labels` should have
        dimensions `(M, N)`.
    connectivity : int, optional
        Pixels with a squared distance less than `connectivity` from each other
        are considered adjacent. It can range from 1 to `labels.ndim`. Its
        behavior is the same as `connectivity` parameter in
        ``scipy.ndimage.generate_binary_structure``.
    mode : {'distance', 'similarity'}, optional
        The strategy to assign edge weights.
            'distance' : The weight between two adjacent regions is the
            :math:`|c_1 - c_2|`, where :math:`c_1` and :math:`c_2` are the mean
            colors of the two regions. It represents the Euclidean distance in
            their average color.
            'similarity' : The weight between two adjacent is
            :math:`e^{-d^2/sigma}` where :math:`d=|c_1 - c_2|`, where
            :math:`c_1` and :math:`c_2` are the mean colors of the two regions.
            It represents how similar two regions are.
    sigma : float, optional
        Used for computation when `mode` is "similarity". It governs how
        close to each other two colors should be, for their corresponding edge
        weight to be significant. A very large value of `sigma` could make
        any two colors behave as though they were similar.
    Returns
    -------
    out : RAG
        The region adjacency graph.
    Examples
    --------
    >>> from skimage import data, segmentation
    >>> from skimage.future import graph
    >>> img = data.astronaut()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)
    References
    ----------
    .. [1] Alain Tremeau and Philippe Colantoni
           "Regions Adjacency Graph Applied To Color Image Segmentation"
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.11.5274
    """
    g = graph.RAG(labels, connectivity=connectivity)

    for n in g:
        g.node[n].update({'labels': [n],
                          'pixel count': 0,
                          'total color': np.zeros((image.shape[2]),
                                                  dtype=np.double)})

    for index in np.ndindex(labels.shape):
        current = labels[index]
        g.node[current]['pixel count'] += 1
        g.node[current]['total color'] += image[index]

    for n in g:
        g.node[n]['mean color'] = (g.node[n]['total color'] /
                                   g.node[n]['pixel count'])

    for x, y, d in g.edges(data=True):
        diff = g.node[x]['mean color'] - g.node[y]['mean color']
        diff = np.linalg.norm(diff)
        if mode == 'similarity':
            d['weight'] = math.e ** (-(diff ** 2) / sigma)
        elif mode == 'distance':
            d['weight'] = diff
        else:
            raise ValueError("The mode '%s' is not recognised" % mode)

    return g


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
