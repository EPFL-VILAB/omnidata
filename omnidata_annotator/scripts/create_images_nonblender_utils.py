"""
  Name: create_images_nonblender_utils.py
  Desc: Contains utilities to parallelize processing
  Usage:
    for import only
"""

import logging
from multiprocessing import Pool
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from load_settings import settings
import io_utils
from io_utils import get_number_imgs
from profiler import Profiler

class KeyboardInterruptError(Exception): pass


def start_logging():
    '''Starts a logger'''
    #   global logger
    logger = io_utils.create_logger(__name__)
    basepath = settings.MODEL_PATH
    return logger, basepath


def parallizing_caller(process_fn_view_num_view_dict):
    """Wrap the process to catch keyboard interrupts"""
    try:
        process_fn, view_num, view_dict = process_fn_view_num_view_dict
        return process_fn((view_num, view_dict))
    except KeyboardInterrupt:
        raise KeyboardInterruptError()


def parallel_for_each_view(process_view):
    ''' Runs image generation given some render helper functions
    Args:
        stop_at: A 2-Tuple of (pt_idx, view_idx). If specified, running
            will cease (not cleaned up) at the given point/view
    '''
    logger, basepath = start_logging()
    with Profiler("Setup", logger) as prf:

        point_infos = io_utils.load_saved_points_of_interest(basepath)
        n_images = io_utils.get_number_imgs(point_infos)
        views_of_point = []

    with Profiler('Render', logger) as pflr:
        img_number = 0
        # Get a list of all the renders we will make
        for point_number, point_info in enumerate(point_infos):
            for view_number, view_dict in enumerate(point_info):
                views_of_point.append((process_view, view_dict['view_id'], view_dict))
                if settings.CREATE_PANOS:
                    break  # we only want to create 1 pano per camera

        # Execute the processing using a process pool
        p = Pool(settings.MAX_CONCURRENT_PROCESSES)
        for output in p.imap_unordered(parallizing_caller, views_of_point):
            img_number += 1
            pflr.step('finished img {}/{}'.format(img_number, n_images))

    return
