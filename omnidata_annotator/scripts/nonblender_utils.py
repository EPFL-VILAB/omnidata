"""
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from   load_settings import settings
from   profiler import Profiler 

import math
import numpy as np
import logging
import sys
import scipy.ndimage as ndi
from   scipy.ndimage import gaussian_filter
import skimage
from   skimage.feature._canny import smooth_with_function_and_mask
import time
import transforms3d
from   transforms3d import euler

# import trimesh
# from   trimesh.bounds import oriented_bounds

logger = settings.LOGGER

def canny_no_nonmax(image, sigma=3.0, mask=None):
    if mask is None:
        mask = np.ones(image.shape, dtype=bool)

    def fsmooth(x):
        return gaussian_filter(x, sigma, mode='constant')

    smoothed = smooth_with_function_and_mask(image, fsmooth, mask)
    magnitude = skimage.filters.sobel(smoothed, mask)
    return magnitude

def convert_cam_to_world( points, view_dict ):
    cam_mat = get_camera_matrix( view_dict )
    homogenized_points = homogenize( points )
    new_points = np.dot( homogenized_points, cam_mat.T )[:,:3].T

    # swap x, y
    new_points = np.stack( [new_points[1,:], new_points[0,:], new_points[2,:]] )
    return new_points

def convert_world_to_cam( points, view_dict ):
    # swap x, y
    new_points = np.stack( [points[1,:], points[0,:], points[2,:]] ).T
    cam_mat = get_camera_matrix( view_dict )
    homogenized_points = homogenize( new_points )
    new_points = np.dot( homogenized_points, np.linalg.inv(cam_mat).T )[:,:3]
    return new_points


def get_camera_matrix_viz( view_dict, flip_xy=False ):
    position = view_dict[ 'camera_location' ]
    rotation_euler = view_dict[ 'camera_rotation_final' ]
    R = transforms3d.euler.euler2mat( *rotation_euler, axes='sxyz' )
    camera_matrix = transforms3d.affines.compose(  position, R, np.ones(3) )
    
    if flip_xy:
        # For some reason the x and y are flipped in room layout
        temp = np.copy(camera_matrix[0,:])
        camera_matrix[0,:] = camera_matrix[1,:]
        camera_matrix[1,:] = -temp
    return camera_matrix

def get_camera_rot_matrix(view_dict, flip_xy=False):
    return get_camera_matrix_viz(view_dict, flip_xy=True)[:3, :3]

def rotate_world_to_cam( points, view_dict ):
    cam_mat = get_camera_rot_matrix( view_dict, flip_xy=True )
    new_points = cam_mat.T.dot(points).T[:,:3]
    return new_points



def get_camera_matrix( view_dict, rotate_global_degrees=0 ):
    position = view_dict[ 'camera_location' ]
    rotation_euler = view_dict[ 'camera_rotation_final' ]
    R = transforms3d.euler.euler2mat( *rotation_euler, axes='sxyz' )
    R = np.linalg.inv( 
        np.dot( 
            np.linalg.inv(R),
            rot_mat_about_z( math.radians( rotate_global_degrees ) ) ) )
#     R[:,:2] = -R[:,:2]
    return transforms3d.affines.compose(  position, R, np.ones(3) )

def get_K( resolution, fov ):
    focal_length = 1. / ( 2 * math.tan( fov / 2. ) ) 
    focal_length *= resolution[0]
    offset = resolution[0] /2.
    K = np.array(
        ((   focal_length,    0, offset),
        (    0  ,  focal_length, offset),
        (    0  ,  0,      1       )), dtype=np.float64)
    
    # Adjust for blender axes
#     K[:,0] = -K[:,0]
    K[:,1] = -K[:,1]
    K[:,2] = -K[:,2]
    return K


def get_bounding_box_for_floor_plan( floor_plan ):
    '''
        Args: 
            floor_plan: A numpy array containing the floor_plan image
        Returns:
            t_bb: the translation from bounding-box space to the points space
            scaling: A 3-tuple containing the extent of the bounding box
    '''
    # extract occupied points
    occupied = np.where( np.equal(floor_plan, 0) )
    points = list(zip(*occupied))
    t_bb, scaling = oriented_bounds( points ) 
    
    logger.info( "Global axis rotation: {} degrees".format( get_rotation_from_bb_mat( t_bb ) )  )
    return t_bb, scaling

def get_focal_length( fov_rads, resolution ):
    focal_length = 1. / ( 2 * math.tan( fov_rads / 2. ) ) 
    focal_length *= resolution
    return focal_length

def get_rotation_from_bb_mat( t_bb ):
    return math.degrees( math.asin( t_bb[0][0] ) )

def homogenize( M ):
    return np.concatenate( [M, np.ones( (M.shape[0],1) )], axis=1 )

def rot_mat_about_z( theta ):
    c,s =np.cos( theta ), np.sin( theta )
    return np.array( [[c,-s,0], [s, c, 0], [0,0,1]] )


def intersect_bbs( ranges1, ranges2 ):
    intersection = []
    for i, (minval1, maxval1) in enumerate( ranges1 ):
        minval2, maxval2 = ranges2[ i ]
        minval = max( minval1, minval2 )
        maxval = min( maxval1, maxval2 )
        intersection.append( (minval, maxval) )
    return intersection

def translate_range( ranges, shift ):
    translated_ranges = []
    for i, (minval, maxval) in enumerate(ranges):
        translated_ranges.append( 
                (minval + shift[i], maxval + shift[i])
            )
    return translated_ranges