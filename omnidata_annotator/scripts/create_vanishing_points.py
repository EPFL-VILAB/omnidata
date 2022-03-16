"""
  Name: create_vanishing_points.py
  Desc: Creates and saves vanishing point information for each image. The vanishing points
    are saved both as vanishing_points_image which are stored as (X,Y) coordinates on the 
    image, and also vanishing_points_gaussian_sphere which are stored as (X,Y,Z) coords 
    in camera space and lie on the unit sphere. 


  Requires (to be run):
    - generate_points.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import numpy as np
import os
import scipy
from scipy import io
from scipy import ndimage
import sys
import transforms3d

# Import from project
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from load_settings import settings
import io_utils
import nonblender_utils
from nonblender_utils import Profiler

basepath = settings.MODEL_PATH


def main():
    global logger
    logger = io_utils.create_logger(__name__)

    # Load points
    point_infos = io_utils.load_saved_points_of_interest(basepath)

    if settings.OVERRIDE_MATTERPORT_MODEL_ROTATION:
        floor_plan = scipy.misc.imread(io_utils.get_floorplan_file(basepath), mode='L')
        t_bb, scaling = nonblender_utils.get_bounding_box_for_floor_plan(floor_plan)
        rotate_degrees = nonblender_utils.get_rotation_from_bb_mat(t_bb)
    else:
        rotate_degrees = 0.0

    with Profiler("vanishing_point", logger=logger) as pflr:
        for point_number, point_info in enumerate(point_infos):
            for view_num, view_of_point in enumerate(point_info):
                
                image_vps, gaussian_sphere_vps = get_vanishing_points(
                    view_of_point,
                    settings.RESOLUTION,
                    -rotate_degrees)
                # for how they're stored: http://www.ipol.im/pub/pre/148/preprint.pdf
                view_of_point['vanishing_points_image'] = {
                    'x': image_vps[0],
                    'y': image_vps[1],
                    'z': image_vps[2]
                }
                view_of_point['vanishing_points_gaussian_sphere'] = {
                    'x': gaussian_sphere_vps[0],
                    'y': gaussian_sphere_vps[1],
                    'z': gaussian_sphere_vps[2]
                }
                view_of_point['model_rotation_degrees'] = rotate_degrees
                view_of_point['resolution'] = settings.RESOLUTION
                io_utils.resave_point(basepath, view_of_point['view_id'], view_of_point)


def rotate_world_to_cam(points, view_dict):
    cam_mat = nonblender_utils.get_camera_rot_matrix(view_dict, flip_xy=True)
    new_points = cam_mat.T.dot(points).T[:, :3]
    return new_points


def rotation_to_make_axes_well_defined(view_dict):
    ''' Rotates the world coords so that the -z direction of the camera 
        is within 45-degrees of the global +x axis '''
    axes_xyz = np.eye(3)
    apply_90_deg_rot_k_times = [
        transforms3d.axangles.axangle2mat(axes_xyz[-1], k * math.pi / 2)
        for k in range(4)]

    global_x = np.array([axes_xyz[0]]).T
    global_y = np.array([axes_xyz[1]]).T
    best = (180., "Nothing")
    for world_rot in apply_90_deg_rot_k_times:
        global_x_in_cam = rotate_world_to_cam(
            world_rot.dot(global_x), view_dict)
        global_y_in_cam = rotate_world_to_cam(
            world_rot.dot(global_y), view_dict)
        # Project onto camera's horizontal (xz) plane
        degrees_away_x = math.degrees(
            math.acos(np.dot(global_x_in_cam, -axes_xyz[2]))
        )
        degrees_away_y = math.degrees(
            math.acos(np.dot(global_y_in_cam, -axes_xyz[2]))
        )
        total_degrees_away = abs(degrees_away_y)  # + abs(degrees_away_y)
        best = min(best, (total_degrees_away, np.linalg.inv(world_rot)))  # python is neat
    return best[-1]


def get_camera_matrix(view_dict, flip_xy=False):
    position = view_dict['camera_location']
    rotation_euler = view_dict['camera_rotation_final']
    R = transforms3d.euler.euler2mat(*rotation_euler, axes='sxyz')
    camera_matrix = transforms3d.affines.compose(position, R, np.ones(3))

    if flip_xy:
        # For some reason the x and y are flipped in room layout
        temp = np.copy(camera_matrix[0, :])
        camera_matrix[0, :] = camera_matrix[1, :]
        camera_matrix[1, :] = -temp
    return camera_matrix


def get_vanishing_points(view_dict, resolution, rotate_degrees):
    # cam_mat = nonblender_utils.get_camera_matrix( view_dict, rotate_degrees )
    cam_mat = get_camera_matrix(view_dict, flip_xy=False)
    world_transformation = rotation_to_make_axes_well_defined(view_dict)
    cam_mat[:3, :3] = np.dot(world_transformation, cam_mat[:3, :3])
    R = cam_mat[:3, :3]

    # Get global axes in camera coords
    dist = 1
    compass_points = [(dist, 0, 0),
                      (0, dist, 0),
                      (0, 0, dist)]
    compass_points = [np.dot(np.linalg.inv(R), p)
                      for p in compass_points]

    # Find tangent direections for each axis
    tangent_directions = []
    for i, p in enumerate(compass_points):
        vp = p / np.linalg.norm(p)
        # if vp[2] < 0: # Tangent direction should be out of principal plane
        #     vp *= -1
        tangent_directions.append(vp)
    gaussian_sphere_vps = [tuple(p / np.linalg.norm(p)) for p in tangent_directions]

    K = get_K((resolution, resolution), view_dict['field_of_view_rads'])
    image_vps = [get_pix(0.1 * point - np.array([0, 0, 0.2]), K) for point in tangent_directions]
    # Project onto image plane
    return image_vps, gaussian_sphere_vps


def get_K(resolution, fov):
    focal_length = 1. / (2 * math.tan(fov / 2.))
    focal_length *= resolution[0]
    offset = resolution[0] / 2.
    K = np.array(
        ((focal_length, 0, offset),
         (0, focal_length, offset),
         (0, 0, 1)), dtype=np.float64)

    # Adjust for blender axes
    #     K[:,0] = -K[:,0]
    K[:, 1] = -K[:, 1]
    K[:, 2] = -K[:, 2]
    return K


def get_pix(point, K):
    ''' Project a 3d point from camera space back onto the image'''
    pix = np.dot(K, point)
    pix /= pix[2]
    return pix[0], pix[1]


if __name__ == "__main__":
    with Profiler("create_vanishing_points.py"):
        main()
