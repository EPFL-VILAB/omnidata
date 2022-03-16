"""
  Name: generate_points.py

  Desc: Selects points that have at least a given number of views and saves information useful 
        for loading them.

"""

# Import these two first so that we can import other packages
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from load_settings import settings
from generate_camera_poses import sample_camera_locations_building, sample_camera_quaternion, sample_camera_locations_object
import io_utils
from utils import Profiler, create_empty

# Import remaining packages
import argparse
import bpy
import bpy_extras.mesh_utils
from bpy_extras.object_utils import world_to_camera_view
import bmesh
import copy
from collections import defaultdict
import glob
import itertools
import json
import math
from mathutils import Vector, Euler
import networkx as nx
import numpy as np
import pprint
import random
import time
import utils
import uuid
import csv

# from pathos.multiprocessing import ProcessPool as Pool
from multiprocessing import Pool

utils.set_random_seed()
pp = pprint.PrettyPrinter(indent=4)

basepath = settings.MODEL_PATH


TASK_NAME = 'point_info'


def delete_all_objects_in_context():
    """ Selects all objects in context scene and deletes them. """
    for obj in bpy.context.scene.objects:
        obj.select = True
        bpy.context.scene.objects.unlink(obj)
        bpy.data.objects.remove(obj)
    bpy.ops.object.delete()


def main():
    global args, logger, summary_cache
    summary_cache = None

    delete_all_objects_in_context()

    # Load the model
    model = io_utils.import_mesh(basepath)
    bbox_corners = [model.matrix_world * Vector(corner) for corner in model.bound_box]

    if settings.POINT_TYPE == 'SWEEP':
        point_info_path = os.path.join(basepath, 'pano', TASK_NAME)
    else:
        point_info_path = os.path.join(basepath, TASK_NAME)

    if os.path.isdir(point_info_path):
        shutil.rmtree(point_info_path)
    else:
        os.makedirs(point_info_path)

    # Get camera poses 
    if settings.GENERATE_CAMERAS:
        camera_poses = sample_camera_poses(model, bbox_corners)

    else:
        camera_poses = io_utils.collect_camera_poses_from_jsonfile(os.path.join(basepath, settings.CAMERA_POSE_FILE))
        
    if settings.NUM_POINTS is None:  
        settings.NUM_POINTS = settings.POINTS_PER_CAMERA * len(camera_poses.keys())

    logger.info("Num points: {0} | Min views: {1} | Max views: {2}".format(
        settings.NUM_POINTS, settings.MIN_VIEWS_PER_POINT, settings.MAX_VIEWS_PER_POINT))  
       
    # Generate the points
    if settings.POINT_TYPE == 'SWEEP':
        generate_points_per_camera(camera_poses, basepath)
    elif settings.POINT_TYPE == 'CORRESPONDENCES':
        generate_point_correspondences(model, camera_poses, basepath)
    else:
        raise NotImplementedError('Unknown settings.POINT_TYPE: ' + settings.POINT_TYPE)




def sample_camera_poses(model, bbox_corners):

    if settings.SCENE:
        camera_locations = sample_camera_locations_building(model, bbox_corners, n_samples=30, distance=settings.MIN_CAMERA_DISTANCE)
    else:
        camera_locations = sample_camera_locations_object(model, n_samples=settings.NUM_CAMERAS)

    num_samples = len(camera_locations)
    camera_quaternions = sample_camera_quaternion(num_samples)

    logger.info("Number of generated camera poses: {}".format(num_samples))

    camera_poses = {}
    camera_pose_list = []
    for camera_id in range(num_samples):
        quaternion_wxyz = camera_quaternions[camera_id]
        position = camera_locations[camera_id]
        rotation = io_utils.convert_quaternion_to_euler(quaternion_wxyz)
        camera_poses[str(camera_id).zfill(4)] = {'position': position, 'rotation': rotation, 'quaternion': quaternion_wxyz}
        camera_pose_list.append(
            {'camera_id': str(camera_id).zfill(4), 'location': position, 'rotation_euler': np.array(rotation).tolist(),
             'rotation_quaternion': quaternion_wxyz})

    save_generated_camera_poses(camera_pose_list)

    return camera_poses


def save_generated_camera_poses(cameras):
    with open(os.path.join(basepath, 'camera_poses.json'), 'w') as jsonfile:
        json.dump(cameras, jsonfile, indent=4)



def get_point_uuid(msg):
    """ Returns a point uuid that is either the passed message """
    point_uuid = str(msg)
    return point_uuid


def generate_point(passed_args):
    n_generated, camera_poses, basepath = passed_args
    model = utils.get_mesh()
    with Profiler("Point {}".format(n_generated), logger) as prf:
        next_point, camera_evaluations, acceptable_cameras = get_viable_point_and_corresponding_cameras(
            model, camera_poses, min_views=settings.MIN_VIEWS_PER_POINT, point_num=n_generated)
        assert len(acceptable_cameras) > 0
        point_data = create_point_info(acceptable_cameras=acceptable_cameras, next_point=next_point,
                                       point_uuid=get_point_uuid(n_generated), camera_extrinsics=camera_poses,
                                       camera_evaluations=camera_evaluations, basepath=basepath)

        duration = prf.step("Point {}/{}".format(n_generated, settings.NUM_POINTS))
    return point_data, tuple(next_point), set(acceptable_cameras), duration


def generate_point_correspondences(model, camera_poses, basepath):
    ''' Generates and saves points into basepath. These points are generated as correspondences
      where each point_uuid.json is an array of view_dicts, or information about a camera which
      has line-of-sight to the desired point. Each view_dict includes information about the
      target point, too.
    Args:
      model: A Blender mesh that will be used to propose points
      camera_poses: A Dict of camera_uuids -> camera extrinsics
      basepath: The directory in which to save points
    Returns:
      None (saves points)
    '''

    with Profiler("Generate fixated points", logger) as prf:
        p = Pool(settings.MAX_CONCURRENT_PROCESSES)
        for k, v in camera_poses.items():
            v['rotation'] = tuple(v['rotation'])
        args_to_pass = [(n_generated, camera_poses, basepath) for n_generated in range(settings.NUM_POINTS)]
        point_datas_and_next_points = p.map(generate_point, args_to_pass)
        point_datas, points_list, cameras_with_view_of_point, durations = zip(*point_datas_and_next_points)
        point_datas = list(point_datas)
        points_list = [Vector(point) for point in points_list]
        print([len(point_data) for point_data in point_datas])
        point_uuids = [point_data[0]['point_uuid'] for point_data in point_datas]


    with Profiler("Generate nonfixated points", logger) as prf:
        add_nonfixated_point_info(point_datas, points_list, cameras_with_view_of_point)
        old_views_count = n_views(point_datas, min_nonfixated=0)
        while True:
            nonfixated_pds = make_nonfixated_point_datas(point_datas, settings.MIN_VIEWS_PER_POINT)
            if settings.SCENE:
                new_pds = prune_fixated(point_datas, nonfixated_pds, points_list, settings.MIN_VIEWS_PER_POINT)
            else:
                new_pds = point_datas
            new_views_count = n_views(new_pds, min_nonfixated=1)
            logger.info("Filtering: {} -> {}".format(old_views_count, new_views_count))

            if new_views_count == old_views_count:
                break
            old_views_count = new_views_count
            point_datas = new_pds

    finalize_point_datas(point_datas, nonfixated_pds, points_list, point_uuids)
    with Profiler("Saving", logger) as prf:
        for point_num, point_data in enumerate(point_datas):
            if point_data:
                save_point_data(point_data, basepath, get_point_uuid(point_num), fixated=True)
        prf.step("fixated")
        for point_num, nfpd in enumerate(nonfixated_pds):
            if nfpd:
                save_point_data(nfpd, basepath, get_point_uuid(point_num), fixated=False)
        prf.step("nonfixated")


def make_nonfixated_point_datas(point_datas, min_views):
    nonfixated_pds = [[] for pd in point_datas]
    for point_num, point_data in enumerate(point_datas):
        for view_num, view_dict in enumerate(point_data):
            for nonfixated_neighbor in view_dict['nonfixated_points_in_view']:
                nonfixated_pds[nonfixated_neighbor].append(view_dict)
    return prune_nonfixated(nonfixated_pds, min_views)


def n_cameras_in_point_data(point_data):
    camera_uuids = set()
    for view_dict in point_data:
        camera_uuids.add(view_dict['camera_uuid'])
    return camera_uuids


def prune_nonfixated(nonfixated_pds, min_views):
    new_nonfixated_pds = [[] for pd in nonfixated_pds]
    culled = set()
    for point_num, point_data in enumerate(nonfixated_pds):
        if len(n_cameras_in_point_data(point_data)) >= min_views:
            new_nonfixated_pds[point_num] = nonfixated_pds[point_num]
        else:
            culled.add(point_num)
    if culled:
        logger.info("Nonfixated: culling points {}".format(culled))
    return new_nonfixated_pds


def prune_fixated(fixated_pds, nonfixated_pds, points_list, min_views):
    for point_num, nfpd in enumerate(nonfixated_pds):
        if len(nfpd) < min_views:
            points_list[point_num] = None

    new_fixated_pds = []
    for point_num, fpd in enumerate(fixated_pds):
        for view_num, view_dict in enumerate(fpd):
            for point_idx, point in enumerate(points_list):
                if point is None and point_idx in view_dict['nonfixated_points_in_view']:
                    view_dict['nonfixated_points_in_view'].remove(point_idx)

    culled = set()
    for point_num, fpd in enumerate(fixated_pds):
        do_keep = is_keep_point(fpd)
        if not do_keep:
            culled.add(point_num)
        for view_num, view_dict in enumerate(fpd):
            if not do_keep:
                view_dict['nonfixated_points_in_view'] = set()

    if culled:
        logger.info("Fixated: culling points {}".format(culled))
    return fixated_pds


def finalize_point_datas(fixated_pds, nonfixated_pds, points_list, point_uuids):
    # Make set -> list
    point_locs = []
    for point_num, fixated_pd in enumerate(fixated_pds):
        point_locs.append(fixated_pd[0]['point_location'])
        # Make set -> list for JSON save
        for view_dict in fixated_pd:
            view_dict['nonfixated_points_in_view'] = list(view_dict['nonfixated_points_in_view'])
        # Only keep the points that link to a nonfixated point
        fixated_pds[point_num] = [vd for vd in fixated_pd if len(vd['nonfixated_points_in_view']) > 0]
        # Keep a record of this transfer
        for new_view_num, view_dict in enumerate(fixated_pds[point_num]):
            view_dict['view_id'] = new_view_num

    # Make a deep copy of views
    for point_num, point_data in enumerate(nonfixated_pds):
        nonfixated_pds[point_num] = [copy.deepcopy(view_dict) for view_dict in point_data]

    # Adjust the model_x/y/z to be the nonfixated point
    for point_num, point_data in enumerate(nonfixated_pds):
        for view_num, view_dict in enumerate(point_data):
            view_dict['point_location'] = tuple(points_list[point_num])

    # Nonfixated pds should just contain references 
    for point_num, point_data in enumerate(nonfixated_pds):
        if not point_data:
            continue
        point_data_for_nonfixated = [make_view_dict_nonfixated(view_dict) for view_dict in point_data]
        pd = fixated_pds[point_num]
        new_point_data = {'point_uuid': point_uuids[point_num],
                          'point_location': point_locs[point_num],
                          'views': point_data_for_nonfixated}
        nonfixated_pds[point_num] = new_point_data


def make_view_dict_nonfixated(view_dict):
    fixated_camera, fixated_camera_data, scene = utils.get_or_create_camera(
        view_dict['camera_location'],
        view_dict['camera_rotation_final'],
        resolution_x=settings.RESOLUTION_X,
        resolution_y=settings.RESOLUTION_Y,
        field_of_view=view_dict['field_of_view_rads'],
        camera_name="viable_point_camera")
    nfvd = io_utils.get_nonfixated_point_data(view_dict['point_location'], fixated_camera)
    nfvd['point_uuid'] = view_dict['point_uuid']
    nfvd['view_id'] = view_dict['view_id']
    nfvd['camera_uuid'] = view_dict['camera_uuid']
    return nfvd


def n_views(point_datas, min_nonfixated=0):
    count = 0
    for point_data in point_datas:
        for view_dict in point_data:
            if len(view_dict['nonfixated_points_in_view']) >= min_nonfixated:
                count += 1
    return count


def prune_point_datas(point_datas, matching, min_views):
    new_point_datas = []
    for point_num, point_data in enumerate(point_datas):
        new_point_data = [view_dict for view_num, view_dict in enumerate(point_data)
                          if (point_num, view_num) in matching]
        new_point_datas.append(new_point_data)

    new_point_datas = [pd for pd in new_point_datas if len(pd) >= min_views]
    return new_point_datas



def add_nonfixated_point_info(point_datas, point_locs, cameras_with_view_of_point):
    empty = utils.create_empty("Nonfixated", (0, 0, 0))
    camera, camera_data, scene = utils.get_or_create_camera(
        location=(0, 0, 0), rotation=(0, 0, 0),
        field_of_view=settings.FIELD_OF_VIEW_MATTERPORT_RADS,
        camera_name='camera_nonfixated')

    for point_num, point_data in enumerate(point_datas):
        for view_num, view_dict in enumerate(point_data):
            view_dict['nonfixated_points_in_view'] = set()
            view_dict['view_id'] = view_num
            chosen = []
            for candidate_point_idx, candidate_point in enumerate(point_locs):
                if evaluate_nonfixated_camera_view_of_point(
                        view_dict, candidate_point,
                        acceptable_cameras=cameras_with_view_of_point[candidate_point_idx],
                        camera=camera, empty=empty):
                    view_dict['nonfixated_points_in_view'].add(candidate_point_idx)
                    chosen.append(candidate_point_idx)
            logger.debug("({}, {}) -> {}".format(point_num, view_num, chosen))
    utils.delete_cameras_and_empties()


def get_random_point_from_mesh(num_points, model):
    """
      Generates a given number of random points from the mesh
    """
    me = model.data
    me.calc_tessface()  # recalculate tessfaces
    tessfaces_select = [f for f in me.tessfaces if f.select]
    random.shuffle(tessfaces_select)
    multiplier = 1 if len(tessfaces_select) >= num_points else num_points // len(tessfaces_select)
    points = [model.matrix_world * p for p in
              bpy_extras.mesh_utils.face_random_points(multiplier, tessfaces_select[:num_points])]
    return points


def get_viable_point_and_corresponding_cameras(model, camera_locations, min_views=3, point_num=None, extra_info=None):
    """
      Keeps randomly sampling points from the mesh until it gets one that is viewable from at least
        'min_views' camera locations.

      Args:
        model: A Blender mesh object
        min_views: The minimum viable number of views
        camera_locations: A list of dicts which have information about the camera location
        point_num: The index of the point in test_assets/points_to_generate.json - needs to be
            specified iff settings.MODE == 'TEST'
        extra_info: Runtime information will be added to this dict

      Returns:
        point: A point that has at least 'min_views' cameras with line-of-sight on point
        visible: A Dict of visible cameras---camera_uuid -> extrinsics
        obliquness: A Dict of camera_uuid->( point_normal, obliqueness_angle )
        extra_info: Dict used for summarizing
    """
    while True:
        utils.delete_objects_starting_with("Candidate")
        # Generate point 
        candidate_point = get_random_point_from_mesh(1, model)[0]
        utils.create_empty("Candidate", candidate_point)
        camera_evaluations = evaluate_camera_views_of_point(camera_locations, candidate_point)

        acceptable_cameras = filter_acceptable_cameras(camera_evaluations)

        # Decide whether to continue looking for points
        if len(acceptable_cameras) >= min_views:
            break

    return candidate_point, camera_evaluations, acceptable_cameras


def evaluate_camera_views_of_point(camera_locations, candidate_point):
    camera_evaluations = {}
    camera, _, scene = utils.get_or_create_camera(location=(0, 0, 0), rotation=(0, 0, 0),
                                                  field_of_view=settings.FIELD_OF_VIEW_MATTERPORT_RADS,
                                                  camera_name="viable_point_camera")
    for camera_uuid, camera_extrinsics in camera_locations.items():
        camera_rotation_euler = Euler(camera_extrinsics['rotation'], settings.EULER_ROTATION_ORDER)
        camera_location = camera_extrinsics['position']
        camera.location = camera_location
        camera.rotation_euler = camera_rotation_euler
        bpy.context.scene.update()

        # Compute whether to use this view
        los_normals_and_obliqueness = try_get_line_of_sight_obliqueness(camera.location, candidate_point)

        if los_normals_and_obliqueness:
            point_normal, camera_obliqueness = los_normals_and_obliqueness
        else:
            point_normal, camera_obliqueness = (None, None)
        camera_evaluations[camera_uuid] = {
            'camera_location': tuple(camera.location),
            'camera_rotation': tuple(camera.rotation_euler),
            'pitch': math.degrees(io_utils.get_pitch_of_point(camera, candidate_point)),
            'point': tuple(candidate_point),
            'los': (los_normals_and_obliqueness is not None),
            'point_normal': point_normal,
            'obliqueness': camera_obliqueness}
           
        logger.debug(camera_uuid + ": \n" + pp.pformat(camera_evaluations[camera_uuid]))
    return camera_evaluations


def evaluate_nonfixated_camera_view_of_point(view_dict, point, acceptable_cameras, camera, empty):
    ''' Returns whether the given camera/pose has LOS to the given point 
        Args:
            view_dict: A view dict for the camera/pose
            point: The target point
            acceptable_cameras: A Set (or something with 'in') that is a list of all the cameras 
                with an acceptable view of the point. This spares us the time requried to raycast.
        Returns:
            bool
    '''
    fixated_point = Vector(view_dict['point_location'])
    if (fixated_point - point).magnitude < settings.LINE_OF_SITE_HIT_TOLERANCE:
        return False  # Don't consider a camera seeing its point as a valid edge
    empty.location = point
    camera.location = view_dict["camera_location"]
    camera.rotation_euler = view_dict["camera_rotation_final"]
    utils.set_camera_fov(camera.data, view_dict["field_of_view_rads"])
    bpy.context.scene.update()

    valid = False
    if (is_point_inside_frustum(camera, point) and
            view_dict['camera_uuid'] in acceptable_cameras):

        valid = True
    return valid


def filter_acceptable_cameras(camera_evaluations):
    acceptable_cameras = []
    for camera_uuid, camera_evaluation in camera_evaluations.items():
        if camera_evaluation['los']: 
            acceptable_cameras.append(camera_uuid)
    return acceptable_cameras


def is_point_inside_frustum(camera, point):
    """ Evaluates whether a given point is inside the frustrum of the given camera """
    cs, ce = camera.data.clip_start, camera.data.clip_end
    co_ndc = world_to_camera_view(bpy.context.scene, camera, point)
    # check wether point is inside frustum
    if (0.0 < co_ndc.x < 1.0 and
            0.0 < co_ndc.y < 1.0 and
            cs < co_ndc.z < ce):
        return True
    else:
        return False



def try_get_line_of_sight_obliqueness(start, end, scene=bpy.context.scene):
    """
      Casts a ray in the direction of start to end and returns the surface
      normal of the face containing 'end', and also the angle between the
      normal and the cast ray. If the cast ray does not hit 'end' before
      hitting anything else, it returns None.

      Args:
        start: A Vector
        end: A Vector
        scene: A Blender scene

      Returns:
        ( normal_of_end, obliqueness_angle )
        normal_of_end: A Vector normal to the face containing end
        obliqueness_angle: A scalar in rads
    """
    scene = bpy.context.scene
    if (bpy.app.version[1] >= 75):
        direction = end - Vector(start)
        (ray_hit, location, normal, index, obj, matrix) = scene.ray_cast(start, direction)
    else:
        direction = end - Vector(start)  # We need to double the distance since otherwise
        farther_end = end + direction  # The ray might stop short of the target
        (ray_hit, obj, matrix, location, normal) = scene.ray_cast(start, farther_end)

    if not ray_hit or (location - end).length > settings.LINE_OF_SITE_HIT_TOLERANCE:
        return None
    obliqueness_angle = min(direction.angle(normal), direction.angle(-normal))
    return normal, obliqueness_angle


def create_point_info(acceptable_cameras, next_point, point_uuid,
                      camera_extrinsics, camera_evaluations, basepath):
    ''' Saves out a CORRESPONDENCE-type point to a file in basepath. 
    Each point_uuid.json is an array of view_dicts, or information about a camera which
    has line-of-sight to the desired point. Each view_dict includes information about the
    target point, too. TODO(sasha): Update comment

    Args: 
        visible_cameras: A list of all camera_poses which have line-of-sight to next_point
        next_point: A 3-tuple of the XYZ coordinates of the target_point
        point_uuid: A uuid to call this point. Defines the filename. 
        camera_evaluations: A dict of camera_uuid -> evaluation info 
        basepath: Directory under which to save point information
    Returns:
        None (Save a point file under basepath)
    '''
    with Profiler("Save point"):
        empty = utils.create_empty("Empty", next_point)
        point_data = []

        # So that we're not just using the same camera for each point
        shuffled_views = list(acceptable_cameras)
        random.shuffle(shuffled_views)
        for view_number, camera_uuid in enumerate(shuffled_views):
            next_point_data = io_utils.get_save_info_for_correspondence(empty,
                                                                        point=next_point, point_uuid=point_uuid,
                                                                        camera_uuid=camera_uuid,
                                                                        resolution_x=settings.RESOLUTION_X,
                                                                        resolution_y=settings.RESOLUTION_Y,
                                                                        camera_fov=sample_fov(),
                                                                        acceptable_cameras=acceptable_cameras,
                                                                        camera_extrinsics=camera_extrinsics,
                                                                        camera_evaluations=camera_evaluations)
            point_data.append(next_point_data)
            if view_number == int(settings.MAX_VIEWS_PER_POINT):
                break
            if view_number == settings.STOP_VIEW_NUMBER:
                break
        utils.delete_objects_starting_with("Empty")  # Clean up
        return point_data


def save_point_data(point_data, basepath, point_uuid, fixated=True):
    if settings.POINT_TYPE == 'SWEEP':
        subdir = os.path.join('pano', TASK_NAME)
    else:
        subdir = TASK_NAME if fixated else 'nonfixated'

    try:
        os.mkdir(os.path.join(basepath, subdir))
    except:
        pass

    if fixated and settings.POINT_TYPE != 'SWEEP' \
               and n_views([point_data], min_nonfixated=1) < settings.MIN_VIEWS_PER_POINT:
        print("Skipping {}".format(point_uuid))
        print("Nonfixated: {}".format([vd['nonfixated_points_in_view'] for vd in point_data]))
        return
        
    point_dir = os.path.join(basepath, subdir)
    task = 'fixated' if fixated else 'nonfixated'
    if fixated:
        for view_num, view_dict in enumerate(point_data):
            if (fixated and 'nonfixated_points_in_view' in view_dict and
                    len(view_dict['nonfixated_points_in_view']) == 0):
                continue
            file_name = io_utils.get_file_name_for(point_dir, point_uuid, view_num,
                                                   view_dict['camera_uuid'], task + 'pose', 'json')
            outfile_path = os.path.join(basepath, subdir, file_name)
            with open(outfile_path, 'w') as outfile:
                json.dump(view_dict, outfile)
    else:
        file_name = io_utils.get_file_name_for(point_dir, point_uuid, 'all',
                                               'allcameras', task + 'pose', 'json')
        outfile_path = os.path.join(basepath, subdir, file_name)
        with open(outfile_path, 'w') as outfile:
            json.dump(point_data, outfile)


def sample_fov():
    # FOV
    z_val = 2
    while abs(z_val) > 1:
        z_val = np.random.normal(loc=0.0, scale=1.)

        # z_val = np.abs(z_val)
        # fov = settings.FIELD_OF_VIEW_MAX_RADS - z_val * (
        #         settings.FIELD_OF_VIEW_MAX_RADS - settings.FIELD_OF_VIEW_MIN_RADS)

        half_range = (settings.FIELD_OF_VIEW_MAX_RADS - settings.FIELD_OF_VIEW_MIN_RADS) / 2
        fov = settings.FIELD_OF_VIEW_MIN_RADS + half_range + z_val * half_range

    return fov


# SWEEPS
def sample_yaw_pitch_roll(sample_i):
    if settings.CREATE_PANOS:
        if sample_i == 0:  # Top
            return math.pi, math.pi / 2, settings.FIELD_OF_VIEW_MATTERPORT_RADS
        elif sample_i == 1:  # Front
            return 0.0, 0.0, settings.FIELD_OF_VIEW_MATTERPORT_RADS
        elif sample_i == 2:  # Right
            return math.pi / 2, 0.0, settings.FIELD_OF_VIEW_MATTERPORT_RADS
        elif sample_i == 3:  # Back
            return math.pi, 0.0, settings.FIELD_OF_VIEW_MATTERPORT_RADS
        elif sample_i == 4:  # Left
            return -math.pi / 2., 0.0, settings.FIELD_OF_VIEW_MATTERPORT_RADS
        elif sample_i == 5:  # Bottom
            return math.pi, -math.pi / 2, settings.FIELD_OF_VIEW_MATTERPORT_RADS
        else:
            raise ValueError('Too many samples for a panorama! (Max 6)')
    else:
        # How to generate samples from a camera sweep
        yaw = np.random.uniform(low=-math.pi, high=math.pi)

        pitch = np.random.normal(loc=0.0, scale=math.radians(15.))

        # FOV
        z_val = 2
        while abs(z_val) > 1:
            z_val = np.random.normal(loc=0.0, scale=1.)

            # z_val = np.abs(z_val)
            # fov = settings.FIELD_OF_VIEW_MAX_RADS - z_val * (
            #         settings.FIELD_OF_VIEW_MAX_RADS - settings.FIELD_OF_VIEW_MIN_RADS)

            half_range = (settings.FIELD_OF_VIEW_MAX_RADS - settings.FIELD_OF_VIEW_MIN_RADS) / 2
            fov = settings.FIELD_OF_VIEW_MIN_RADS + half_range + z_val * half_range
             
        return yaw, pitch, fov


def generate_points_per_camera(camera_poses, basepath):
    ''' Generates and saves points into basepath. Each point file corresponds to one cameara and
      contains an array of different view_dicts for that camera. These view_dicts are distinct from
      the ones created by generate_point_correspondences since these views to not share a target point.
    Args:
      camera_poses: A Dict of camera_uuids -> camera extrinsics
      basepath: The directory in which to save points
    Returns:
      None (saves points)
    '''
    # Generate random points for each camera:
    for camera_uuid in camera_poses.keys():
        with Profiler("Save point", logger):
            point_data = []
            point_uuid = get_point_uuid(camera_uuid)

            # Save each sampled camera position into point_data
            for sample_i in range(settings.NUM_POINTS):
                yaw, pitch, fov = sample_yaw_pitch_roll(sample_i)
                view_dict = io_utils.get_save_info_for_sweep(
                    fov, pitch, yaw, point_uuid, camera_uuid, camera_poses)
                point_data.append(view_dict)

            # Save result out
            save_point_data(point_data, basepath, point_uuid, fixated=True)


# SUMMARIES

def is_keep_point(point_data):
    return n_views([point_data], min_nonfixated=1) >= settings.MIN_VIEWS_PER_POINT



if __name__ == '__main__':
    global logger
    logger = io_utils.create_logger(__name__)
    with Profiler("generate_points.py", logger):
        main()
