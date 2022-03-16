"""
  Name: io_utils.py
  Author: Sasha Sax, CVGL
  Desc: Contains utilities for saving and loading information

  Usage: for import only
"""

from load_settings import settings

try:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix, Quaternion, Euler
    import utils
    from utils import create_camera, axis_and_positive_to_cube_face, cube_face_idx_to_skybox_img_idx
except:
    print("Can't import Blender-dependent libraries in io_utils.py...")


import ast
import csv
import glob
from itertools import groupby
import json
import logging
import math
from natsort import natsorted, ns
import numpy as np
import os
import sys

axis_and_positive_to_skybox_idx = {
    ("X", True): 1,
    ("X", False): 3,
    ("Y", True): 0,
    ("Y", False): 5,
    ("Z", True): 2,
    ("Z", False): 4
}

skybox_number_to_axis_and_rotation = {5: ('X', -math.pi / 2),
                                      0: ('X', math.pi / 2),
                                      4: ('Y', 0.0),
                                      3: ('Y', math.pi / 2),
                                      2: ('Y', math.pi),
                                      1: ('Y', -math.pi / 2)}

img_format_to_ext = {"png": 'png', "jpeg": "jpg", "jpg": "jpg"}
logger = settings.LOGGER


def collect_camera_poses_from_csvfile(infile):
    """
      Reads the camera uuids and locations from the given file

      Returns:
        points: A Dict of the camera locations from uuid -> position, rotation, and quaterion.
                Quaterions are wxyz ordered
    """
    points = {}

    with open(infile) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            uuid = row[0]
            position = (float(row[1]), float(row[2]), float(row[3]))
            quaternion_wxyz = (float(row[7]), float(row[4]), float(row[5]), float(row[6]))
            logger.debug("Camera: {0}, rotation: {1}".format(uuid, quaternion_wxyz))
            # quaternion_xyzw = (float(row[4]), float(row[5]), float(row[6]), float(row[7]))
            rotation = convert_quaternion_to_euler(quaternion_wxyz)
            points[uuid] = (position, rotation, quaternion_wxyz)
            points[uuid] = {'position': position, 'rotation': rotation, 'quaternion': quaternion_wxyz}
    csvfile.close()
    return points

def collect_camera_poses_from_jsonfile(infile):
    """
      Reads the camera uuids and locations from the given file

      Returns:
        points: A Dict of the camera locations from uuid -> position, rotation, and quaterion.
                Quaterions are wxyz ordered
    """
    points = {}

    with open(infile) as jsonfile:
        cameras = json.load(jsonfile)
        for camera in cameras:
            uuid = camera['camera_id']
            position = camera['location']
            quaternion_wxyz = camera['rotation_quaternion']
            logger.debug("Camera: {0}, rotation: {1}".format(uuid, quaternion_wxyz))
            rotation = convert_quaternion_to_euler(quaternion_wxyz)
            points[uuid] = {'position': position, 'rotation': rotation, 'quaternion': quaternion_wxyz}

    return points


def convert_quaternion_to_euler(quaternion):
    blender_quat = Quaternion(quaternion)
    result = blender_quat.to_euler(settings.EULER_ROTATION_ORDER)

    # levels the quaternion onto the plane images were taken at
    result.rotate_axis('X', math.pi / 2)

    return result


def create_logger(logger_name, filename=None):
    logging.basicConfig(filename=filename)
    logger = logging.getLogger(logger_name)
    logger.setLevel(settings.LOGGING_LEVEL)
    return logger


def delete_materials():
    ''' Deletes all materials in the scene. This can be useful for stanardizing meshes. '''
    # https://blender.stackexchange.com/questions/27190/quick-way-to-remove-thousands-of-materials-from-an-object
    C = bpy.context
    obj = C.object
    obj.data.materials.clear()


def get_2d_point_from_3d_point(three_d_point, K, RT):
    P = K * RT
    product = P * Vector(three_d_point)
    two_d_point = (product[0] / product[2], product[1] / product[2])
    return two_d_point


def get_2d_point_and_decision_vector_from_3d_point(camera_data, location, rotation, target):
    K = get_calibration_matrix_K_from_blender(camera_data)
    RT = get_3x4_RT_matrix_from_blender(Vector(location), rotation)
    P = K * RT
    decision_vector = P * Vector(target)
    x, y = get_2d_point_from_3d_point(target, K, RT)
    return (x, y, decision_vector)


def get_3x4_RT_matrix_from_blender(location, rotation):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation

    # Use matrix_world instead to account for all constraints
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes

    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1 * R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv * R_world2bcam
    T_world2cv = R_bcam2cv * T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
    ))

    return RT


def get_calibration_matrix_K_from_blender(camd):
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    # scale = scene.render.resolution_percentage / 100
    scale = 1
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    f_in_px = (f_in_mm * resolution_x_in_px) / sensor_width_in_mm

    if (camd.sensor_fit == 'VERTICAL'):
        # the sensor height is fixed (sensor fit is horizontal), 
        # the sensor width is effectively changed with the pixel aspect ratio
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else:  # 'HORIZONTAL' and 'AUTO'
        # the sensor width is fixed (sensor fit is horizontal), 
        # the sensor height is effectively changed with the pixel aspect ratio
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0  # only use rectangular pixels

    K = Matrix(
        ((f_in_px, skew, u_0),
         (0, f_in_px, v_0),
         (0, 0, 1)))

    return K



def get_floorplan_file(directory):
    f_path = os.path.join(directory, "plans", "semantic%2Ffloorplan_000.png")
    return f_path
    if os.path.isfile(f_path):
        return f_path
    else:
        return os.path.join(directory, "plans", "semantic%2Ffloorplan_000.png")



def get_file_name_for(dir, point_uuid, view_number, camera_uuid, task, ext):
    """
      Returns the filename for the given point, view, and task

      Args:
        dir: The parent directory for the model
        task: A string definint the task name
        point_uuid: The point identifier
        view_number: This is the nth view of the point
        camera_uuid: An identifier for the camera
        ext: The file extension to use
    """
    if settings.CREATE_TRAJECTORY:
        view_specifier = str(view_number).zfill(4)
    else:
        view_specifier = view_number
    filename = "point_{0}_view_{1}_domain_{2}.{3}".format(point_uuid, view_specifier, task, ext)
    return os.path.join(dir, filename)


def get_model_file(dir, typ='RAW'):
    if typ == 'RAW':
        model_file = settings.MODEL_FILE
    elif typ == 'SEMANTIC':
        model_file = settings.SEMANTIC_MODEL_FILE
    elif typ == 'RGB':
        model_file = settings.RGB_MODEL_FILE
    else:
        raise ValueError('Unknown type of model file: {0}'.format(typ))
        
    return os.path.join(dir, model_file)


def get_number_imgs(point_infos):
    if settings.CREATE_PANOS:
        return len(point_infos)
    else:
        n_imgs = 0
        if settings.CREATE_FIXATED:
            n_imgs += sum([len(pi) for pi in point_infos])
       
        return n_imgs



def get_pitch_of_point(camera, point):
    """
      Args:
        camera: A Blender camera
        point: A 3-tuple of coordinates of the target point

      Returns:
        pitch: A float
    """

    # Just check whether the direction of the target point is within pi / 12 of the plane of rotation
    point_in_local_coords = camera.matrix_world.inverted() * Vector(point)
    angle_to_normal = Vector((0, 1, 0)).angle(point_in_local_coords)
    angle_to_plane = math.pi / 2. - angle_to_normal
    return angle_to_plane



def get_save_info_for_correspondence(empty, point, point_uuid,
                                     camera_uuid, resolution_x, resolution_y, camera_fov,
                                     acceptable_cameras, camera_extrinsics, camera_evaluations):
    """
      Creates info for a point and camera that allows easy loading of a camera in Blender

      Args:
        empty: An Empty located at the point
        point: The xyz coordinates of the point to create the save info for
        point_uuid: The uuid pertaining to this point
        point_normal: The normal of the face the point lies on
        camera_uuid: The uuid of the camera for which we will be creating info for
        cameras: This a dict of many cameras for which camera_uuid is a key
        obliqueness_angle: Angle formed between the point_normal and camera->point_location, in rads
        resolution: Skybox camera resolution

      Returns:
        save_dict: A Dict of useful information. Currently it's keys are
          camera_distance: The distance from the camera to the point in meters
          camera_location: The location of the camera in the 3d model
          camera_original_rotation: The rotation_euler of the camera in the 3d model
          img_path: The path to the unique image for this uuid that has line-of-sight on the point
          model_x: The x coordinate of the point in the model
          model_y: The y coordinate of the point in the model
          model_z: The z coordinate of the point in the model
          nonfixated_pixel_x:
          nonfixated_pixel_y:
          obliqueness_angle: Angle formed between the point_normal and camera->point_location, in rads
          point_normal: The normal of the face the point lies on
          rotation_of_skybox: The Euler rotation that, when the camera is set to inside the cube, will provide the skybox image
          rotation_from_original_to_point: Apply to camera_original_rotation to aim camera at target
          skybox_img: The unique skybox image number that has line-of-sight on the point
          skybox_pixel_x: The exact x pixel in the skybox image where the point will be
          skybox_pixel_y: The exact y pixel in the skybox image where the point will be
          uuid: The uuid of this camera
    """
    point_data = {}

    # Unpack the camera extrinsics
    location = camera_extrinsics[camera_uuid]['position']
    rotation_euler = camera_extrinsics[camera_uuid]['rotation']

    # Save basic info
    point_data['camera_distance'] = (Vector(location) - Vector(point)).magnitude
    point_data['camera_uuid'] = camera_uuid
    point_data['camera_location'] = tuple(location)
    point_data['camera_rotation_original'] = tuple(rotation_euler)
    point_data['field_of_view_rads'] = camera_fov
    point_data['obliqueness_angle'] = camera_evaluations[camera_uuid]['obliqueness']
    point_data['point_location'] = tuple(point)
    point_data['point_normal'] = tuple(camera_evaluations[camera_uuid]['point_normal'])
    point_data['point_uuid'] = point_uuid
    point_data['resolution'] = resolution_x # ????????

    ## SKYBOX
    # Find and save skybox number
    camera, camera_data, scene = utils.get_or_create_camera(location, rotation_euler,
                                                            field_of_view=settings.FIELD_OF_VIEW_MATTERPORT_RADS,
                                                            camera_name="Camera_for_pitch")
    point_data.update(get_skybox_path_info(point, camera_uuid, camera, empty))
    point_data['point_pitch'] = get_pitch_of_point(camera, point)

    ## FIXATED
    # Now save the rotation needed to point at the target
    fixated_camera, fixated_camera_data, scene = utils.get_or_create_camera(location, rotation_euler,
                                                                            resolution_x=resolution_x,
                                                                            resolution_y=resolution_y,
                                                                            field_of_view=camera_fov,
                                                                            camera_name="Camera_for_rotation")
    utils.point_camera_at_target(fixated_camera, empty)
    point_data['camera_rotation_from_original_to_final'] = tuple(
        utils.get_euler_rotation_between(
            camera.rotation_euler,
            fixated_camera.rotation_euler))
    point_data['camera_rotation_final'] = tuple(fixated_camera.rotation_euler)
    point_data['camera_rotation_final_quaternion'] = tuple(fixated_camera.rotation_euler.to_quaternion())

    # NONFIXATED
    #   point_data.update( get_nonfixated_point_data( point, fixated_camera ) )
    return point_data


def get_nonfixated_point_data(point, camera):
    x, y, _ = get_2d_point_and_decision_vector_from_3d_point(
        camera.data, camera.location, camera.rotation_euler, point)

    return {'nonfixated_pixel_x': int(round(x)),
            'nonfixated_pixel_y': int(round(y))}


def get_skybox_path_info(point, camera_uuid, camera, empty):
    point_data = {}
    skybox_number = get_skybox_img_number_containing_point(camera.location, camera.rotation_euler, empty)
    point_data['skybox_img'] = skybox_number
    point_data['skybox_img_path'] = os.path.join("./img/high", "{0}_skybox{1}.jpg".format(camera_uuid, skybox_number))
    point_data['point_pitch'] = get_pitch_of_point(camera, point)
    return point_data


def get_save_info_for_sweep(fov, pitch, yaw, point_uuid, camera_uuid, cameras):
    """
      Creates info for a point and camera that allows easy loading of a camera in Blender

      Args:
        fov: The field of view of the camera
        pitch: The pitch of the camera relative to its plane of rotation
        yaw: The yaw of the camera compared to its initial Euler coords
        point_uuid: The uuid pertaining to this point
        camera_uuid: The uuid of the camera for which we will be creating info for
        cameras: This a dict of many cameras for which camera_uuid is a key

      Returns:
        save_dict: A Dict of useful information. Currently it's keys are
          {
          "camera_k_matrix":  # The 3x3 camera K matrix. Stored as a list-of-lists,
          "field_of_view_rads": #  The Camera's field of view, in radians,
          "camera_original_rotation": #  The camera's initial XYZ-Euler rotation in the .obj,
          "rotation_from_original_to_point":
          #  Apply this to the original rotation in order to orient the camera for the corresponding picture,
          "point_uuid": #  alias for camera_uuid,
          "camera_location": #  XYZ location of the camera,
          "frame_num": #  The frame_num in the filename,
          "camera_rt_matrix": #  The 4x3 camera RT matrix, stored as a list-of-lists,
          "final_camera_rotation": #  The camera Euler in the corresponding picture,
          "camera_uuid": #  The globally unique identifier for the camera location,
          "room": #  The room that this camera is in. Stored as roomType_roomNum_areaNum
          }
    """
    point_data = {}

    # Save basic info
    point_data['camera_uuid'] = camera_uuid
    point_data['point_uuid'] = point_uuid

    # Unpack the camera extrinsics
    camera_extrinsics = cameras[camera_uuid]
    location = camera_extrinsics['position']
    rotation_euler = camera_extrinsics['rotation']
    point_data['camera_original_rotation'] = tuple(rotation_euler)
    point_data['camera_location'] = location

    # Save initial camera locatoin
    camera, camera_data, scene = create_camera(location, rotation_euler,
                                               field_of_view=settings.FIELD_OF_VIEW_MATTERPORT_RADS,
                                               camera_name="Camera_save_point_1")

    # Save the rotation_euler for the camera to point at the skybox image in the model
    new_camera, new_camera_data, scene = create_camera(location, rotation_euler,
                                                       resolution_x=settings.RESOLUTION_X,
                                                       resolution_y=settings.RESOLUTION_Y,
                                                       field_of_view=fov,
                                                       camera_name="Camera_save_point_2")
    new_camera.rotation_euler.rotate_axis('Y', yaw)
    new_camera.rotation_euler.rotate_axis('X', pitch)
    point_data['rotation_from_original_to_point'] = tuple(
        utils.get_euler_rotation_between(
            camera.rotation_euler,
            new_camera.rotation_euler))
    point_data['final_camera_rotation'] = tuple(new_camera.rotation_euler)
    point_data['field_of_view_rads'] = fov

    def matrix_to_list_of_lists(mat):
        lst_of_lists = list(mat)
        lst_of_lists = [list(vec) for vec in lst_of_lists]
        return lst_of_lists

    point_data['camera_rt_matrix'] = matrix_to_list_of_lists(
        get_3x4_RT_matrix_from_blender(Vector(location), new_camera.rotation_euler))
    point_data['camera_k_matrix'] = matrix_to_list_of_lists(
        get_calibration_matrix_K_from_blender(new_camera_data))

    utils.delete_objects_starting_with("Camera_save_point_1")  # Clean up
    utils.delete_objects_starting_with("Camera_save_point_2")  # Clean up
    # utils.delete_objects_starting_with( "Camera" ) # Clean up
    return point_data


def get_skybox_img_number_containing_point(camera_location, camera_rotation_euler, empty_at_target):
    """
      This gets the image index of the skybox image.

      It works by finding the direction of the empty from the camera and then by rotating that vector into a
      canonical orientation. Then we can use the dimension with the greatest magnitude, and the sign of that
      coordinate in order to determine the face of the cube that the empty projects onto.
    """
    empty_direction = (empty_at_target.location - Vector(camera_location)).normalized()
    empty_direction.normalize()
    empty_direction.rotate(camera_rotation_euler.to_matrix().inverted())

    # The trick to finding the cube face here is that we can convert the direction
    max_axis, coord_val = max(enumerate(empty_direction), key=lambda x: abs(x[1]))  # Find the dim with
    sign = (coord_val >= 0.0)
    max_axis = ["X", "Y", "Z"][max_axis]  # Just make it more readable

    return axis_and_positive_to_skybox_idx[(max_axis, sign)]



def import_mesh(dir, typ='RAW'):
    ''' Imports a mesh with the appropriate processing beforehand.
    Args:
      dir: The dir from which to import the mesh. The actual filename is given from settings.
      typ: The type of mesh to import. Must be one of ['RAW', 'SEMANTIC', 'SEMANTIC_PRETTY', 'LEGO']
        Importing a raw model will remove all materials and textures.
    Returns:
      mesh: The imported mesh.
    '''
    model_fpath = get_model_file(dir, typ=typ)

    if '.obj' in model_fpath:
        bpy.ops.import_scene.obj(
            filepath=model_fpath,
            axis_forward=settings.OBJ_AXIS_FORWARD,
            axis_up=settings.OBJ_AXIS_UP)
        model = join_meshes()  # OBJs often come in many many pieces
        bpy.context.scene.objects.active = model
        
        if typ in ['RGB', 'SEMANTIC']:
            return

        for img in bpy.data.images:  # remove all images
            bpy.data.images.remove(img, do_unlink=True)

        delete_materials()

    elif '.ply' in model_fpath:
        bpy.ops.import_mesh.ply(filepath=model_fpath)

    model = bpy.context.object

    # Should we add this?? (Needed for clevr and google scanned objects)
    # selection = bpy.context.selected_objects
    # for o in selection:
    #     bpy.context.scene.objects.active = o
    #     bpy.ops.mesh.customdata_custom_splitnormals_clear()
    #################

    return model


def join_meshes():
    ''' Takes all meshes in the scene and joins them into a single mesh.
    Args:
        None
    Returns:
        mesh: The single, combined, mesh 
    '''
    # https://blender.stackexchange.com/questions/13986/how-to-join-objects-with-python
    scene = bpy.context.scene
    obs = []
    for ob in scene.objects:
        # whatever objects you want to join...
        if ob.type == 'MESH':
            obs.append(ob)

    ctx = bpy.context.copy()
    # one of the objects to join
    ctx["object"] = obs[0]
    ctx['active_object'] = obs[0]
    ctx['selected_objects'] = obs
    ctx["selected_editable_objects"] = obs
    # we need the scene bases as well for joining
    ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]
    bpy.ops.object.join(ctx)

    for ob in bpy.context.scene.objects:
        # whatever objects you want to join...
        if ob.type == 'MESH':
            return ob




def remesh_and_join_meshes():
    ''' Takes all meshes in the scene and joins them into a single mesh.
    Args:
        None
    Returns:
        mesh: The single, combined, mesh 
    '''
    # https://blender.stackexchange.com/questions/13986/how-to-join-objects-with-python
    scene = bpy.context.scene

    obs = []
    for ob in scene.objects:

        ###### Remesh modifier
        modifier = ob.modifiers.new(name="Remesh", type='REMESH')
        modifier.octree_depth = 7
        modifier.mode = 'SHARP'
        modifier.scale = 0.9
        # modifier.sharpness = 2
        modifier.use_remove_disconnected = False
        modifier.use_smooth_shade = True
        ##########

        ##### clean 
        # bm = bmesh.new()
        # bm.from_mesh(ob.data)
        # bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        # bm.to_mesh(ob.data)
        # bm.clear()
        # ob.data.update()
        # bm.free()

        # bpy.context.scene.objects.active = ob
        # bpy.ops.object.mode_set(mode='EDIT')
        # bpy.ops.mesh.select_all(action='SELECT')
        # # bpy.ops.mesh.fill_holes()
        # bpy.ops.mesh.normals_make_consistent(inside=False)
        # bpy.ops.object.editmode_toggle()
        ######

        # whatever objects you want to join...
        if ob.type == 'MESH':
            obs.append(ob)

    ctx = bpy.context.copy()
    # one of the objects to join
    ctx['active_object'] = obs[0]
    ctx['selected_objects'] = obs
    # we need the scene bases as well for joining
    ctx['selected_editable_bases'] = [scene.object_bases[ob.name] for ob in obs]
    bpy.ops.object.join(ctx)
    for ob in scene.objects:
        # whatever objects you want to join...
        if ob.type == 'MESH' and not ob.name.startswith('Ground_Mesh'):
            return ob



def load_saved_points_of_interest(model_dir):
    """
      Loads all the generated points that have multiple views.

      Args:
        dir: Parent directory of the model. E.g. '/path/to/model/u8isYTAK3yP'

      Returns:
        point_infos: A list where each element is the parsed json file for a point
    """

    if settings.CREATE_PANOS:
        point_files = natsorted(glob.glob(os.path.join(model_dir, "pano", "point_info", "point_*.json")))
    else:
        point_files = natsorted(glob.glob(os.path.join(model_dir, "point_info", "point_*.json")))

    point_infos = []
    for point_num, files_for_point in groupby(point_files, key=lambda x: parse_filename(x)['point_uuid']):
        pi = []
        # print("point num : ", point_num)
        for view_file in files_for_point:
            with open(view_file) as f:
                pi.append(json.load(f))
        point_infos.append(pi)
    logger.info("Loaded {0} points of interest.".format(len(point_infos)))
    return point_infos


def load_nonfixated_points_of_interest(model_dir):
    point_files = natsorted(glob.glob(os.path.join(model_dir, "nonfixated", "point_*.json")))
    point_infos = []
    for point_file in point_files:
        with open(point_file) as f:
            point_infos.append(json.load(f))
    return point_infos


def load_model_and_points(basepath, typ='RAW'):
    ''' Loads the model and points
    Args:
        basepath: The model path
    Returns:
        A Dict:
            'camera_poses': 
            'point_infos':
            'model: The blender mesh
    '''
    utils.delete_all_objects_in_context()
    point_infos = load_saved_points_of_interest(basepath)
    model = import_mesh(basepath, typ=typ)
    return {'point_infos': point_infos, 'model': model}



def parse_filename(filename):
    fname = os.path.basename(filename).split(".")[0]
    toks = fname.split('_')
    if toks[0] == "camera":
        point_uuid = toks[1]
        domain_name = toks[-1]
        view_num = toks[5]
    elif len(toks) == 6:
        point, point_uuid, view, view_num, domain, domain_name = toks
    elif len(toks) == 7:
        point, point_uuid, view, view_num, domain, domain_name, _ = toks

    return {'point_uuid': point_uuid, 'view_number': view_num, 'domain': domain_name}


def resave_point(basepath, view_num, view_dict):
    file_name = get_file_name_for(os.path.join(basepath, 'point_info'),
                                  view_dict['point_uuid'], view_num,
                                  view_dict['camera_uuid'], 'fixatedpose', 'json')
    outfile_path = os.path.join(basepath, 'point_info', file_name)
    print(outfile_path)
    with open(outfile_path, 'w') as outfile:
        json.dump(view_dict, outfile)



def safe_make_output_folder(basepath, task_name):
    if not os.path.exists(os.path.join(basepath, task_name)):
        os.mkdir(os.path.join(basepath, task_name))

if __name__ == '__main__':
    import argparse

    args = argparse.Namespace()
    load_settings(args)



  
