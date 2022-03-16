"""
  Name: create_images_utils.py
  Desc: Contains utilities which can be used to run 
  
"""

import logging
import os
import sys
from load_settings import settings


try:
    import bpy
    import numpy as np
    from mathutils import Vector, Matrix, Quaternion, Euler
    import io_utils
    from io_utils import get_number_imgs
    import utils
    from utils import Profiler

except:
    print("Can't import Blender-dependent libraries in io_utils.py. Proceeding, and assuming this is kosher...")

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

trans = [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0,0,0,1]],

    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0,0,0,1]],

    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0,0,0,1]],

    [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0,0,0,1]],
    [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0,0,0,1]],
    [[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0,0,0,1]],

    [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[0, 0, -1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0,0,0,1]],
    [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0,0,0,1]],
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0,0,0,1]],

    [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, -1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0,0,0,1]],    
    [[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0,0,0,1]],    

    
]


def start_logging():
    ''' '''
    #   global logger
    logger = io_utils.create_logger(__name__)
    utils.set_random_seed()
    basepath = os.getcwd()
    return logger, basepath


def setup_rendering(setup_scene_fn, setup_nodetree_fn, logger, save_dir, apply_texture=None):
    ''' Sets up everything required to render a scene 
    Args:
    Returns:
        render_save_path: A path where rendered images will be saved (single file)
    '''
    scene = bpy.context.scene
    if apply_texture:
        apply_texture(scene=bpy.context.scene)
    setup_scene_fn(scene)
    render_save_path = setup_nodetree_fn(scene, save_dir)
    return render_save_path


def KRT_from_P(P):
    N = 3
    H = P[:,0:N]  # if not numpy,  H = P.to_3x3()

    [K,R] = rf_rq(H)
    K /= K[-1,-1]

    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = np.diag(np.sign(np.diag(K)))

    K = K * sg
    R = sg * R
    # det(R) negative, just invert; the proj equation remains same:
    if (np.linalg.det(R) < 0):
       R = -R
    # C = -H\P[:,-1]
    C = np.linalg.lstsq(-H, P[:,-1])[0]
    T = -R*C
    return K, R, T

# RQ decomposition of a numpy matrix, using only libs that already come with
# blender by default
#
# Author: Ricardo Fabbri
# Reference implementations: 
#   Oxford's visual geometry group matlab toolbox 
#   Scilab Image Processing toolbox
#
# Input: 3x4 numpy matrix P
# Returns: numpy matrices r,q
def rf_rq(P):
    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = np.linalg.qr(P[ ::-1, ::-1], 'complete')
    q = q.T
    q = q[ ::-1, ::-1]
    r = r.T
    r = r[ ::-1, ::-1]

    if (np.linalg.det(q) < 0):
        r[:,0] *= -1
        q[0,:] *= -1
    return r, q



def setup_and_render_image(task_name, basepath, view_number, view_dict, execute_render_fn, logger=None,
                           clean_up=True):
    ''' Mutates the given camera and uses it to render the image called for in 
        'view_dict'
    Args:
        task_name: task name + subdirectory to save images
        basepath: model directory
        view_number: The index of the current view
        view_dict: A 'view_dict' for a point/view
        execute_render_fn: A function which renders the desired image
        logger: A logger to write information out to
        clean_up: Whether to delete cameras after use
    Returns:
        None (Renders image)
    '''
    scene = bpy.context.scene
    camera_uuid = view_dict["camera_uuid"]
    point_uuid = view_dict["point_uuid"]
    if "camera_rotation_original" not in view_dict:
        view_dict["camera_rotation_original"] = view_dict["camera_original_rotation"]

    camera, camera_data, scene = utils.get_or_create_camera(
        location=view_dict['camera_location'],
        rotation=view_dict["camera_rotation_original"],
        field_of_view=view_dict["field_of_view_rads"],
        scene=scene,
        camera_name='RENDER_CAMERA')

    if settings.CREATE_PANOS:
        utils.make_camera_data_pano(camera_data)
        save_path = io_utils.get_file_name_for(
            dir=get_save_dir(basepath, task_name),
            point_uuid=camera_uuid,
            view_number=settings.PANO_VIEW_NAME,
            camera_uuid=camera_uuid,
            task=task_name,
            ext=io_utils.img_format_to_ext[settings.PREFERRED_IMG_EXT.lower()])
        camera.rotation_euler = Euler(view_dict["camera_rotation_original"],
                                      settings.EULER_ROTATION_ORDER)
        execute_render_fn(scene, save_path)

    elif settings.CREATE_FIXATED:
        # if settings.HYPERSIM: bpy.context.scene.camera.data.clip_end = 10000

        save_path = io_utils.get_file_name_for(
            dir=get_save_dir(basepath, task_name),
            point_uuid=point_uuid,
            view_number=view_number,
            camera_uuid=camera_uuid,
            task=task_name,
            ext=io_utils.img_format_to_ext[settings.PREFERRED_IMG_EXT.lower()])
        # Aim camera at target by rotating a known amount
        camera.rotation_euler = Euler(view_dict["camera_rotation_original"])
        camera.rotation_euler.rotate(
            Euler(view_dict["camera_rotation_from_original_to_final"]))

        execute_render_fn(scene, save_path)


    else:
        raise ('Neither settings.CREATE_PANOS nor settings.CREATE_FIXATED is specified')

    if clean_up:
        utils.delete_objects_starting_with("RENDER_CAMERA")  # Clean up

    
def get_save_dir(basepath, task_name):
    if settings.CREATE_PANOS:
        return os.path.join(basepath, 'pano', task_name)
    else:
        return os.path.join(basepath, task_name)


def run(setup_scene_fn, setup_nodetree_fn, model_dir, task_name, apply_texture_fn=None):
    ''' Runs image generation given some render helper functions 
    Args:
        stop_at: A 2-Tuple of (pt_idx, view_idx). If specified, running will cease (not cleaned up) at the given point/view'''

    utils.set_random_seed()
    logger = io_utils.create_logger(__name__)

    with Profiler("Setup", logger) as prf:
        model_info = io_utils.load_model_and_points(model_dir)
        if apply_texture_fn:
            apply_texture_fn(scene=bpy.context.scene)
        if settings.SHADE_SMOOTH:
            current_mode = bpy.context.object.mode
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.mark_sharp(clear=True)
            bpy.ops.mesh.mark_sharp(clear=True)
            bpy.ops.mesh.mark_sharp(clear=True, use_verts=True)
            bpy.ops.mesh.faces_shade_smooth()
            bpy.ops.object.mode_set(mode=current_mode)
            bpy.ops.object.shade_smooth()
        execute_render = utils.make_render_fn(setup_scene_fn, setup_nodetree_fn,
                                              logger=logger)  # takes (scene, save_dir)
        n_imgs = get_number_imgs(model_info['point_infos'])

    with Profiler('Render', logger) as pflr:
        img_number = 0
        for point_number, point_info in enumerate(model_info['point_infos']):
            for view_number, view_dict in enumerate(point_info):
                img_number += 1
                view_id = view_number if settings.CREATE_PANOS else view_dict['view_id']
                setup_and_render_image(task_name, model_dir,
                                       clean_up=True,
                                       execute_render_fn=execute_render,
                                       logger=logger,
                                       view_dict=view_dict,
                                       view_number=view_id) #view_number
                pflr.step('finished img {}/{}'.format(img_number, n_imgs))
                if settings.CREATE_PANOS:
                    break  # we only want to create 1 pano per camera
                
    return




