"""
  Name: create_semantic_images.py
  Author: Sasha Sax, CVGL
  Desc: Creates semantically tagged versions standard RGB images by using the matterport models and 
    semantic labels.
    This reads in all the point#.json files and rendering the corresponding images with semantic labels. 

  Usage:
    blender -b -noaudio -enable-autoexec --python create_semantic_images.py --
"""

# Import these two first so that we can import other packages
from __future__ import division

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

#
import io_utils

# Import remaining packages
import bpy
import bpy_extras.mesh_utils
import bmesh
import trimesh
from collections import defaultdict, Counter
import glob
import json
import math
from mathutils import Vector, Euler, Color
# from matplotlib import cm
import numpy as np
import random
import settings
import shutil  # Temporary dir
import time
import tempfile  # Temporary dir
import utils
import uuid as uu
from utils import Profiler
from plyfile import *
import numpy as np
from scipy.signal import find_peaks
from create_images_utils import *

SCRIPT_DIR_ABS_PATH = os.path.dirname(os.path.realpath(__file__))
TASK_NAME = 'rgb_new'

utils.set_random_seed()
basepath = settings.MODEL_PATH



def get_face_colors():
    """
      Find colors for each face.

      Returns:
        objects: A list of face colors.
    """

    path_in = os.path.join(basepath, settings.MODEL_FILE)

    file_in = PlyData.read(path_in)
    vertices_in = file_in.elements[0]
    faces_in = file_in.elements[1]

    v_r = vertices_in['red'].tolist()
    v_g = vertices_in['green'].tolist()
    v_b = vertices_in['blue'].tolist()
    v_alpha = None
    try:
        v_alpha = vertices_in['alpha'].tolist()
    except:
        pass
    if True:
        v_colors = [[v_r[i], v_g[i], v_b[i]] for i in range(len(v_r))]
    else:
        v_colors = [[v_r[i], v_g[i], v_b[i], v_alpha[i]] for i in range(len(v_r))]

    faces  = [f[0] for f in faces_in]
    face_colors = trimesh.visual.color.vertex_to_face_color(v_colors, faces) / 255.

    all_colors = set()
    # face_to_color_idx = {}
    for face_idx, color in enumerate(face_colors):
        all_colors.add(tuple(color))
        # face_to_color_idx[face_idx] = list(all_colors).index(tuple(color))

    print("*****", len(list(all_colors)), flush=True)
    print("!!!!", face_colors[1], flush=True)

    return face_colors, list(all_colors)


def main():
    global basepath
    global TASK_NAME
    utils.delete_all_objects_in_context()

    model = io_utils.import_mesh(basepath)

    if settings.CREATE_PANOS:
        engine='CYCLES'
    else:
        engine = 'BI'
    add_face_materials(engine, model)

    point_infos = io_utils.load_saved_points_of_interest(basepath)

    # render + save
    for point_info in point_infos:
        for view_number, view_dict in enumerate(point_info):
            view_id = view_number if settings.CREATE_PANOS else view_dict['view_id']
            setup_and_render_image(TASK_NAME, basepath,
                                   clean_up=True,
                                   execute_render_fn=render_rgb_img,
                                   logger=None,
                                   view_dict=view_dict,
                                   view_number=view_id)

            if settings.CREATE_PANOS:
                    break  # we only want to create 1 pano per camera



def add_materials_to_mesh(materials_dict, mesh):
    bpy.context.scene.objects.active = mesh
    materials_idxs = {}  # defaultdict( dict )
    for label, mat in materials_dict.items():
        bpy.ops.object.material_slot_add()
        mesh.material_slots[-1].material = mat
        materials_idxs[label] = len(mesh.material_slots) - 1
    return materials_dict, materials_idxs



def build_materials_dict(engine, all_colors):
    '''
    Args:
        colormap: A function that returns a color for a face

    Returns:
        materials_dict: A dict: materials_dict[ face index ] -> material

    '''

    materials_dict = {}
    for color_idx, color in enumerate(all_colors):
        materials_dict[color] = utils.create_material_with_color(color[:3], name=str(color_idx),
                                                                       engine=engine)
        if color_idx % 1000 == 0:
            print(color_idx, flush=True)
    print("!!!!!!!! done")
    return materials_dict


def add_face_materials(engine, mesh):


    with Profiler("Read face colors") as prf:
        face_colors, all_colors = get_face_colors()
        

    materials_dict = build_materials_dict(engine, all_colors)

    # create materials
    with Profiler('Create materials on mesh'):
        _, materials_idxs = add_materials_to_mesh(materials_dict, mesh)

    bpy.context.scene.objects.active = mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(mesh.data)
    bm.select_mode = {'FACE'}  # Go to face selection mode

    # Deselect all faces
    for face in bm.faces:
        face.select_set(False)
    mesh.data.update()
    bm.faces.ensure_lookup_table()

    with Profiler("Applying materials") as prf:
        # Count the votes and apply materials
        for i, face in enumerate(bm.faces):  # Iterate over all of the object's faces
            face.material_index = materials_idxs[tuple(face_colors[i])]  # Assing random material to face

        mesh.data.update()
        bpy.ops.object.mode_set(mode='OBJECT')


'''
    RENDER
'''


def render_rgb_img(scene, save_path):
    """
      Renders an image from the POV of the camera and save it out

      Args:
        camera: A Blender camera already pointed in the right direction
        camera_data:
        scene: A Blender scene that the camera will render
        save_path: Where to save the image
        model: The model in context after loading the .ply
        view_dict: The loaded view dict from point_uuid.json
    """
    save_path_dir, img_filename = os.path.split(save_path)
    with Profiler("Render") as prf:

        utils.set_preset_render_settings(scene, presets=['BASE', 'NON-COLOR'])
        render_save_path = setup_scene_for_rgb_render(scene, save_path_dir)
        prf.step("Setup")

        bpy.ops.render.render()
        prf.step("Render")

    with Profiler("Saving") as prf:
        shutil.move(render_save_path, save_path)


def setup_scene_for_rgb_render(scene, outdir):
    """
      Creates the scene so that a depth image will be saved.

      Args:
        scene: The scene that will be rendered
        camera: The main camera that will take the view
        model: The main model
        outdir: The directory to save raw renders to

      Returns:
        save_path: The path to which the image will be saved
    """
    # Use node rendering for python control
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links

    # Make sure there are no existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    #  Set up a renderlayer and plug it into our remapping layer
    inp = tree.nodes.new('CompositorNodeRLayers')

    if (bpy.app.version[1] >= 70):  # Don't apply color transformation -- changed in Blender 2.70
        scene.view_settings.view_transform = 'Raw'
        scene.sequencer_colorspace_settings.name = 'Non-Color'

    # Save it out
    if outdir:
        out = tree.nodes.new('CompositorNodeOutputFile')
        ident = str(uu.uuid4())
        out.file_slots[0].path = ident
        out.base_path = outdir
        # out.format.color_mode = 'BW'
        # out.format.color_depth = settings.DEPTH_BITS_PER_CHANNEL
        out.format.color_mode = 'RGB'
        out.format.color_depth = settings.COLOR_BITS_PER_CHANNEL
        out.format.file_format = settings.PREFERRED_IMG_EXT.upper()
        links.new(inp.outputs[0], out.inputs[0])
        ext = utils.img_format_to_ext[settings.PREFERRED_IMG_EXT.lower()]
        temp_filename = "{0}0001.{1}".format(ident, ext)
        return os.path.join(outdir, temp_filename)
    else:
        out = tree.nodes.new('CompositorNodeComposite')
        links.new(inp.outputs[0], out.inputs[0])
        return None


if __name__ == '__main__':
    with Profiler("create_rgb_images.py"):
        main()
