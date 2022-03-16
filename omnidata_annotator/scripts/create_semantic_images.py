"""
  Name: create_semantic_images.py
  Desc: Creates semantically tagged versions standard RGB images by using the matterport models and 
    semantic labels.
    This reads in all the point#.json files and rendering the corresponding images with semantic labels. 

"""

# Import these two first so that we can import other packages
from __future__ import division

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import io_utils

# Import remaining packages
import bpy
import bpy_extras.mesh_utils
import bmesh
from collections import defaultdict, Counter
import glob
import json
import math
from mathutils import Vector, Euler, Color
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
TASK_NAME = 'semantic'

utils.set_random_seed()
basepath = settings.MODEL_PATH



def get_face_semantics():
    """
      Get mesh face colors.

      Returns:
        face_to_color: A Dict of face index to colors.
        colors: List of all colors in mesh faces.
    """
    path_in = os.path.join(basepath, settings.SEMANTIC_MODEL_FILE)

    file_in = PlyData.read(path_in)
    face_colors = file_in.elements[1]['color']

    face_to_color = {}
    colors = set()
    for f_idx, f_color in enumerate(face_colors):
        color = f_color / 255.
        colors.add(tuple(color))
        face_to_color[f_idx] = color

    return face_to_color, list(colors)
    

def main():
    global basepath
    global TASK_NAME
    utils.delete_all_objects_in_context()

    model = io_utils.import_mesh(basepath, typ='SEMANTIC')

    if settings.CREATE_PANOS:
        engine='CYCLES'
    else:
        engine = 'BI'

    semantically_annotate_mesh(engine, model)

    point_infos = io_utils.load_saved_points_of_interest(basepath)

    # render + save
    for point_info in point_infos:
        for view_number, view_dict in enumerate(point_info):
            view_id = view_number if settings.CREATE_PANOS else view_dict['view_id']
            setup_and_render_image(TASK_NAME, basepath,
                                   clean_up=True,
                                   execute_render_fn=render_semantic_img,
                                   logger=None,
                                   view_dict=view_dict,
                                   view_number=view_id)

            if settings.CREATE_PANOS:
                    break  # we only want to create 1 pano per camera


'''
    SEMANTICS
'''


def add_materials_to_mesh(materials_dict, mesh):
    bpy.context.scene.objects.active = mesh
    materials_idxs = {}  # defaultdict( dict )
    for label, mat in materials_dict.items():
        bpy.ops.object.material_slot_add()
        mesh.material_slots[-1].material = mat
        materials_idxs[label] = len(mesh.material_slots) - 1
    return materials_dict, materials_idxs



def build_materials_dict(engine, colors):
    '''
    Args:
        colors: A list of all mesh face colors.
    Returns:
        materials_dict: A dict: materials_dict[ color idx ] -> material

    '''

    materials_dict = {}
    for color_idx, color in enumerate(colors):
        materials_dict[color_idx] = utils.create_material_with_color(color, name=str(color_idx),
                                                                       engine=engine)
    return materials_dict


def semantically_annotate_mesh(engine, mesh):

    with Profiler("Read semantic annotations") as prf:
        face_to_color, face_colors = get_face_semantics()


    materials_dict = build_materials_dict(engine, face_colors)

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
            color = face_to_color[i]
            color_idx = face_colors.index(tuple(color))
            face.material_index = materials_idxs[color_idx]  # Assing random material to face

        mesh.data.update()
        bpy.ops.object.mode_set(mode='OBJECT')


'''
    RENDER
'''


def render_semantic_img(scene, save_path):
    """
      Renders an image from the POV of the camera and save it out

      Args:
        scene: A Blender scene that the camera will render
        save_path: Where to save the image
    
    """
    save_path_dir, img_filename = os.path.split(save_path)
    with Profiler("Render") as prf:

        utils.set_preset_render_settings(scene, presets=['BASE', 'NON-COLOR'])
        render_save_path = setup_scene_for_semantic_render(scene, save_path_dir)
        prf.step("Setup")

        bpy.ops.render.render()
        prf.step("Render")

    with Profiler("Saving") as prf:
        shutil.move(render_save_path, save_path)


def setup_scene_for_semantic_render(scene, outdir):
    """
      Creates the scene so that a depth image will be saved.

      Args:
        scene: The scene that will be rendered
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
    with Profiler("create_semantic_images.py"):
        main()
