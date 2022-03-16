"""
  Name: create_semantic_images.py
  Desc: Creates RGB images using texture UV maps.
"""

# Import these two first so that we can import other packages
from __future__ import division

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import io_utils

# Import remaining packages
import bpy
from bpy import context, data, ops
import bpy_extras.mesh_utils
import bmesh
import trimesh
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
TASK_NAME = 'rgb'

utils.set_random_seed()
basepath = settings.MODEL_PATH

def main():
    global basepath
    global TASK_NAME
    utils.delete_all_objects_in_context()

    model = io_utils.import_mesh(basepath, typ='RGB')

    if settings.CREATE_PANOS:
        engine='CYCLES'
    else:
        engine = 'BI'

    bpy.context.scene.render.engine = 'BLENDER_RENDER'

    utils.make_materials_shadeless(engine='BI')
    # add_face_materials(engine, model)

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


def create_texture_from_img(filepath):
    """
      Creates a texture of the given image.

      Args:
        filepath: A string that contains the path to the Image

      Returns:
        texture: A texture that contains the given Image
    """
    texture = bpy.data.textures.new("ImageTexture", type='IMAGE')
    img = bpy.data.images.load(filepath)
    texture.image = img
    # To bleed the img over the seams
    texture.extension = 'EXTEND'
    # For sharp edges
    texture.use_mipmap = False
    texture.use_interpolation = False
    #   texture.filter_type = 'BOX'
    texture.filter_size = 0.80
    return texture

def add_face_materials(engine, mesh):
    """
      Read the texture from a png file, and apply it to the mesh.
      Args:
        model: The model in context after loading the .ply
        engine: The render engine
    """
    context_obj = bpy.context.object
    context_obj_data = context_obj.data
    bpy.context.scene.objects.active = context_obj

    # texture_image = bpy.data.images.load(os.path.join(basepath, settings.TEXTURE_FILE))

    for idx, mat in enumerate(bpy.data.materials):
        mat_name = mat.name
        if not mat_name.endswith('.jpg') and not mat_name.endswith('.png'): continue
        texture = create_texture_from_img(os.path.join(basepath, mat_name))
        material = utils.create_material_with_texture(texture, name=mat_name)
        material.use_shadeless = True
        bpy.ops.object.mode_set(mode='OBJECT')
        context_obj_data.materials.append(material)

    bpy.types.SpaceView3D.show_textured_solid = True


    


'''
    RENDER
'''


def render_rgb_img(scene, save_path):
    """
      Renders an image from the POV of the camera and save it out
      Args:
        scene: A Blender scene that the camera will render
        save_path: Where to save the image
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
      Creates the scene so that a rgb image will be saved.
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
    with Profiler("create_rgb_images.py"):
        main()


# ./omnidata-annotate.sh --model_path=/model --task=rgb_UV with MODEL_FILE=TEEsavR23oF.obj