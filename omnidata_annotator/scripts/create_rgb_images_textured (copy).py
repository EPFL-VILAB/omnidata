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

    model = io_utils.import_mesh(basepath)

    if settings.CREATE_PANOS:
        engine='CYCLES'
    else:
        engine = 'BI'

    # add_face_materials(engine, model)

    point_infos = io_utils.load_saved_points_of_interest(basepath)

    ########################3
    bpy.context.scene.objects.active = model
    model.select = True
    mat = bpy.data.materials.new('material_1')
    model.active_material = mat
    mat.use_vertex_color_paint = True
    bpy.ops.paint.vertex_paint_toggle()

    scn = bpy.context.scene
    if len(bpy.context.active_object.data.materials) == 0:
        bpy.context.active_object.data.materials.append(bpy.data.materials['Material'])
        print("!!!! if")
    else:
        bpy.context.active_object.data.materials[0] = bpy.data.materials['Material']
        print("!!!! else")

    # scn.render.alpha_mode = 'TRANSPARENT'
    # bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True

    #####################3333
    # print("!!!!!!!!!!!!1 ", model.name)
    # # model.select_set(True)
    # # bpy.data.objects[model.name].select_set(True)
    # bpy.ops.paint.vertex_paint_toggle()

    # #bpy.context.area.ui_type = 'ShaderNodeTree'

    # #bpy.ops.material.new()

    # mat = bpy.data.materials.get("Material")
    # print("!!!!!!!!!!!!! mar: ", mat)

    # if len(bpy.context.active_object.data.materials) == 0:
    #     bpy.context.active_object.data.materials.append(bpy.data.materials['Material'])
    #     print("!!!! if")
    # else:
    #     bpy.context.active_object.data.materials[0] = bpy.data.materials['Material']
    #     print("!!!! else")

    # if mat:
    #     bpy.context.scene.use_nodes = True
    #     mat.node_tree.nodes.new("ShaderNodeVertexColor")
    #     mat.node_tree.links.new(mat.node_tree.nodes[2].outputs['Color'], mat.node_tree.nodes[1].inputs['Base Color'])


    # # bpy.context.scene.render.filepath = '~/Desktop/photos/img.jpg'
    # # bpy.context.scene.render.engine = 'CYCLES'
    # # bpy.ops.render.render('INVOKE_DEFAULT', write_still=True)
    ############################

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




def add_face_materials(engine, mesh):
    """
      Read the texture from a png file, and apply it to the mesh.
      Args:
        model: The model in context after loading the .ply
        engine: The render engine
    """
    texture_image = bpy.data.images.load(os.path.join(basepath, settings.TEXTURE_FILE))
    image_texture = bpy.data.textures.new('export_texture', type = 'IMAGE')
    image_texture.image = texture_image
    image_material = bpy.data.materials.new('TextureMaterials')
    image_material.use_shadeless = True

    material_texture = image_material.texture_slots.add()
    material_texture.texture = image_texture
    material_texture.texture_coords = 'UV'
    bpy.ops.object.mode_set(mode='OBJECT')
    context_obj = bpy.context.object
    context_obj_data = context_obj.data
    context_obj_data.materials.append(image_material)
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
        # render_save_path = setup_scene_for_rgb_render(scene, save_path_dir)

        ident = str(uu.uuid4())
        ext = utils.img_format_to_ext[settings.PREFERRED_IMG_EXT.lower()]
        temp_filename = "{0}0001.{1}".format(ident, ext)
        render_save_path = os.path.join(save_path_dir, temp_filename)

        prf.step("Setup")

        print("******************* ", render_save_path, save_path)
        scene.render.filepath = os.path.join(temp_filename)

        bpy.ops.render.render(write_still=True)
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