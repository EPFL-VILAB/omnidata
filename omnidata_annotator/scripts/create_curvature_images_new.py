"""
  Name: create_curvature_images.py
  Author: Sasha Sax, CVGL
  Desc: Creates images where color corresponds to the Gaussian curvature
    Since we need to used vertex colors to render out curvature, and the
    maximum bit-depth for vertex colors is 8-bits, we need to be somewhat
    careful about how we store the curvatures.
    What we do is store the principal curvatures K1 and K2 in the RG channels
    of the RGB image. Blue is currently unused, but is left available in case
    we need to do something clever with it later.

  Usage:
    blender -b -noaudio --enable-autoexec --python create_curvature_images.py --

  Requires (to be run):
    - generate_points.py
"""

import random
import bpy
import bmesh
import math
import os
import subprocess
import sys
from subprocess import call, check_output

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from load_settings import settings 
import create_images_utils
from create_images_utils import get_number_imgs
import io_utils
import utils
from utils import Profiler

import matplotlib
import matplotlib.cm as cm
import numpy as np
import plyfile
from plyfile import PlyData, PlyElement

TASK_NAME = "principal_curvature"
basepath = settings.MODEL_PATH
scriptpath = os.path.dirname(__file__)
utils.set_random_seed()

ACCEPTED_CURVATURE_MODES = ["GAUSSIAN_DISPLAY", "PRINCIPAL_CURVATURES"]

err_file = os.path.join(scriptpath, 'curvature_err.txt')

def main():
    global logger
    logger = io_utils.create_logger(__name__)
    

    model_fpath = os.path.join(basepath, settings.MODEL_FILE)
    k1_fpath = model_fpath.replace(".", "_k1.").replace("obj", "ply")
    k2_fpath = model_fpath.replace(".", "_k2.").replace("obj", "ply")
  

    def recolor_model(scene):
        utils.delete_all_objects_in_context() # ??

        if settings.CURVATURE_OUTPUT_MODE == "GAUSSIAN_DISPLAY":
            gausses = PlyData.read(model_fpath).elements[0]['quality']
            radius_sq = settings.MIN_CURVATURE_RADIUS ** 2
            gausses = clip_curvatures(gausses, radius_sq)
            colors = map_to_color(gausses, min_radius=settings.MIN_CURVATURE_RADIUS, type='gaussian')

        elif settings.CURVATURE_OUTPUT_MODE == "PRINCIPAL_CURVATURES":
            k1 = clip_curvatures(
                PlyData.read(k1_fpath).elements[0]['quality'],
                radius=settings.MIN_CURVATURE_RADIUS)
            k2 = clip_curvatures(
                PlyData.read(k2_fpath).elements[0]['quality'],
                radius=settings.MIN_CURVATURE_RADIUS)
            colors = map_to_color(k1, k2, min_radius=settings.MIN_CURVATURE_RADIUS, type='principal')

        else:
            raise NotImplementedError("Only modes {} are supported.".format(ACCEPTED_CURVATURE_MODES))

        bpy.ops.import_mesh.ply(filepath=k1_fpath)
        obj = bpy.context.object
        color_mesh_faces(obj, colors, coloring_type="VERTEX")
        apply_vertex_color_as_material_fn(scene)
        

    create_images_utils.run(
        set_scene_render_settings,
        setup_nodetree_for_render,
        model_dir=basepath,
        task_name=TASK_NAME,
        apply_texture_fn=recolor_model)


    return


def clip_curvatures(vals, radius=0.03):
    """ Clips principal curvatues to be in a defined range.
        A principal of curvature of k corresponds to the curvature
        (in one direction) of a sphere or radius r = 1/k. Since
        Our mesh has somewhat of a low resolution, so we don't
        want to consider curvatures much higher than some threshhold.
        vals:
            the principal curvatures to clip
        radius:
            the radius of the smallest sphere to consider, in meters
    """
    print("!!!!!!!!! {:.6f} - {:.6f}".format(vals.max(), vals.min()))
    max_val = 1.0 / radius
    vals[vals > max_val] = max_val
    vals[vals < -max_val] = -max_val
    return vals


def map_to_color(k1, k2=None, min_radius=settings.MIN_CURVATURE_RADIUS, type='principal'):
    if type == 'principal':
        # Mapping [-1/r, 1/r] -> [0, 254]
        max_val = float(2 ** settings.BLENDER_VERTEX_COLOR_BIT_DEPTH - 1) - 1.
        remapped1 = np.round(((k1 * min_radius) + 1.0) / 2. * max_val)
        remapped1 /= (max_val + 1.0)
        remapped2 = np.round(((k2 * min_radius) + 1.0) / 2. * max_val)
        remapped2 /= (max_val + 1.0)
        colors = []
        colors = [(remapped1[i], remapped2[i], 0.) for i in range(len(remapped1))]
        return colors

    elif type == 'gaussian':
        # Assume that k1 gives us the gaussian curvature
        radius_sq = min_radius ** 2
        remapped = ((k1 * radius_sq) + 1.0) / 2.0
        colors = [cm.bwr(1 - v)[:3] for v in remapped]
        return colors


import pdb


def color_mesh_faces(obj, colors=None, coloring_type="FACE"):
    coloring_type = coloring_type.upper()
    if coloring_type.upper() not in ["VERTEX", "FACE"]:
        raise ValueError('coloring_type must be one of ["VERTEX", "FACE"]')

    mesh = obj.data
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()

    """
    let us assume for sake of brevity that there is now 
    a vertex color map called  'Col'    
    """
    color_layer = mesh.vertex_colors.active.data
    if coloring_type == "VERTEX":
        vert_loop_map = {}
        # Color per vertex
        for l in mesh.loops:
            try:
                vert_loop_map[l.vertex_index].append(l.index)
            except KeyError:
                vert_loop_map[l.vertex_index] = [l.index]

        for vertex_index in vert_loop_map:
            if colors is None:
                rgb = [random.random() for k in range(3)]
            else:
                rgb = colors[vertex_index]
            for loop_index in vert_loop_map[vertex_index]:
                color_layer[loop_index].color = rgb
    elif coloring_type == "FACE":
        # Color per face
        i = 0
        for poly_idx, poly in enumerate(mesh.polygons):
            if colors is None:
                rgb = [random.random() for k in range(3)]
            else:
                rgb = colors[poly_idx]
            for idx in poly.loop_indices:
                color_layer[idx].color = rgb
                i += 1


######################
# RENDERING FUNCTIONS
######################
def apply_vertex_color_as_material_fn(scene):
    """ Use the vertex color as input to the material """
    material = bpy.data.materials.new('Curvature')
    material.use_shadeless = True
    material.use_mist = False
    material.use_vertex_color_paint = True

    # Now apply this material to the mesh
    mesh = utils.get_mesh()
    bpy.context.scene.objects.active = mesh
    bpy.ops.object.material_slot_add()
    mesh.material_slots[0].material = material


def set_scene_render_settings(scene):
    """
      Sets the render settings for speed.

      Args:
        scene: The scene to be rendered
    """
    utils.set_preset_render_settings(scene, presets=['BASE'])

    # Set passes
    scene.render.layers["RenderLayer"].use_pass_combined = True
    scene.render.layers["RenderLayer"].use_pass_z = False
    scene.render.layers["RenderLayer"].use_pass_normal = False
    scene.render.use_sequencer = False


def setup_nodetree_for_render(scene, tmpdir):
    """
        Creates the scene so that a surface normals image will be saved.
        Note that this method works, but not for Blender 2.69 which is
        the version that exists on Napoli. Therefore, prefer the other
        method 'setup_scene_for_normals_render_using_matcap'

        Args:
            scene: The scene that will be rendered
            tmpdir: The directory to save raw renders to

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

    inp = tree.nodes.new('CompositorNodeRLayers')
    image_data = inp.outputs[0]
    bpy.data.worlds["World"].horizon_color = (0.5, 0.5, 0.5)

    # Now save out the normals image and return the path
    save_path = utils.create_output_node(tree, image_data, tmpdir,
                                         color_mode='RGB',
                                         file_format=settings.PREFERRED_IMG_EXT)
    return save_path


if __name__ == "__main__":
    with Profiler(os.path.dirname(os.path.basename(__file__))):
        main()
