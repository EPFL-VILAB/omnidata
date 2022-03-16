"""
  Name: create_reshading_images.py
  Desc: Creates reshading version of standard RGB images.
    This reads in all the point#.json files and renders the corresponding reshading images. 
    Given an RGB image, the label is the shading function S that results from having a single point 
    light at the camera origin, and S is multiplied by a constant fixed albedo.

  Requires:
    - generate_points.py
"""

import bpy
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from load_settings import settings
import create_images_utils
import io_utils
import utils
from profiler import Profiler

TASK_NAME = 'reshading'
basepath = settings.MODEL_PATH

def main():
    if settings.CREATE_PANOS:
        raise EnvironmentError('{} is unable to create panos.'.format(os.path.basename(__file__)))

    def make_materials_diffuse(scene=None):
        utils.make_materials_diffuse()
        if settings.SHADE_SMOOTH:
            bpy.ops.object.shade_smooth()

    apply_texture_fn = make_materials_diffuse

    create_images_utils.run(
        set_render_settings,
        setup_scene_for_semantic_render,
        model_dir=basepath,
        task_name=TASK_NAME,
        apply_texture_fn=apply_texture_fn)


'''
    RENDER
'''


def setup_scene_for_semantic_render(scene, outdir):
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
    obj_camera = bpy.context.scene.camera

    # Create new lamp datablock
    if "Lamp_albedo" not in bpy.data.lamps.keys():
        lamp_data = bpy.data.lamps.new(name="Lamp_albedo", type='POINT')
        lamp_data.energy = settings.LAMP_ENERGY
        lamp_data.use_specular = False
    else:
        lamp_data = bpy.data.lamps["Lamp_albedo"]

    # Create new object with our lamp datablock
    if "Lamp_albedo" not in bpy.data.objects.keys():
        lamp_object = bpy.data.objects.new(name="Lamp_albedo", object_data=lamp_data)
        scene.objects.link(lamp_object)
    else:
        lamp_object = bpy.data.objects["Lamp_albedo"]
    lamp_object.location = obj_camera.location
    # lamp_object.cycles.max_bounce = 1

    lamp_object.data.falloff_type = settings.LAMP_FALLOFF
    lamp_object.data.distance = settings.LAMP_HALF_LIFE_DISTANCE

    # And finally select it make active
    lamp_object.select = True
    scene.objects.active = lamp_object

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

    image_data = inp.outputs[0]
    save_path = utils.create_output_node(tree, image_data, outdir,
                                         color_mode='RGB',
                                         file_format=settings.PREFERRED_IMG_EXT,
                                         color_depth=settings.COLOR_BITS_PER_CHANNEL)

    return save_path 


def set_render_settings(scene):
    """
      Sets the render settings for speed.
      Args:
        scene: The scene to be rendered
    """
    # GPU rendering

    scene.cycles.device = settings.CYCLES_DEVICE

    bpy.types.WorldLighting.indirect_bounces = 0
    scene.render.use_sss = False
    scene.render.use_textures = False
    #   scene.render.use_envmaps = False
    #   scene.render.use_raytrace = False
    #   scene.render.use_antialiasing = False
    #   scene.render.use_shadows = False
    #   scene.display_settings.display_device = 'None'
    scene.sequencer_colorspace_settings.name = 'Non-Color'


    # Quality settings
    scene.render.resolution_percentage = 100
    scene.render.tile_x = settings.TILE_SIZE
    scene.render.tile_y = settings.TILE_SIZE
    scene.render.image_settings.color_mode = 'BW'
    scene.render.image_settings.color_depth = settings.COLOR_BITS_PER_CHANNEL
    scene.render.image_settings.file_format = settings.PREFERRED_IMG_EXT.upper()


if __name__ == "__main__":
    with Profiler(os.path.dirname(os.path.basename(__file__))):
        main()
