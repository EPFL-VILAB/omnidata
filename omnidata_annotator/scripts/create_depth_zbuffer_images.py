"""
  Name: create_depth_zbuffer_images.py
  Desc: Creates depth versions of standard RGB images by using the matterport models.
    This reads in all the point#.json files and renders the corresponding images in depth,
    where depth is defined relative to the plane normal to the camera direction.  

  Requires:
    - generate_points.py
"""

# Import these two first so that we can import other packages
import bpy
import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from load_settings import settings
import create_images_utils
import utils

TASK_NAME = 'depth_zbuffer'
basepath = settings.MODEL_PATH

def main():
    if settings.CREATE_PANOS:
        raise EnvironmentError('{} is unable to create panos.'.format(os.path.basename(__file__)))

    apply_texture_fn = None

    create_images_utils.run(
        set_render_settings,
        setup_nodetree_for_render,
        model_dir=basepath,
        task_name=TASK_NAME,
        apply_texture_fn=apply_texture_fn)


def set_render_settings(scene):
    """
      Sets the render settings for speed.

      Args:
        scene: The scene to be rendered
    """
    utils.set_preset_render_settings(scene, presets=['BASE', 'NON-COLOR'])
    scene.render.layers["RenderLayer"].use_pass_combined = False


def setup_nodetree_for_render(scene, outdir):
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
    mapv = tree.nodes.new('CompositorNodeMapValue')
    mapv.size[
        0] = 1. / settings.DEPTH_ZBUFFER_MAX_DISTANCE_METERS  # Want all the values to fit into [0, 2**(settings.COLOR_DEPTH)]
    links.new(inp.outputs[2], mapv.inputs[0])
    image_data = mapv.outputs[0]

    save_path = utils.create_output_node(tree, image_data, outdir,
                                         color_mode='BW',
                                         file_format=settings.PREFERRED_IMG_EXT,
                                         color_depth=settings.DEPTH_BITS_PER_CHANNEL)
    return save_path  # Save it out


if __name__ == "__main__":
    with utils.Profiler("create_depth_zbuffer_images.py"):
        main()
