"""
  Name: settings.py
  Desc: Contains all the settings that our scripts will use.

  Usage: for import only
"""
from math import pi
import math
from multiprocessing import cpu_count
from platform import platform
import sys


# -----Images -----
CREATE_FIXATED = True
CREATE_PANOS = False
CREATE_TRAJECTORY = False
PANO_VIEW_NAME = 'equirectangular'
PREFERRED_IMG_EXT = 'PNG'  

# -----File paths  -----
MESHLAB_SERVER_PATH = 'meshlabserver'
CAMERA_POSE_FILE = "camera_poses.json"

MODEL_PATH = ''
MODEL_FILE = "mesh.ply"  
SEMANTIC_MODEL_FILE = "mesh.obj"       
RGB_MODEL_FILE = "mesh.obj"     


# -----Render settings and performance -----
RESOLUTION = 512
RESOLUTION_X = 512                      #1024:hypersim
RESOLUTION_Y = 512                      #768:hypersim
SENSOR_HEIGHT = 20                      #18:clevr , 20:all
SENSOR_WIDTH = 20                       #32:clevr , 20:all
TILE_SIZE = 128
PANO_RESOLUTION = (2048, 1024)
MAX_CONCURRENT_PROCESSES = cpu_count()
SHADE_SMOOTH = False
OBJ_AXIS_FORWARD = 'Y'
OBJ_AXIS_UP = 'Z'     


#############################
# Task settings 
#############################

# -----Cameras------
GENERATE_CAMERAS = True
SCENE=True
MAX_CAMERA_ROLL = 10        # in degrees

# camera sampling in the building
MIN_CAMERA_DISTANCE = 0.5   # in meters
MIN_CAMERA_HEIGHT = 0.2       # in meters
MAX_CAMERA_HEIGHT = 2       # in meters
MIN_CAMERA_DISTANCE_TO_MESH = 0.1  # in meters

FLOOR_THICKNESS = 0.25      # in meters
FLOOR_HEIGHT = 2            # in meters

# camera sampling on a sphere surrounding the mesh
NUM_CAMERAS = 15
SPHERE_SCALING_FACTOR = 2
# MIN_DISTANCE_TO_OBJECT = 0.2
# MAX_DISTANCE_TO_OBJECT = 1

# -----Points------
POINT_TYPE = 'CORRESPONDENCES' # 'CORRESPONDENCES' or 'SWEEP' : The basis for how points are generated
NUM_POINTS = None
POINTS_PER_CAMERA = 5
MIN_VIEWS_PER_POINT = 1
MAX_VIEWS_PER_POINT = -1
STOP_VIEW_NUMBER = -1   # Generate up to (and including) this many views. -1 to disable.

# -----RGB -----
USE_TEXTURE=True
TEXTURE_FILE = 'texture.png'
TEXTURE_FOLDER = 'textures'

# -----Color and depth -----
BLENDER_VERTEX_COLOR_BIT_DEPTH = 8
COLOR_BITS_PER_CHANNEL = '8'  # bits per channel. PNG allows 8, 16.
DEPTH_BITS_PER_CHANNEL = '16'  # bits per channel. PNG allows 8, 16.
# With 128m and 16-bit channel, has sensitivity 1/512m (128 / 2^16)
DEPTH_ZBUFFER_MAX_DISTANCE_METERS = 128    #64:clevr , 128:all, 0.5:google_scanned  
# With 128m and 16-bit channel, has sensitivity 1/512m (128 / 2^16)
DEPTH_EUCLIDEAN_MAX_DISTANCE_METERS = 128  #64:clevr , 128:all, 0.5:google_scanned 

# -----Curvature------
MIN_CURVATURE_RADIUS = 0.03           #0.0001:clevr , 0.03:all  # in meters
CURVATURE_OUTPUT_MODE = "PRINCIPAL_CURVATURES"  #PRINCIPAL_CURVATURES, GAUSSIAN_DISPLAY
K1_MESHLAB_SCRIPT = "meshlab_principal_curvatures_k1.mlx"  # Settings can be edited directly in this XML file
K2_MESHLAB_SCRIPT = "meshlab_principal_curvatures_k2.mlx"  # Settings can be edited directly in this XML file
K1_PYMESHLAB_SCRIPT = "pymeshlab_principal_curvatures_k1.mlx"
K2_PYMESHLAB_SCRIPT = "pymeshlab_principal_curvatures_k2.mlx"

FILTER_SCALE = 0.1
MAX_PROJ_ITERS = 35
 

# -----Edge------
EDGE_3D_THRESH = None  # 0.01

CANNY_RGB_BLUR_SIGMA = 2.0              # 1.0:clevr, 1.0:google_scanned, 1.0:carla, 3.0:all
CANNY_RGB_MIN_THRESH = None  # 0.1
CANNY_RGB_MAX_THRESH = None  # 0.8
CANNY_RGB_USE_QUANTILES = True

# -----Keypoint------
# How many meters to use for the diameter of the search radius
# The author suggests 0.3 for indoor spaces:
#   http://www.pcl-users.org/NARF-Keypoint-extraction-parameters-td2874685.html
KEYPOINT_SUPPORT_SIZE = 0.2     #0.3:all , 0.3:clevr, 0.2:replica-gso, 1.0:carla

# Applies a blur after keypoint detection, radius
KEYPOINT_BLUR_RADIUS = 5

# ----Reshading-----
LAMP_ENERGY = 2                  #2.5:all , 10:clevr, 1.5:google_scanned, replica-gso:2
LAMP_HALF_LIFE_DISTANCE = 8.0      
LAMP_FALLOFF = 'INVERSE_SQUARE'  

# ----Segmentation----
SEGMENTATION_2D_BLUR = 3.0
SEGMENTATION_2D_SCALE = 200
SEGMENTATION_2D_CUT_THRESH = 0.005
SEGMENTATION_2D_SELF_EDGE_WEIGHT = 2.0

SEGMENTATION_25D_SCALE = 200
SEGMENTATION_25D_DEPTH_WEIGHT = 2.
SEGMENTATION_25D_NORMAL_WEIGHT = 1.
SEGMENTATION_25D_EDGE_WEIGHT = 10.
SEGMENTATION_25D_CUT_THRESH = 1.0
SEGMENTATION_25D_SELF_EDGE_WEIGHT = 1.0

# ----Vanishing points-----
OVERRIDE_MATTERPORT_MODEL_ROTATION = False  # Whether to try to find model rotation with BBoxes


# ----Field of view----
FIELD_OF_VIEW_MIN_RADS = math.radians(30)
FIELD_OF_VIEW_MAX_RADS = math.radians(125)
FIELD_OF_VIEW_MATTERPORT_RADS = math.radians(90)
LINE_OF_SITE_HIT_TOLERANCE = 0.001  # Matterport has 1 unit = 1 meter, so 0.001 is 1mm


# DO NOT CHANGE -- effectively hardcoded
CYCLES_DEVICE = 'GPU'  # Not yet implemented!
EULER_ROTATION_ORDER = 'XYZ'  # Not yet implemented!

RANDOM_SEED = 42  # None to disable

DEPTH_ZBUFFER_SENSITIVITY = float(DEPTH_ZBUFFER_MAX_DISTANCE_METERS) / float(2 ** int(DEPTH_BITS_PER_CHANNEL))
DEPTH_EUCLIDEAN_SENSITIVITY = float(DEPTH_EUCLIDEAN_MAX_DISTANCE_METERS) / float(2 ** int(DEPTH_BITS_PER_CHANNEL))