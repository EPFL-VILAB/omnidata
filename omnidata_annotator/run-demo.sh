#!/usr/bin/env bash


./omnidata-annotate.sh --model_path=/model --task=points \
   with GENERATE_CAMERAS=True     SCENE=True \
        MIN_CAMERA_HEIGHT=1       MAX_CAMERA_ROLL=10 \
        MIN_CAMERA_DISTANCE=1     MIN_CAMERA_DISTANCE_TO_MESH=0.3 \
        MIN_VIEWS_PER_POINT=3     NUM_POINTS=12 \
        MODEL_FILE=mesh.ply       POINT_TYPE=CORRESPONDENCES


./omnidata-annotate.sh --model_path=/model --task=rgb \
    with RGB_MODEL_FILE=mesh.obj  CREATE_FIXATED=True \
         OBJ_AXIS_FORWARD=Y       OBJ_AXIS_UP=Z  


./omnidata-annotate.sh --model_path=/model --task=normal \
    with MODEL_FILE=mesh.ply  CREATE_FIXATED=True


./omnidata-annotate.sh --model_path=/model --task=depth_zbuffer \
    with MODEL_FILE=mesh.ply  DEPTH_ZBUFFER_MAX_DISTANCE_METERS=8

./omnidata-annotate.sh --model_path=/model --task=depth_euclidean \
    with MODEL_FILE=mesh.ply  DEPTH_EUCLIDEAN_MAX_DISTANCE_METERS=8

./omnidata-annotate.sh --model_path=/model --task=reshading \
    with MODEL_FILE=mesh.ply  LAMP_ENERGY=2.5

./omnidata-annotate.sh --model_path=/model --task=keypoints2d

./omnidata-annotate.sh --model_path=/model --task=keypoints3d \
    with KEYPOINT_SUPPORT_SIZE=0.3

./omnidata-annotate.sh --model_path=/model --task=edge2d \
    with CANNY_RGB_BLUR_SIGMA=0.5

./omnidata-annotate.sh --model_path=/model --task=edge3d
  with EDGE_3D_THRESH=None

./omnidata-annotate.sh --model_path=/model --task=segment2d \
  with SEGMENTATION_2D_BLUR=3     SEGMENTATION_2D_CUT_THRESH=0.005  \
       SEGMENTATION_2D_SCALE=500  SEGMENTATION_2D_SELF_EDGE_WEIGHT=2

./omnidata-annotate.sh --model_path=/model --task=segment25d \
  with SEGMENTATION_2D_SCALE=200        SEGMENTATION_25D_CUT_THRESH=1  \
       SEGMENTATION_25D_DEPTH_WEIGHT=2  SEGMENTATION_25D_NORMAL_WEIGHT=1 \
       SEGMENTATION_25D_EDGE_WEIGHT=10
