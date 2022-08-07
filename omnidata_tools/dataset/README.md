
<div align="center">

# Omnidata: _Omnidata Starter Dataset (OSD)_
[`Project Website`](https://omnidata.vision) &centerdot; [`Paper`](https://arxiv.org/abs/2110.04994) &centerdot; [`Github`](https://github.com/EPFL-VILAB/omnidata#readme) &centerdot; [**`>> [Data] <<`**](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/dataset#readme) &centerdot; [`Pretrained Weights`](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_tools/torch#readme) &centerdot; [`Annotator`](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_annotator#readme) &centerdot; 

**Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans (ICCV 2021)**

</div>

---

| Omnidata Starter Dataset (OSD): *standardized subsets from 1 to 2200 scenes (up to 14 million views)*|
|:---:|
|![](https://github.com/EPFL-VILAB/omnidata/raw/main/omnidata_annotator/assets/point_5.gif) <br> _Below: Download instructions and dataset information_ |
|![](https://omnidata.vision/assets/main_page/starter_dataset.jpg)|

## Table of Contents

- [Standardized Subsets](#standardized-subsets)
- [Dataset statistcs](#data-statistics)
    - [View effects of subsampling](https://omnidata.vision/designer/)
- [Label descriptions](#modalities)
- [Downloading](#downloading)
    - [Examples](#examples)
- [Pretrained Normal + Depth Models](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_tools/torch#readme) [[Upload your image](https://omnidata.vision/demo/)]

<!--   https://omnidata.vision/assets/main_page/method_web.jpg -->

## Standardized Subsets
We annotated several collections of 3D meshes in order to produce the Omnidata Starter Dataset (OSD). The result is a 14-million-image multiview dataset of indoor, outdoor, scene-level and object-level images from 2200 real and rendered scenes.


| Dataset | Train <br> (# buildings) | Val  <br> (# buildings) | Test <br> (# buildings) | Train  <br> (# images) | Val  <br> (# images)  |  Test  <br> (# images)  | Total <br> (# images)  | 
|---------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:| :-----:| 
|||||||||
| **Taskonomy** | 379 |  75 |  79  | 3,416,314  | 538,567 | 629,581  |684,052 | 
| **Replica**  |  10  | 4  | 4 | 56,783 |  23,725 |  23,889 |  4,150 | 
| **Replica + GSO**  |  10 |  4  | 4  | 107,404  | 43,450  | 42,665  | 31,167 | 
| **Hypersim**  |  365  | 46  | 46  | 59,543  | 7,386  | 7,690  | 74,619 | 
| **BlendedMVS**  |  74 |  73 | 79,023  | 16,787 |  16,766 341 |  112,576 | 
| **Habitat-Matterport**  |  800  | 100 |  - | 8,470,855  | 1,061,021  | -  | 564,328 | 
| **CLEVR**  |  1  | 0  | 0  | 60,000 | 6,000  | 6,000  | 72,000 |
|||||||||
| _Total (no CLEVR)_  |_1,905_ |  _303_  | _206_  | _12,189,922_ |  _1,690,936_  | _720,591_  |  1,434,892_ | 



You can explore a single model uploaded to a github repo [here](https://github.com/alexsax/taskonomy-sample-model-1). 

### Descriptions of the components

> **Taskonomy**: 4.5 million images of 534 buildings scanned with Matterport 3D cameras. These are real (non-rendered) images with geometry reconstructed from the Matterport SfM and MVS pipeline. The images are real, but the geometry is approximate. The following subsets of Taskonomy can be downloaded with the download tool and used with the provided dataloaders:

| Tasknonomy Subset   |      Train  <br> (# buildings)    |  Val <br> (# buildings)  |  Test <br> (# buildings)  |  Total <br> (# buildings) |
|----------:|:-------------:|:-------------:|:------:|:------:|  
| Debug |  1 | 0 | 0 | 1 | 
| Tiny |  25 | 5 | 5 | 35 | 
| Medium |  98 |  20 | 20 | 138 |  
| Full | 344 | 67 | 71 | 482 | 
| Full+ | 381 |  75 | 81 | 537 |  
> **Replica**: 18 scanned buildings with pixel-aligned geometry, but the meshes are rendered with the Habitat renderer

> **GSO + Replica**: Google Scanned Objects dataset scattered around the Replica dataset at 3 levels of clutter (in # of objects per square meter). The objects are allowed to settle in the Habitat environment and then rendered. Cameras are generated relatively close to the objects, yielding object-centric views.

> **Blended MVS**: 115k images of outdoor scenes reconstructed using the Altizure SfM + MVS pipeline. We've noticed that sometimes the aerial poses are inaccurate, which led to some images containing inaccurate geometry labeling in our pipeline. 

> **Hypersim**: 75k images from a few hundred scenes constructed by artists and RGB rendered with V-Ray. The images are high-quality and have pixel-aligned meshes, but sometimes the the mesh assets have flat screens outside the windows which leads to noisy labels in both the original Hypersim dataset and with our pipeline. 



## Data Statistics 
| Component |  Distance from Camera | # Cam. per point | Field of View	 | Cam. Pitch | Cam. Roll |
|:---------:|:---------:|:---------:|:---------:|:---------:| :---------:|
| **Taskonomy** | ![image](https://user-images.githubusercontent.com/5157485/183261932-0ed8232b-d962-4f73-ad95-15f62d69d662.png) | ![image](https://user-images.githubusercontent.com/5157485/183261960-3411284f-0484-4f1b-861b-92bb986a307d.png) | ![image](https://user-images.githubusercontent.com/5157485/183262034-64913e17-afae-435c-b621-6860f505ad9b.png) | ![image](https://user-images.githubusercontent.com/5157485/183262067-d284c7fd-b2cc-4665-8784-5838fca8ad2b.png) | ![image](https://user-images.githubusercontent.com/5157485/183262148-f1c17d58-5f39-4003-93aa-5d36e26178de.png) |
| **Replica** | ![image](https://user-images.githubusercontent.com/5157485/183261938-41d98298-b553-4f43-bc25-b93878c7f673.png) | ![image](https://user-images.githubusercontent.com/5157485/183261963-937f385c-5e06-48f7-a07c-64384227ae62.png) | ![image](https://user-images.githubusercontent.com/5157485/183262038-57acfad3-ca4f-4b81-afda-b49cb881f0db.png) | ![image](https://user-images.githubusercontent.com/5157485/183262076-d5407210-887e-4213-905d-bb64b9784f7f.png) | ![image](https://user-images.githubusercontent.com/5157485/183262153-686dd7ea-25b3-4df8-b6fd-3c046aafeaeb.png) |
| **GSO + Replica** | ![image](https://user-images.githubusercontent.com/5157485/183261941-f35899af-d776-4393-abe7-7f643c9d4879.png) | ![image](https://user-images.githubusercontent.com/5157485/183261967-28fff790-6ca1-42bd-bf01-4bcad9141224.png) | ![image](https://user-images.githubusercontent.com/5157485/183262048-cc980caf-1e01-4d00-8a29-3c8039f4c186.png) | ![image](https://user-images.githubusercontent.com/5157485/183262080-bd8fcf33-5ea0-477a-ac63-39697701eb4f.png) | ![image](https://user-images.githubusercontent.com/5157485/183262161-c639c15b-25bf-42d2-ac32-885bc9caa7aa.png) |
| **HM3D** | ![image](https://user-images.githubusercontent.com/5157485/183261945-d133853e-e360-46a7-bd93-cdb26473a02d.png) | ![image](https://user-images.githubusercontent.com/5157485/183261972-5aca5aae-9f2c-43a1-a530-09d9ab2fcfc8.png) | ![image](https://user-images.githubusercontent.com/5157485/183262059-238b5be9-0ba5-47eb-a382-ba68784ae4c1.png) | ![image](https://user-images.githubusercontent.com/5157485/183262087-0e5fd466-cbc0-4e97-8235-09a1c7272fc9.png) | ![image](https://user-images.githubusercontent.com/5157485/183262164-3dfdc063-495a-45e8-896a-c239c07690aa.png) |
 
 
_Note: You can play around with what happens if you subsample based on these statistcs and more using the [Dataset Designer Demo](https://omnidata.vision/designer/)._
 
##  Modalities 
| Per-Image Information |  Name <br> (_downloader, dataloader_)  | Description  | File Format on Disk| 
|-------------------:|---------------------|:---------------------|:---------------------|
| **RGB** | `rgb` | Color images | 3-channel 8-bit PNG| 
| **Camera Pose** | `point_info` | R, T matrices usable directly with the PyTorch3D dataloader | json |
| **Camera Intrinsics** | `point_info` | Camera projection matrix K and K_inv used for raycasting, see PyTorch3D implementation | json |
| **Correspondences (flow)** | `fragments` | Value at each pixel contains the triangle index on the scene mesh. Consistent across images. | 3-channel 8-bit png encoded here |
| **Distance (Euclidean)** | `depth_euclidean` | Raycast distance at each pixel from the camera location (R, T) to the mesh. In increments of 1/512m. | Single-channel 16-bit |
| **Depth (Z-Buffer)** | `depth_zbuffer` | Depth image using pinhole camera projection model. Not meaningful for Hypersim camera model. In increments of 1/512m. | Single-channel 16-bit |
| **Surface normals** | `normal` |  Surface normals of the mesh relative to the camera. See dataloaders for transforming to world coordinates. | 3-channel 8-bit PNG |
| **Instance Segmentation** | `semantic` |  Segmentation masks indicated instance id and semantic class, if available. | 3-channel 8-bit PNG |
| **Semantic Segmentation** | `semantic` |  Segmentation masks indicated instance id and semantic class, if available. | 3-channel 8-bit PNG |
| **2D Graphcut Segmentation** | `segm`  |  Segmentation masks indicated instance id and semantic class, if available. | 3-channel 8-bit PNG |
| **2.5D Graphcut Segmentation** | `segm` |  Segmentation masks indicated instance id and semantic class, if available. | 3-channel 8-bit PNG |
| **Edges (2D texture)** | `edge_texture`   |  Output of SciPy Canny edge detector on meshes, without nonmax suppression. | 1-channel 8-bit PNG |
| **Edges (3D occlusion)** | `edge_occlusion` |  Gaussian-blurred derivative of depth images. | 1-channel 8-bit PNG |
| **Keypoints (2D, SIFT)** | `segment_unsup2d` |  Keypoint heatmaps generated from SIFT. | 1-channel 8-bit PNG |
| **Keypoints (3D, SIFT)** | `segment_unsup25d` |  Keypoint heatmaps generated from NARF. | 1-channel 8-bit PNG |
| **Principal Curvature** | `principal_curvature` |  Radius of curvature of mesh along major and minor axes, estimated using Meshlab. Principal curvatures are encoded in the first two channels. Zero curvature is encoded as the pixel value 127 | 3-channel 8-bit PNG |
| Reshading | `reshading` | Rendered image of mesh with point light source in Blender. A function of distance and relative normals, it can be used to estimate the shading portion of intrinsic image decomposition. | 1-channel 8-bit PNG |
| **Masks (valid pixels)** | `mask_valid` | A value of 0 indicates missing geometry at this pixel, 255 indicates non-infinite depth. | 1-channel 8-bit PNG |
 

## Downloading
 ```bash
conda install -c conda-forge aria2
pip install 'omnidata-tools'

omnitools.download point_info rgb depth_euclidean mask_valid fragments \
    --components replica taskonomy \
    --subset tiny \
    --dest ./omnidata_starter_dataset/ \
    --name YOUR_NAME --email YOUR_EMAIL --agree_all
```
Run the following: (Estimated download time for [RGB + 1 Task + Masks]: 1 day) (Full dataset [30TB]: 5 days)


### Examples

```bash
omnitools.download -h  # Help
```

Download a debug RGB-D multiview dataset
```bash
omnitools.download rgb depth_euclidean mask_valid \
    --components taskonomy replica \
    --subset debug \
    --dest ./omnidata_starter_dataset/ \
    --name YOUR_NAME --email YOUR_EMAIL --agree_all
```

Download wide-baseline multiview geometry and flow (correspondence) information for 5 million images
```bash
omnitools.download point_info rgb depth_euclidean normals mask_valid fragments \
    --components taskonomy replica_gso replica hypersim \
    --subset fullplus \
    --dest ./omnidata_starter_dataset/ \
    --name YOUR_NAME --email YOUR_EMAIL --agree_all
```

Download geometry and camera information for 15 million indoor/outdoor/scene/object images
```bash
omnitools.download point_info rgb depth_euclidean normals mask_valid \
    --components taskonomy replica_gso replica hypersim hm3d blended_mvg \
    --subset fullplus \
    --dest ./omnidata_starter_dataset/ \
    --name YOUR_NAME --email YOUR_EMAIL --agree_all
```

## Citing
```
@inproceedings{eftekhar2021omnidata,
  title={Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets From 3D Scans},
  author={Eftekhar, Ainaz and Sax, Alexander and Malik, Jitendra and Zamir, Amir},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10786--10796},
  year={2021}
}
```

