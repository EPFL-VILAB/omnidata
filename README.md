<div align="center">

# Omni ↦ Data (Steerable Datasets)
# -- Under Construction --
**A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans**

[`Project Website`](https://omnidata.vision) &centerdot; [`Docs`](//docs.omnidata.vision) &centerdot; [`Annotator Repo`](https://github.com/Ainaz99/omnidata-annotator) &centerdot; [`Starter Data`](//docs.omnidata.vision/starter_dataset.html) &centerdot;  [**`>> [Tools] <<`**](https://github.com/Ainaz99/omnidata-tools) &centerdot; [`Paper Code`](https://github.com/Ainaz99/Omnidata)

</div>

---

Table of Contents
=================
   * [Omnidata Annotator](#omnidata-annotator)
   * [Omnidata Tools](#omnidata-tools)
   * [Omnidata Documentation](#omnidata-documentation)
   * [Paper Code](#paper-code)
   * [Citing](#citation)

# Omnidata Annotator
![](./assets/point_5.gif)
[`Omnidata-annotator`](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_annotator) contains the dockerized annotator pipeline introduced in the paper. Omnidata annotator bridges the gap between 3D scans and static vision datasets by creating "steerable" multi-task datasets with 21 different mid-level cues from 3D meshes. It generates the data with as many images and cameras as desired to cover the space. The rendering pipeline offers control over the sampling and generation process, and different dataset design choices such as camera parameters. 13 of the 21 mid-level cues are listed below:
```bash
RGB (8-bit)              Surface Normals (8-bit)     Principal Curvature (8-bit)
Re(shading) (8-bit)      Depth Z-Buffer (16-bit)     Depth Euclidean (16-bit)
Texture Edges (16-bit)   Occlusion Edges (16-bit)    Keypoints 2D (16-bit)
Keypoints 3D (16-bit)    2D Segmentation (8-bit)     2.5D Segmentation (8-bit)
Semantic Segmentation (8-bit)
```

# Omnidata Tools
![](./assets/depth_to_norm.gif)
[`Omnidata-tools`](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_tools) includes [strong pretrained models](https://docs.omnidata.vision/pretrained.html#Pretrained-Models) for depth and surface normal estimation, [training code](//docs.omnidata.vision/training.html), [dataloaders](https://docs.omnidata.vision/dataloaders.html), starter dataset [download and upload utilities](//docs.omnidata.vision/omnitools.html), the first publicly [available implementation](https://docs.omnidata.vision/training.html#MiDaS-Implementation) of [MiDaS training code](https://github.com/isl-org/MiDaS), an implementation of the [3D image refocusing augmentation](https://docs.omnidata.vision/training.html#3D-Depth-of-Field-Augmentation) introduced in the paper, and more (detailed in the [docs](//docs.omnidata.vision)).

**Install this package:** `pip install 'omnidata-tools'` <br>
**Documentation**: [https://docs.omnidata.vision](//docs.omnidata.vision) for details of this package.  <br>
**Project Overview**: The [project website](https://omnidata.vision) or the [ICCV21 paper](https://omnidata.vision/#paper) provide a broad overview of the project.

# Citation
If you find the code or models useful, please cite the paper:
```
@inproceedings{eftekhar2021omnidata,
  title={Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets From 3D Scans},
  author={Eftekhar, Ainaz and Sax, Alexander and Malik, Jitendra and Zamir, Amir},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10786--10796},
  year={2021}
}
```
