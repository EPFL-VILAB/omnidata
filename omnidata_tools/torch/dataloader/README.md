<div align="center">


# Omnidata: _PyTorch3D + Pytorch-Lightning Dataloaders_
[`Project Website`](https://omnidata.vision) &centerdot; [`Paper`](https://arxiv.org/abs/2110.04994) &centerdot; [`Github`](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_tools/torch) &centerdot; [`Data`](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/dataset#readme) &centerdot; [**`>> [PyTorch Utils + Weights] <<`**](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch#readme) &centerdot;  [`Annotator`](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_annotator#readme) 

**Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets from 3D Scans (ICCV 2021)**

</div>


---

## Single- and Multi-View Dataloaders
We provide a set of modular PyTorch dataloaders in the `dataloaders` directory ([here](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch/dataloader)) that work for multiple components of the dataset or for any combination of modalities. The [notebook here](https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/00_usage_dataloader.ipynb) shows how to use the dataloader, how to load multiple overlapping views, and how to unproject the images into the same scene. The image below shows an example from the notebook, using the visualization tools in the dataloaders folder. New components datasets (e.g. those annotated with the [annotator](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_annotator)) can be added as a file in the `dataloader/component_datasets` and used with the dataloader. The current dataloaders work for Taskonomy, Replica, GSO-in-Replica, Hypersim, HM3D, and BlendedMVS++

| |
| :-------------:|
| Notebook visualization of multiple views |
| ![](https://user-images.githubusercontent.com/5157485/205193809-f5bb1759-d7b6-4157-8b60-fb595bbe57bf.png) |


### 3D, Camera Pose, Camera Intrinsics
The dataloaders are designed to be used directly with [PyTorch3d](https://pytorch3d.org/). The dataloader conventions follow that package, which is generally the same as OpenGL . To get the camera info (intrinsics K and extrinsics [R|t]), make sure to include `point_info` in the `tasks` (example in notebook)

### Multi-View Sampling
We provide several options for multi-view dataloading. You can change the number of sampled views by increasing the `num_positive` option beyond 1. The simplest (default) one simply uses the filenames to infer which images in each scene fixate on the same point--it works for Taskonomy and Replica only.

*CENTER_VISIBLE sampler*: We suggest using the `multiview_sampling_method='CENTER_VISIBLE'` sampler, which takes some anchor view and then samples new views whose center point is visible in the first image. Multiple 'hops' are allowed using the `multiview_path_len` option, so (for example--2 hops) you can sample views whose center is visible in some other image whose center is visible in the first image. You can also sample views from cameras nearby the first camera. 
Here are the options:
```python
  multiview_path_len:  int             = 1         # Number of hops
  sampled_camera_type: str             = 'BACKOFF' # 
  #    BACKOFF:   Try selecting views like the last option in 'backoff_order', and if not possible then second last, etc.
  #    DIFFERENT: New views must come from a camera different than the first, and not be fixated on the same 3D point
  #    FIXATED:   New views must come from a different camera, but are fixated on the same 3D points
  #    SAME:      New views come from the same camera as the first image, but are fixated somewhere else
  backoff_order:       List[str]       = field(default_factory=lambda: ['SAME', 'FIXATED', 'DIFFERENT'])
  sampled_camera_knn:  Optional[int]   = -1   # Sample views taken from K nearest cameras (in 3D space). -1 allows any camera.
```

The CENTER_VISIBLE sampler requires downloading the scene-level metadata that defines the view connectivity graph. The tar files contain .hdf5 files for each scene in the dataset, and we've generated these structs for Taskonomy, Replica, Hypersim, and GSO-in-Replica. The total file size is about 6.5GB for these components (5m images). You can download and untar the files using the following
```bash
# Place the untarred directories in the same directory as the main omnidata dataset. 
# pip install gdown
# gdown '1bvdgtHtKHEtSwYpYlNz2bIxilvinfC54&confirm=t'
# gdown '1avTBx5JbKj9GiGk9rWsufGppo1DxqQ-l&confirm=t'
# tar -xf scene_metadata_hs_r_gso_t.tar
# tar -xf scene_multiview_metadata_hs_r_gso_t.tar
````

### PyTorch-Lightning
We provide a pytorch-lighthning datamodule that will sample evenly from each component. That is using k components, each batch has a fraction 1/k from each of the components. The implementation is in the dataloaders folder [here](https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/dataloader/pytorch_lightning_datamodule.py).
