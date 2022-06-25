# Omnidata GitHub Documentation
> <strong>Omnidata (i.e. steerable datasets) is a method of generating multi-modal multi-task computer vision datasets from 3D meshes</strong>: <br> [ [Main Website](https://omnidata.vision/) ] [ [Arxiv](https://arxiv.org/abs/2110.04994) ] -- [ Demos: <a href='//omnidata.vision/demo/'>Pretrained Models</a>  |  <a href='//omnidata.vision/designer/'>Dataset Designer</a> | <a href='//github.com/EPFL-VILAB/omnidata/tree/main/omnidata_annotator'>Annotation Generation</a> ] 

<br>

In addition to the above, we also provide a starter dataset, downloader and dataloader, pretrained models weights, all source code and a Docker. Links and explanations are below:

> **[Omnidata starter dataset](https://docs.omnidata.vision/starter_dataset.html):** comprised of 14 million viewpoint captures from over 2000 spaces with annotations for 21 different mid-level vision cues per image ([detailed statistics](https://docs.omnidata.vision/starter_dataset.html)). The dataset covers very diverse scenes (indoors and outdoors) and viewpoints (FoVs, scene- and object-centric). It builds on existing 3D datasets (Hypersim, Taskonomy, Replica, Google Scanned Objects, BlendedMVS, and some annotations are provided for CLEVR, too).

> **[Downloader tool](https://docs.omnidata.vision/starter_dataset_download.html):** parallelization tool to download specific combinations of sub-datasets/annotations. Python dataset/dataloader classes are available [here](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch). 

> **[Pretrained models](//github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch)** (depth, surface normals): try the models [online demo](//omnidata.vision/demo/) by uploading your own image--or download the [model weights](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch).

> **[Dataset designer demo](https://omnidata.vision/designer/):** Try a demo to explore the impact of different data generation parameters

> **[Annotator](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_annotator):** we provide all source code (requires Blender), and a standalone [Docker and tutorial](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_annotator).

---



<br>

#### Just let me clone the repo
Sure :)

```bash
git clone https://github.com/EPFL-VILAB/omnidata-tools
cd omnidata-tools
pip install -e .    # this will install the python requirements (and also install the CLI)
```

<br>


#### Just let me download the data
More info [here](https://docs.omnidata.vision/starter_dataset_download.html).
```bash
# Make sure everything is installed
sudo apt-get install aria2
pip install 'omnidata-tools' # Just to make sure it's installed

# Install the 'debug' subset of the Replica and Taskonomy components of the dataset
omnitools.download rgb normals point_info \
  --components replica taskonomy \
  --subset debug \
  --dest ./omnidata_starter_dataset/ --agree-all
```

<br>

<!-- <img src="https://raw.githubusercontent.com/alexsax/omnidata-tools/main/docs/images/omnidata_front_page.jpg?token=ABHLE3LC3U64F2QRVSOBSS3BPED24" alt="Website main page" style='max-width: 100%;'/> -->
