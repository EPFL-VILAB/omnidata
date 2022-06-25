<div align="center">

# Omnidata (Steerable Datasets)

**<strong>A scalable pipeline for generating multi-modal vision datasets from 3D meshes</strong>**

[`Main Website`](https://omnidata.vision) &centerdot; [`Paper`](https://arxiv.org/abs/2110.04994) &centerdot; [**`>> [GitHub] <<`**](//docs.omnidata.vision)  &centerdot; <it> [`Pretrained (online demo)`]('//omnidata.vision/demo/) &centerdot;  [`Data Generation (online demo)`](https://github.com/EPFL-VILAB/omnidata-tools/tree/main/omnidata_tools/torch) &centerdot; [`Annotator Demo (docker)`](//github.com/EPFL-VILAB/omnidata/tree/main/omnidata_annotator) </it>

_Ainaz Eftekhar*, Alexander Sax*, Roman Bachmann, Jitendra Malik, Amir Zamir_
 
</div>

---

In addition to the presentation content above, we also provide a starter dataset, downloader and dataloader, pretrained models weights, all source code and a Docker. Links and explanations below:

> **[Omnidata starter dataset](https://docs.omnidata.vision/starter_dataset.html):** comprised of 14 million viewpoint captures from over 2000 spaces with annotations for 21 different mid-level vision cues per image ([detailed statistics](https://docs.omnidata.vision/starter_dataset.html)). The dataset covers very diverse scenes (indoors and outdoors) and viewpoints (FoVs, scene- and object-centric). It builds on existing 3D datasets (Hypersim, Taskonomy, Replica, Google Scanned Objects, BlendedMVS, and some annotations are provided for CLEVR, too).

> **[Downloader tool](https://docs.omnidata.vision/starter_dataset_download.html):** parallelization tool to download specific combinations of sub-datasets/annotations. Python dataset/dataloader classes are available [here](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch). 

> **[Pretrained models](//github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch)** (depth, surface normals): try the models [online demo](//omnidata.vision/demo/) by uploading your own image--or download the [model weights](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch).

> **[Annotator](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_annotator):** we provide all [source code](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_annotator) (requires Blender), and a standalone [Docker and tutorial](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_annotator). Explore the impact of different data generation parameters with the [online dataset designer demo](https://omnidata.vision/designer/).

<br>


---

<br>

## FAQ

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

#### How did you do [X] in the paper?
A code dump for the training and experiments, exactly as used in the paper, is [here](https://github.com/EPFL-VILAB/omnidata/tree/main/paper_code). This code is for reference--do not expect this code to run on your machine!

<br>

#### Citation
```
@inproceedings{eftekhar2021omnidata,
  title={Omnidata: A Scalable Pipeline for Making Multi-Task Mid-Level Vision Datasets From 3D Scans},
  author={Eftekhar, Ainaz and Sax, Alexander and Malik, Jitendra and Zamir, Amir},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10786--10796},
  year={2021}
}
```
<!-- <img src="https://raw.githubusercontent.com/alexsax/omnidata-tools/main/docs/images/omnidata_front_page.jpg?token=ABHLE3LC3U64F2QRVSOBSS3BPED24" alt="Website main page" style='max-width: 100%;'/> -->
