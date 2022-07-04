# Omnidata Docs
> <strong>Quick links to docs</strong>: [ <a href='/omnidata-tools/pretrained.html'>Pretrained Models</a> ]  [ <a href='/omnidata-tools/starter_dataset.html'>Starter Dataset ]  [ <a href='//omnidata-tools/annotator_usage.html'>Annotator Demo</a> ] 



**This site is intended to be a wiki/documentation site for everything that we open-sourced from the paper.** There are three main folders: the annotator, utilities (dataloaders, download tools, pretrained models, etc), and a code dump of stuff from the paper that is just for reference. 

(Check out the main site for an overview of 'steerable datastes' and the 3D → 2D rendering pipeline).



<br>

#### Download the code
If you want to see and edit the code, then you can clone the github and install with: 

```bash
git clone https://github.com/EPFL-VILAB/omnidata-tools
cd omnidata-tools
pip install -e .    # this will install the python requirements (and also install the CLI)
```
This is probably the best option for you if you want to use the pretrained models, dataloaders, etc in other work.

<br>


#### Install just CLI tools (`omnitools`)
If you are only interested in using the [CLI tools](/omnidata-tools/omnitools.html), you can install them with: `pip install omnidata-tools`. This might be preferable if you only want to quickly download the starter data, or if you just want a simple way to manipulate the vision datasets output by the annotator.

_Note:_ The annotator can also be used with a [docker-based](/omnidata-tools/annotator_usage.html) CLI, but you don't need to use the annotator to use the starter dataset, pretrained models, or training code.


<br>


> ...were you looking for the [research paper](//omnidata.vision/#paper) or [project website](//omnidata.vision)? 

<!-- <img src="https://raw.githubusercontent.com/alexsax/omnidata-tools/main/docs/images/omnidata_front_page.jpg?token=ABHLE3LC3U64F2QRVSOBSS3BPED24" alt="Website main page" style='max-width: 100%;'/> -->
