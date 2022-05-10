
import functools, requests
import multiprocessing as mp
from ..metadata import RemoteBucketStorageMetadata, ZippedModel


# SPLITS | Which spaces to download

from .component_datasets import taskonomy, hypersim, replica, replica_gso, blendedmvg

# --- Specifics ---
class OmnidataMetadata(RemoteBucketStorageMetadata):
  def __init__(self, base_url='https://datasets.epfl.ch/omnidata/', **kwargs):
    super().__init__(base_url=base_url, **kwargs)

  @functools.cached_property
  def links(self): return [k for k in requests.get(self.link_file).text.splitlines()
      if k.endswith(self.expected_suffix) and not any([d in k for d in ('depth_zbuffer2', 'mask_valid2')])]
  
  def parse(self, url) -> ZippedModel:
    '''Most omnidata files are stored like: https://datasets.epfl.ch/omnidata//omnidata_tars/depth_euclidean/blendedMVS/depth_euclidean-blendedMVS-000000000000000000000000.tar.gz'''
    split_urls = url.split('/') 
    if not ((url_len := len(split_urls)) == 8): raise ValueError(f'Expected url to be split into 8 components, not {url_len}: "{url}"')
    domain, component_name, fname = split_urls[5:8]
    domain2, component_name2, *model_name = fname[:-len(self.expected_suffix)].split('-')
    if not url.endswith(self.expected_suffix): raise ValueError(f'Expected compressed url to end with "{self.expected_suffix}": {url}')
    if not component_name == component_name2: raise ValueError(f'Expected domain from fname ("{component_name2}" in "{fname}") to match domain in url ("{component_name}" in "{url}").') 
    if not domain == domain2: raise ValueError(f'Expected domain from fname ("{domain2}" in "{fname}") to match domain in url ("{domain}" in "{url}").') 
    if len(model_name) == 0:  raise ValueError(f'Model name in {fname} has 0 length after domain + component.')  
    model_name = "-".join(model_name)
    return ZippedModel(component_name=component_name, domain=domain, model_name=model_name, url=url, tar_structure=self.tar_structure, checksum=self.checksum(url))


class TaskonomyMetadata(RemoteBucketStorageMetadata):
  def __init__(self, base_url='https://datasets.epfl.ch/taskonomy/', expected_suffix='.tar', tar_structure=('domain',), **kwargs):
    super().__init__(base_url=base_url, expected_suffix=expected_suffix, tar_structure=tar_structure, **kwargs)

  def parse(self, url) -> ZippedModel:
    '''Taskonony links are stored as: https://datasets.epfl.ch/taskonomy/adairsville_class_object.tar'''
    split_urls = url.split('/') 
    if not url.endswith(self.expected_suffix): raise ValueError(f'Expected compressed url to end with "{self.expected_suffix}": {url}')
    if not (url_len := len(split_urls)) == 5: raise ValueError(f'Expected url to be split into 5 components, not {url_len}: "{url}"')
    if not (component_name := split_urls[-2]) == 'taskonomy': raise ValueError(f'Expected component name to be "taskonomy", not "{component_name}"')
    fname = split_urls[-1][:-len(self.expected_suffix)]
    model_name, *domain = fname.split('_')
    if len(domain) == 0:  raise ValueError(f'Domain name in {fname} has 0 length after model.')  
    domain = "_".join(domain)
    tar_structure = ('domain', 'model_name') if domain == 'fragments' else self.tar_structure
    return ZippedModel(component_name=component_name, domain=domain, model_name=model_name, url=url, tar_structure=tar_structure, checksum=self.checksum(url))

STARTER_DATASET_REMOTE_SERVER_METADATAS = [
  OmnidataMetadata( base_url='https://datasets.epfl.ch/omnidata/', expected_suffix='.tar'),
  TaskonomyMetadata( base_url='https://datasets.epfl.ch/taskonomy/'),
]


STARTER_DATA_COMPONENTS = {
  'taskonomy': taskonomy,
  'hypersim': hypersim,
  'replica': replica,
  'replica_gso': replica_gso,
  'blendedmvg': blendedmvg,
  'hm3d': None,
  'clevr_simple': None,
  'clevr_complex': None
}

STARTER_DATA_COMPONENT_TO_SPLIT = {k: getattr(v, 'split_to_spaces', None) for k, v in STARTER_DATA_COMPONENTS.items()}
STARTER_DATA_COMPONENT_TO_SUBSET = {k: getattr(v, 'subset_to_spaces', None) for k, v in STARTER_DATA_COMPONENTS.items()}


STARTER_DATA_LICENSES = {
  'omnidata':       'https://raw.githubusercontent.com/EPFL-VILAB/omnidata-tools/main/LICENSE',
  'replica':        'https://raw.githubusercontent.com/facebookresearch/Replica-Dataset/main/LICENSE',
  'hypersim':       'https://raw.githubusercontent.com/apple/ml-hypersim/master/LICENSE.txt',
  'replica_gso': 'https://creativecommons.org/licenses/by/4.0/legalcode',
  'clevr_simple':   'https://creativecommons.org/licenses/by/4.0/legalcode',
  'clevr_complex':  'https://creativecommons.org/licenses/by/4.0/legalcode',
  'blendedmvg':     'https://creativecommons.org/licenses/by/4.0/legalcode',
  'taskonomy':      'https://raw.githubusercontent.com/StanfordVL/taskonomy/master/data/LICENSE',
  'hm3d':           'https://matterport.com/matterport-end-user-license-agreement-academic-use-model-data'
}

if __name__ == '__main__':
  print(STARTER_DATA_COMPONENT_TO_SUBSET['replica'])
  [print(rsm.info) for rsm in STARTER_DATASET_REMOTE_SERVER_METADATAS]
