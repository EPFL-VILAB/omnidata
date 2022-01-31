from collections import namedtuple
import functools
from fastcore.basics import store_attr
import requests
from   typing import Any, Optional, Dict, List, Iterable, Tuple, Union, Callable



class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'

def notice(msg): print(f'[{bcolors.OKGREEN + bcolors.BOLD}NOTICE{bcolors.ENDC}] {msg}')
def header(msg): print(f'[{bcolors.HEADER + bcolors.BOLD}HEADER{bcolors.ENDC}] {msg}')
def license(msg): print(f'[{bcolors.WARNING + bcolors.BOLD}LICENSE{bcolors.ENDC}] {msg}')
def underline(msg): print(f'{bcolors.UNDERLINE}{msg}{bcolors.ENDC}')
def failure(msg): print(f'[{bcolors.FAIL + bcolors.BOLD}FAILURE{bcolors.ENDC}] {msg}')

def print_and_log_failure(msg, error_list):
    failure(msg)
    error_list.append(msg)

class ZippedModel:
  def __init__(self, component_name, domain, model_name, url, tar_structure, checksum=None):
    self.component_name = component_name
    self.domain         = domain.lower()
    self.model_name     = model_name
    self.url            = url
    self.ext = ".".join(self.url.split('/')[-1].split('.')[1:])
    self.fname = f'{domain}__{component_name}__{model_name}.{self.ext}'
    self.checksum = checksum
    self.tar_structure = tar_structure

class RemoteStorageMetadata:
  ''' Contains/gets metadata about what/where data is stored on some remote endpoint. '''
  def __init__(self, 
      link_file, checksum_file=None, expected_suffix='.tar', 
      tar_structure=('domain', 'component_name', 'model_name')
    ) -> None: store_attr()
  
  def parse(self, url) -> ZippedModel: raise NotImplementedError
  def checksum(self, url) -> str: return self.checksums[url]
  @functools.cached_property  
  def links(self)      -> List[str]:  return [k for k in requests.get(self.link_file).text.splitlines() if k.endswith(self.expected_suffix)]
  @functools.cached_property  
  def checksums(self)    -> Dict[str, str]:  return {k.split()[1]: k.split()[0] for k in requests.get(self.checksum_file).text.splitlines() if k.endswith(self.expected_suffix)}
  @functools.cached_property  
  def models(self)     -> List[ZippedModel]:  return [self.parse(url) for url in self.links]
  @functools.cached_property  
  def domains(self)    -> List[str]:  return set([m.domain for m in self.models])
  @functools.cached_property  
  def components(self) -> List[str]:  return set([m.component_name for m in self.models])
  @functools.cached_property 
  def info(self)       -> str:  return join_recursive('\n', [
      f'Data location: {self.link_file}',
      f'    Links: ({len(self.links)})',
      f'    Domains: ({len(self.domains)})',
      [f'      {d}' for d in self.domains],
      f'    Components: ({len(self.components)})',
      [f'      {c}' for c in self.components]
    ])

class RemoteBucketStorageMetadata(RemoteStorageMetadata):
  '''
    Remote storage, but assumes there are two files at:
      BASE_URL/links.txt
      BASE_URL/md5sum.txt
    
    - links.txt should describe the full url download link for each model
    - md5sum.txt should be two-column with a row for each model: first col is md5sum 
        and second col is the part of the URL specific to the model (e.g. everything after *.edu/)
  '''
  def __init__(self, base_url, expected_suffix='.tar.gz', tar_structure=('domain', 'component_name', 'model_name')) -> None:
    self.base_url = base_url
    super().__init__(
      link_file=f'{base_url}/links.txt', checksum_file=f'{base_url}/md5sum.txt',
      expected_suffix=expected_suffix, tar_structure=tar_structure
    )

  def checksum(self, url): return self.checksums.get(url.replace(self.base_url,''))

def join_recursive(s, args):
  return s.join([x if isinstance(x, str) else join_recursive(s, x) for x in args])
