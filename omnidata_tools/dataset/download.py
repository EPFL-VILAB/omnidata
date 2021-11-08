import aria2p
import multiprocess as mp
from functools import partial
import atexit, os, signal, shutil, tempfile, time
from   subprocess import call, run, Popen, check_output
import tarfile 
from   typing import Dict, Optional, Iterable
from   argparse import SUPPRESS

from .metadata import RemoteStorageMetadata, ZippedModel, bcolors, notice, header, license, failure
from .starter_dataset import STARTER_DATASET_REMOTE_SERVER_METADATAS, STARTER_DATA_COMPONENT_TO_SPLIT, STARTER_DATA_COMPONENT_TO_SUBSET, STARTER_DATA_COMPONENTS, STARTER_DATA_LICENSES
from fastcore.script import *
import tqdm

__all__ = ['main']
FORBIDDEN_COMPONENTS = set('habitat2')

### Information
def log_parameters(metadata_list, domains, subset, split, components, dest, dest_compressed, ignore_checksum, **kwargs):
  header('-------------------------------------')
  checksum = f'{bcolors.WARNING}False{bcolors.ENDC}' if ignore_checksum else f'{bcolors.OKGREEN}True{bcolors.ENDC}'
  header(f'From {bcolors.OKGREEN}SERVERS{bcolors.ENDC}: (using checksum validation: {checksum})')
  for rsm in metadata_list: header(f'    {bcolors.UNDERLINE}{rsm.link_file}{bcolors.ENDC}')
  header('')
  header(f'Data {bcolors.OKGREEN}parameters{bcolors.ENDC}: (what to download)') 
  header(f'    {bcolors.WARNING}Domains{bcolors.ENDC}    = {domains}') 
  header(f'    {bcolors.WARNING}Components{bcolors.ENDC} = {components}') 
  header(f'    {bcolors.WARNING}Subset{bcolors.ENDC}     = {subset}') 
  header(f'    {bcolors.WARNING}Split{bcolors.ENDC}      = {split}') 
  header('')
  header(f'Data {bcolors.OKGREEN}locations{bcolors.ENDC}:') 
  header(f'    {bcolors.WARNING}Dataset (extracted){bcolors.ENDC}      = {dest}') 
  header(f'    {bcolors.WARNING}Compressed files   {bcolors.ENDC}      = {dest_compressed}') 
  header('-------------------------------------\n\n')
  # print(f'[{bcolors.OKGREEN}FETCHING{bcolors.ENDC}] metadata from:')

def end_notes(**kwargs):
  notice(f'[{bcolors.OKGREEN + bcolors.BOLD}Download complete{bcolors.ENDC}]')
  notice(f'    Number of model files downloaded={len(kwargs["models"])}')
  notice('Recap:')
  log_parameters(**kwargs)

##

### Pre-download validation
def licenses_clickthrough(components, require_prompt, component_to_license):
  components = set(components + ['omnidata']) # Make sure everyone accepts omnidata terms
  license('Before continuing the download, please review the terms of use for each of the following component datasets:')
  for component in components:
    license(f"    {bcolors.WARNING}{component}{bcolors.ENDC}: \x1B]8;;{component_to_license[component]}\x1B\\{component_to_license[component]}\x1B]8;;\x1B\\")
  if not require_prompt: notice("Confirmation supplied by option '--agree_all'\n"); return
  else: 
    while True: 
      res = input("By entering 'y', I confirm that I have read and accept the above linked terms and conditions [y/n]: ").lower()
      if res == 'y': break
      elif res == 'n': print(f'[{bcolors.FAIL + bcolors.BOLD}EXIT{bcolors.ENDC}] Agreement declined: cancelling download.'); exit(0)
  notice("Agreement accepted. Continuing download.\n")
  return

def validate_checksums_exist(models):
  models_without_checksum = [m for m in models if m.checksum is None]
  if len(models_without_checksum) > 0: 
    show_k = 100
    notice(f'Found {len(models_without_checksum)} models without checksums:')
    for m in models_without_checksum[:show_k]: print(f'    {m.url}')
    if len(models_without_checksum) > show_k:  print(f'    and {len(models_without_checksum) - show_k} more...')
    print(f'Since "--ignore_checksum=False", cannot continue. Aborting.')
    exit(1)

def filter_models(models, domains, subset, split, components, component_to_split, component_to_subset):
  # for m in models:
  #   if m.component_name.lower() == 'replica':
  #     notice(f'{m.component_name} [{m.domain}]')
  #     print(components, domains)
  #     print(m.component_name.lower() in components)
  #     print(subset == 'all' or component_to_subset[m.component_name.lower()] is None or m.model_name in component_to_subset[m.component_name.lower()][subset]) 
  #     print(split == 'all' or component_to_split[m.component_name.lower()] is None or m.model_name in component_to_split[m.component_name.lower()])
  #     print('all' in domains or m.domain in domains)
  #     print("\n")
  # notice(f"Servers contain {len(models)} models")
  # notice(f'{set([m.component_name for m in models])}')
  filtered = [m for m in models 
    if (m.component_name.lower() in components)
    and (subset == 'all' or component_to_subset[m.component_name.lower()] is None or m.model_name in component_to_subset[m.component_name.lower()][subset]) 
    and (split == 'all' or component_to_split[m.component_name.lower()] is None or m.model_name in component_to_split[m.component_name.lower()])
    and ('all' in domains or m.domain in domains)
    ]
  # notice(f"Filtered down to {len(filtered)} models based on specified criteria.")
  return filtered
## 

### Downloading
def ensure_aria2_server(aria2_create_server, aria2_uri, aria2_secret, connections_total, connections_per_server_per_download, aria2_cmdline_opts, **kwargs):
  if not aria2_uri or not aria2_create_server: return None
  a2host, a2port = ":".join(aria2_uri.split(':')[:-1]), aria2_uri.split(':')[-1]
  notice(f"Opening aria2c download daemon in background: {bcolors.WARNING}Run {bcolors.OKCYAN}'aria2p'{bcolors.WARNING} in another window{bcolors.ENDC} to view status.") 
  n = connections_total 
  x = connections_per_server_per_download if connections_per_server_per_download is not None else connections_total
  x = min(x, 16)
  a2server = Popen(('aria2c --enable-rpc --rpc-listen-all --disable-ipv6 -c --auto-file-renaming=false ' +
                    # '--optimize-concurrent-downloads ' + 
                    f'-s{n}  -j{n}  -x{x} {aria2_cmdline_opts}').split())
  atexit.register(os.kill, a2server.pid, signal.SIGINT)
  return aria2p.API(aria2p.Client(host=a2host, port=a2port, secret=aria2_secret))


def download_tar(url, output_dir='.', output_name=None, n=20, n_per_server=10,
  checksum=None, max_tries_per_model=3, aria2api=None, dryrun=False,
  ) -> Optional[str]:
  '''Downloads "url" to output filename. Returns downloaded fpath.'''
  fname = url.split('/')[-1] if output_name is None else output_name
  fpath = os.path.join(output_dir, fname)
  if dryrun: print(f'Downloading "{url}"" to "{fpath}"'); return fpath
  # checksum = checksum[:-3] + '000'
  # print(checksum)
  if aria2api is not None:
    options_dict = { 'out': fname, 'dir': output_dir, 'check_integrity': True}
    if checksum is not None: options_dict['checksum'] = f"md5={checksum}"
    while (max_tries_per_model := max_tries_per_model-1) > 0:
      res = aria2api.client.add_uri(uris=[url], options=options_dict)
      success = wait_on(aria2api, res)
      if success: break
    if not success: return None
  else:
    # os.makedirs(output_dir, exist_ok=True)
    # cmd = f'lftp -e "pget -n {n} {url} -o {fpath}"'
    # # print(cmd)
    # call(cmd, shell=True) 
    options = f'-c --auto-file-renaming=false'
    if n_per_server is None: n_per_server = min(n, 16)
    options += f' -s {n} -j {n} -x {n_per_server}' # N connections
    if checksum is not None: options += f' --check-integrity=true --checksum=md5={checksum}'
    cmd = f'aria2c -k 1M -d {output_dir} -o {fname} {options} "{url}"'
    # print(cmd)
    call(cmd, shell=True) 

    # os.makedirs(output_dir, exist_ok=True)
    # cmd = f'axel -q -o {fpath} -c -n {n} "{url}" '
    # success = True
    # while (max_tries_per_model := max_tries_per_model-1) > 0:
    #   call(cmd, shell=True)
    #   if checksum is not None: success = (check_output(['md5sum', fpath], encoding='UTF-8').split()[0] == checksum)
    # if not success: return None
  return fpath

def wait_on(a2api, gid, duration=0.2):
  while not (a2api.get_downloads([gid])[0].is_complete or a2api.get_downloads([gid])[0].has_failed):
    time.sleep(duration)
  success = a2api.get_downloads([gid])[0].is_complete 
  a2api.remove(a2api.get_downloads([gid]))
  return success
##


### Untarring
def untar(fpath, model, dest=None, ignore_existing=True,
    output_structure=('domain', 'component_name', 'model_name'), # Desired directory structure
    dryrun=False
  ) -> None:
  dest_fpath = os.path.join(dest, *[getattr(model, a) for a in output_structure])
  if dest is not None: os.makedirs(dest, exist_ok=True)
  if os.path.exists(dest_fpath) and ignore_existing: notice(f'"{dest_fpath}" already exists... skipping'); return
  with tempfile.TemporaryDirectory(dir=dest) as tmpdirname:
    src_fpath = os.path.join(tmpdirname, *[getattr(model, a) for a in model.tar_structure])
    if dryrun: print(f'Extracting "{fpath}"" to "{tmpdirname}" and moving "{src_fpath}" to "{dest_fpath}"'); return
    with tarfile.open(fpath) as tar:
      tar.extractall(path=tmpdirname)
    shutil.move(src_fpath, dest_fpath)

##

@call_parse
def download(
  domains:     Param("Domains to download (comma-separated or 'all')", str, nargs='+'),
  subset:      Param("Subset to download", str, choices=['debug', 'tiny', 'medium', 'full', 'fullplus'])='debug',
  split:       Param("Split to download", str, choices=['train', 'val', 'test', 'all'])='all',
  components:  Param("Component datasets to download (comma-separated)", str, nargs='+',
    choices=['all','replica','taskonomy','gso_in_replica','hypersim','blendedmvs','hm3d','clevr_simple','clevr_complex'])='all',
  dest:             Param("Where to put the uncompressed data", str)='uncompressed/',
  dest_compressed:  Param("Where to download the compressed data", str)='compressed/',
  keep_compressed:  Param("Don't delete compressed files after decompression", bool_arg)=False,
  only_download:    Param("Only download compressed data", bool_arg)=False,
  max_tries_per_model:    Param("Number of times to try to download model if checksum fails.", int)=3,  
  connections_total:      Param("Number of simultaneous aria2c connections overall (note: if not using the RPC server, this is per-worker)", int)=32,
  connections_per_server_per_download: Param("Number of simulatneous aria2c connections per server per download. Defaults to 'total_connections' (note: if not using the RPC server, this is per-worker)", int)=None,
  n_workers:              Param("Number of workers to use", int)=min(mp.cpu_count(), 16),
  num_chunk:        Param("Download the kth slice of the overall dataset", int)=0,
  num_total_chunks: Param("Download the dataset in N total chunks. Use with '--num_chunk'", int)=1, 
  ignore_checksum:  Param("Ignore checksum validation", bool_arg)=True,
  dryrun:           Param("Keep compressed files even after decompressing", store_true)=False,
  aria2_uri:              Param("Location of aria2c RPC (if None, use CLI)", str)="http://localhost:6800", 
  aria2_cmdline_opts:     Param("Opts to pass to aria2c", str)='',  
  aria2_create_server:    Param("Create a RPC server at aria2_uri", bool_arg)=True, 
  aria2_secret:           Param("Secret for aria2c RPC", str)='',
  agree_all:      Param("Agree to all license clickwraps.", store_true)=False, 
  ):
  ''' 
    Downloads Omnidata starter dataset.
    ---
    The data is stored on the remote server in a compressed format (.tar.gz).
    This function downloads the compressed and decompresses it.

    Examples:
      download rgb normals point_info --components clevr_simple clevr_complex --connections_total 30
  '''
  # The following data could instead be supplied from the remote server:
  metadata_list = STARTER_DATASET_REMOTE_SERVER_METADATAS # Param("Metadata servers to search", Iterable[RemoteStorageMetadata])
  component_to_split = STARTER_DATA_COMPONENT_TO_SPLIT    # Param("Train/Val/Test splits for each component", Dict[str, Dict[str, Iterable[str]]])
  component_to_subset = STARTER_DATA_COMPONENT_TO_SUBSET  # Param("Debug/.../fullplus splits for each component", Dict[str, Iterable[str]])
  component_to_license = STARTER_DATA_LICENSES            # Param("Licenses for each component dataset", Dict[str, Iterable[str]])

  if components == 'all': components = list(component_to_license.keys())
  log_parameters(**locals())
  licenses_clickthrough(components, require_prompt=not agree_all, component_to_license=component_to_license)
  aria2 = ensure_aria2_server(**locals())

  # Determine which models to use
  models = [metadata.parse(url)
            for metadata in metadata_list 
            for url in metadata.links]
  models = filter_models(models, domains, subset, split, components, 
            component_to_split=component_to_split,
            component_to_subset=component_to_subset)
  notice(f'Found {len(models)} matching blobs on remote serverss.')
  models = models[num_chunk::num_total_chunks] # Parallelization: striped slice of models array
  if not ignore_checksum: validate_checksums_exist(models)


  # Process download

  def process_model(model):
    try:
      tar_fpath = download_tar(
                  model.url, output_dir=dest_compressed, output_name=model.fname, 
                  checksum=None if ignore_checksum else model.checksum,
                  n=connections_total, n_per_server=connections_per_server_per_download,
                  aria2api=aria2, dryrun=dryrun)
      if tar_fpath is None: return
      if only_download:     return
      untar(tar_fpath, dest=dest, model=model, ignore_existing=True, dryrun=dryrun)
      if not keep_compressed: os.remove(tar_fpath)
    except Exception as e:
      failure(f"Failure when processing model {model.url} (stacktrace below)")
      raise e

  if n_workers < 1 : 
    for model in tqdm.tqdm(models): process_model(model)
  else:
    with mp.Pool(n_workers) as p:
      r = list(tqdm.tqdm(p.imap(process_model, models), total=len(models)))
      # p.map(process_model, models)

  # Cleanup
  end_notes(**locals())


if __name__ == '__main__':
  a2server = Popen('aria2c --enable-rpc --rpc-listen-all --disable-ipv6 -c --auto-file-renaming=false -s 10 -x 10'.split())

  time.sleep(0.2)
  model = ZippedModel(
    component_name='taskonomy', domain='point_info', model_name='yscloskey', 
    url='https://datasets.epfl.ch/taskonomy/yscloskey_point_info.tar'
  )
  tar_format = ('domain',)
  dest_compressed = '/tmp/omnidata/compressed'
  dest = '/tmp/omnidata/uncompressed'

  tar_fpath = download_tar(model.url, output_dir=dest_compressed,
    checksum='md5=9f9752d74b07bcc164af4a6c61b0eca1',
    output_name=model.fname)
  untar(tar_fpath, dest=dest, model=model, tar_format=tar_format, ignore_existing=True)
  os.remove(tar_fpath)

  # Terminate the process
  os.kill(a2server.pid, signal.SIGINT)

