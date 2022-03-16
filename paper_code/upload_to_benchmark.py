import os
import sys
import glob
import uuid
import time
import argparse
import random
import subprocess
from tqdm import tqdm

VALID_BENCHMARKS = ["normal_bench", "depth_bench", "occfold_bench", "planar_bench"]

CURL_COMMAND_TEMPLATE = 'curl ' \
'-F "password=USER_PASSWORD" ' \
'-F "email=EMAIL_ADDR" ' \
'-F "benchmark=BENCHMARK_NAME" ' \
'-F "authors=AUTHORS" ' \
'-F "sub_id=SUBMISSION_ID" ' \
'-F "final=IS_FINAL" ' \
'-F "part=TAR_PART" ' \
'-F "b_public=MAKE_PUBLIC" ' \
'-F "publication=PUBLICATION_TITLE" ' \
'-F "url_publication=PUBLICATION_URL" ' \
'-F "sub_name=NAME_OF_SUBMISSION" ' \
'-F "affiliation=AFFILIATION_NAME" ' \
'-F "data=@FILE_UPLOAD_PATH" https://oasis.cs.princeton.edu/submit2'

def upload_files_to_server(temp_directory, args):
    FORM_DICT = {
        'BENCHMARK_NAME': args.task,
        'USER_PASSWORD': args.password,
        'MAKE_PUBLIC': 'Yes' if args.public else 'No',
        'SUBMISSION_ID': str(uuid.uuid4())[:8],
        'EMAIL_ADDR': args.email.replace('@','#AT#'),
        'NAME_OF_SUBMISSION': args.submission_name,
        'AFFILIATION_NAME': args.affiliation,
        'PUBLICATION_TITLE': args.publication_title,
        'PUBLICATION_URL': args.publication_url,
        'AUTHORS': args.authors
    }
    curl_command = CURL_COMMAND_TEMPLATE
    for k,v in FORM_DICT.items():
        curl_command = curl_command.replace(k,v)
    assert len(curl_command.split('@')) == 2 and '$' not in curl_command
    gz_files = glob.glob(os.path.join(temp_directory, '*'))
    upload_iter = tqdm(gz_files)
    upload_iter.set_description(f"Uploading {tmp_dir} to evaluation server (may take a while)")
    for i,gz_file in enumerate(upload_iter):
        assert os.path.exists(gz_file)
        part = gz_file.split('.')[-1]
        cmd = curl_command.replace('FILE_UPLOAD_PATH', gz_file).replace('TAR_PART', part).replace("IS_FINAL", str(i == len(gz_files)-1))
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
        output = output.decode('utf-8')
        assert "Error" not in output, output

def create_tar_chunks(source_directory, temp_directory):
    command1 = f"tar czvf - {source_directory}"
    command2 = f"split --bytes=1000MB - {temp_directory}/{source_directory.split('/')[-1]}.tar.gz."
    ps = subprocess.Popen(command1.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ps2 = subprocess.Popen(command2.split(), stdin=ps.stdout,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    num_files = len(glob.glob(os.path.join(source_directory, '*')))
    upload_iter = tqdm(iter(ps.stderr.readline, ""), total=num_files, disable=False)
    upload_iter.set_description(f"Zipping {source_directory} into {temp_directory}")
    for i, stdout_line in enumerate(upload_iter):
        if ps.poll() is not None:
            break
        # print(i, stdout_line, num_files)
    if ps.poll() != 0:
        raise Exception(f"Command \"{command1} | {command2}\" failed with exit code {ps.poll()}")

if __name__ == "__main__":

    # Read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('submission_directory', help='The directory containing .npy files to tar and submit.')
    parser.add_argument('--task', required=True, help='one of ' + str(VALID_BENCHMARKS) + '.')
    parser.add_argument('--affiliation', required=True, help='Your Affiliation (will not be publicly displayed).')
    parser.add_argument('--publication_title', default="", help='Publication Title.')
    parser.add_argument('--publication_url', default="", help='Link to Publication.')
    parser.add_argument('--authors', default="", help='Authors.')
    parser.add_argument('--submission_name', required=True, help='Submission Name (The name that will appear on the leaderboard).')
    parser.add_argument('--email', required=True, help='Email account entered when receiving a password for OASIS.')
    parser.add_argument('--password', required=True, help='OASIS account password. Requested via the OASIS login page. Valid for four hours.')
    parser.add_argument('--public', action="store_true", help='Make the submission public.')
    parser.add_argument('--temp_directory', type=str, default=None, help='The local path to a temporary directory. If not provided, a directory oasis_upload_tmp/ will be created instead.')
    parser.add_argument('--skip_taring', action="store_true", default=False, help='Assume the submission is already tarred into the temporary directory.')
    args = parser.parse_args()
    args.submission_directory = args.submission_directory.rstrip('/')

    # Verify correct directory structure
    assert os.path.exists(args.submission_directory)
    assert os.path.isdir(args.submission_directory)
    assert all(f.endswith('.npy') for f in glob.iglob(os.path.join(args.submission_directory, '*')))
    assert args.task in VALID_BENCHMARKS, f"--task must belong to {VALID_BENCHMARKS}"
    assert '@' in args.email and '#AT#' not in args.email
    assert '@' not in args.publication_url

    # Create temporary directory
    if args.temp_directory is None:
        tmp_dir = 'oasis_upload_tmp'
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
            print(f"INFO: No temporary directory was specified using --temp_directory. Creating a directory {tmp_dir}/")
        elif not os.path.isdir(tmp_dir):
            raise Exception(f"{tmp_dir} already exists but isn't a directory. Please remove/rename this file.")
        else:
            print(f"INFO: No temporary directory was specified using --temp_directory. Using directory {tmp_dir}/") 
    else:
        tmp_dir = args.temp_directory.rstrip('/')
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        print(f"INFO: Using specified temp directory '{tmp_dir}/'") 


    # Zip folder into 1GB chunks
    if not args.skip_taring:
        create_tar_chunks(args.submission_directory, tmp_dir)

    # Run CURL commands sequentially
    upload_files_to_server(tmp_dir, args)