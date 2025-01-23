import os
import shutil
import argparse
import json
import urllib.request
import hashlib
from typing import Union

from bdpy.dataset.utils import download_file
from tqdm import tqdm


def main(cfg):
    with open(cfg.filelist, 'r') as f:
        filelist = json.load(f)

    target = filelist[cfg.target]

    for fl in target['files']:
        output = os.path.join(target['save_in'], fl['name'])
        os.makedirs(target['save_in'], exist_ok=True)

        # Downloading
        if not os.path.exists(output):
            if isinstance(fl['url'], str):
                print(f'Downloading {output} from {fl["url"]}')
                download_file(fl['url'], output, progress_bar=True, md5sum=fl['md5sum'])
            else:
                # fl['url'] and fl['md5sum'] are lists
                for url, md5 in zip(fl['url'], fl['md5sum']):
                    print(f'Downloading {output} from {url}')
                    download_file(url, output, progress_bar=True, md5sum=md5)

        # Postprocessing
        if 'postproc' in fl:
            for pp in fl['postproc']:
                if pp['name'] == 'merge':
                    pass
                elif pp['name'] == 'unzip':
                    print(f'Unzipping {output}')
                    if 'destination' in pp:
                        dest = pp['destination']
                    else:
                        dest = './'
                    shutil.unpack_archive(output, extract_dir=dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='files.json')
    parser.add_argument('target')

    cfg = parser.parse_args()

    main(cfg)
