import os
import zipfile

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    os.system('wget {} -O {}'.format(url, filepath))
    return filepath

def ungzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    file_zip = zipfile.ZipFile(filepath, 'r')
    for file in file_zip.namelist():
        file_zip.extract(file, dirpath)
    file_zip.close()

def download_xnli(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    url = 'https://dl.fbaipublicfiles.com/XNLI/XNLI-MT-1.0.zip'
    ungzip(download(url, dirpath))

def download_xnli_mt(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    url = 'https://dl.fbaipublicfiles.com/XNLI/XNLI-15way.zip'
    ungzip(download(url, dirpath))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    xnli_dir = os.path.join(base_dir, 'data')
    mt_dir = os.path.join(base_dir, 'data')
    download_xnli(xnli_dir)
    download_xnli_mt(mt_dir)
