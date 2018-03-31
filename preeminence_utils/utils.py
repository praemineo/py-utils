import sys
import tarfile
import os

def save_weights():
    print "Called new save weights"


def create_checkpoint_tar(save_path):
    checkpoint_suffixes = [".data-00000-of-00001", ".index", ".meta"]

    tar_path = save_path + ".tar"

    with tarfile.open(tar_path, 'w') as tar:
        for suffix in checkpoint_suffixes:
            checkpoint_file = save_path+suffix
            tar.add(checkpoint_file, arcname=os.path.basename(checkpoint_file))

    return tar_path

def untar(tar_path,extract_path):
    with tarfile.open(tar_path) as tar_file:
        tar_file.extractall(path=extract_path)

def rprint(string):
    sys.stdout.write("\r{}".format(string))
    sys.stdout.flush()

