import sys
import tarfile
import os


def create_checkpoint_tar(save_path):
    """
    Create a tar archive of all checkpoint files for a particular checkpoint

    :param save_path: Path to the checkpoints
    :return: Path to the created tar file
    """
    checkpoint_suffixes = [".data-00000-of-00001", ".index", ".meta"]

    tar_path = save_path + ".tar"

    with tarfile.open(tar_path, 'w') as tar:
        for suffix in checkpoint_suffixes:
            checkpoint_file = save_path+suffix
            tar.add(checkpoint_file, arcname=os.path.basename(checkpoint_file))

    return tar_path

def untar(tar_path,extract_path):
    """
    Extract tar file to a destination

    :param tar_path: Path to the tar file
    :param extract_path: Path to extract the tar
    :return: Nothing
    """
    with tarfile.open(tar_path) as tar_file:
        tar_file.extractall(path=extract_path)

def rprint(string):
    """
    Print a string in one line by overwriting the current line.
    Used to print training and testing progress.

    :param string: String to be printed
    :return: Nothing
    """
    sys.stdout.write("\r{}".format(string))
    sys.stdout.flush()

