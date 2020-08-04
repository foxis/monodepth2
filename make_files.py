import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Builds test files given a roverc dataset.')

    parser.add_argument('--dataset_path', type=str,
                        help='path to a dataset of images', required=True)
    parser.add_argument('--files_path', type=str,
                        help='path to a files.txt file', required=True)

    return parser.parse_args()


def make_files(args):
    path = args.dataset_path
    files_path = args.files_path

    sequences = os.listdir(path)
    sequences.sort()
    with open(files_path, "w") as f:
        for sequence in sequences:
            images = os.listdir(os.path.join(path, sequence))
            f.writelines([f"{sequence} {img[:-4]} 0\n" for img in images])


if __name__ == '__main__':
    args = parse_args()
    make_files(args)
