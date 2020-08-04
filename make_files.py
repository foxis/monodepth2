import argparse
import os
from PIL import Image

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

            for a, b in zip(images[:-1], images[1:]):
                if int(b[:-4]) - int(a[:-4]) > 1:
                    print(sequence, "missing", a, b)

            valid_images = []
            for img in images:
                try:
                    with open(os.path.join(path, sequence, img), 'rb') as _f:
                        with Image.open(_f) as _img:
                            _img.convert('RGB')

                    valid_images += [img]
                except Exception as err:
                    print(sequence, img, err)

            print(f"Sequence {sequence} images: {len(images)}, valid: {len(valid_images)}")

            f.writelines([f"{sequence} {img[:-4]} 0\n" for img in valid_images[1:-1]])


if __name__ == '__main__':
    args = parse_args()
    make_files(args)
