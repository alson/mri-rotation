from sklearn.model_selection import train_test_split
from pathlib import PosixPath
import argparse

parser = argparse.ArgumentParser(description='Split all images in train and test set (strasfied by source image name)')
parser.add_argument('source_folder')
parser.add_argument('train_folder')
parser.add_argument('test_folder')
parser.add_argument('test_fraction', type=float, default=0.2, nargs='?')
args = parser.parse_args()

if args.test_fraction < 0.0 or args.test_fraction >= 1.0:
    raise ValueError

src_dir = PosixPath(args.source_folder)
train_dir = PosixPath(args.train_folder)
test_dir = PosixPath(args.test_folder)

base_files = [fn.name for fn in src_dir.glob('000/*.npy')]

train_files, test_files = train_test_split(base_files, test_size=args.test_fraction)

for base_name in train_files:
    for fn in src_dir.glob(f'[0-9][0-9][0-9]/{base_name}'):
        dest_dir = train_dir / fn.parts[-2]
        dest_dir.mkdir(parents=True, exist_ok=True)
        fn.rename(dest_dir / fn.name)

for base_name in test_files:
    for fn in src_dir.glob(f'[0-9][0-9][0-9]/{base_name}'):
        dest_dir = test_dir / fn.parts[-2]
        dest_dir.mkdir(parents=True, exist_ok=True)
        fn.rename(dest_dir / fn.name)
