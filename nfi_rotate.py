from math import sqrt, ceil
import argparse
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import rotate
from skimage.transform import resize

OUT_PATH = Path('data')

parser = argparse.ArgumentParser(description='Rotate middle slice and export')
parser.add_argument('filename')
args = parser.parse_args()

img = nib.load(args.filename)
filename_path = Path(args.filename)
basename = filename_path.stem
suffix = filename_path.suffix

data = img.get_fdata()
data_by_slice = data.swapaxes(2, 0).swapaxes(1, 2)
mid_slice = data_by_slice[len(data_by_slice) // 2]
big_square_size = ceil(sqrt(mid_slice.shape[0]**2 + mid_slice.shape[1]**2))
big_square = np.full((big_square_size, big_square_size), mid_slice[0][0])
x_in_square, y_in_square = ((big_square_size - np.array(mid_slice.shape)) // 2) - 1
big_square[x_in_square:x_in_square+mid_slice.shape[0], y_in_square:y_in_square+mid_slice.shape[1]] = mid_slice
square100x100 = resize(big_square, [100, 100], order=3)

for angle in range(360):
    rotated_slice = rotate(square100x100, angle, reshape=False)
    square100x100 = resize(rotated_slice, [100, 100], order=3)
    folder = OUT_PATH / f'{angle:03}'
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f'{basename}.npy', square100x100.astype(np.float16))
