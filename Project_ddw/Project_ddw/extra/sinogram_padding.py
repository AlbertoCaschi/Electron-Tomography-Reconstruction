
file_path = "C:\\Users\\Alberto\\Desktop\\Electron tomography\\datasets\\circle\\sinograms\\sinogram_-50_5_50.mrc"

import numpy as np
import mrcfile

with mrcfile.open(file_path) as mrc:
    raw_sinogram = np.squeeze(mrc.data)

angles = np.arange(-40, 45, 5)
full_sinogram = np.zeros((362, 180)) 

for i, angle in enumerate(angles):
    column_index = int(angle + 90)
    full_sinogram[:, column_index] = raw_sinogram[:, i]

full_sinogram_float = full_sinogram.astype(np.float32)

with mrcfile.new('C:\\Users\\Alberto\\Desktop\\Electron tomography\\My_work\\padded_sinogram.mrc', overwrite=True) as new_mrc:
    new_mrc.set_data(full_sinogram_float)

print("MRC file saved as padded_sinogram.mrc")