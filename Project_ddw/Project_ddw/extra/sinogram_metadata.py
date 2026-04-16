
file_path = "C:\\Users\\Alberto\\Desktop\\Electron tomography\\DeepDeWedge\\tutorial\\tutorial_data\\tutorial_data\\tomo_all_frames.rec"

import mrcfile
with mrcfile.open(file_path) as mrc:
    print(mrc.header)
    print(mrc.extended_header)