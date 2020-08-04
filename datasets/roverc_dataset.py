# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset


class RoverCDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[1.8, 0, 0.5, 0],
                           [0, 1.8, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (320, 240)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:05d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, f_str)
        return image_path

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
