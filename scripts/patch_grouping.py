import gc
import math
import os
import pickle
from pathlib import Path
import psutil
import numpy as np
from tqdm import tqdm

from scripts.utils import generate_histogram, draw_patches, count_consecutive


def group_cells_by_patches(corpus_name, labelled_cells, key_points, images, H: int, W: int, K: int, patch_sampling,
                           scales,
                           from_pickle=False, to_pickle=True, write_image=False):
    if from_pickle:
        with open("/".join(['corpus', corpus_name, 'cells_labeled.pickle']), 'rb') as handle:
            labelled_cells = pickle.load(handle)
        with open("/".join(['corpus', corpus_name, 'key_points.pickle']), 'rb') as handle:
            key_points = pickle.load(handle)

    patches = {}
    patches_2d_dim = {}
    dims_nbr_vw_by_patch = [None for _ in range(len(scales))]
    for page_no, visual_words in labelled_cells.items():

        # compute the dimension of the patches
        patches_2d_dim[page_no] = (
            math.floor((images[page_no].shape[0] - H) / patch_sampling),
            math.floor((images[page_no].shape[1] - W) / patch_sampling),
            K * 3
        )

        patches[page_no] = np.zeros(patches_2d_dim[page_no], dtype=np.ushort)

        # dense sampling of the visual words with a step of "patch_sampling" relative to the original image
        for y_patch in tqdm(range(patches_2d_dim[page_no][0])):
            patch_ya, patch_yb = y_patch * patch_sampling, y_patch * patch_sampling + H
            for x_patch in range(patches_2d_dim[page_no][1]):
                patch_xa, patch_xb = x_patch * patch_sampling, x_patch * patch_sampling + W

                patchs_by_scale = []
                for i in range(len(scales)):
                    scale = scales[i]
                    if scale in key_points[page_no]:
                        # get the y coordinates of the visual words
                        key_points_2d_y = key_points[page_no][scale][:, :, 0]  # top left corner
                        # get the x coordinates of the visual words
                        key_points_2d_x = key_points[page_no][scale][:, :, 1]  # top left corner

                        # get the visual words that are inside the patch as a patch
                        mask = (patch_ya <= key_points_2d_y) & (key_points_2d_y < patch_yb) & \
                               (patch_xa <= key_points_2d_x) & (key_points_2d_x < patch_xb)

                        # filter the key points that are inside the patch
                        key_points_masked = key_points[page_no][scale][mask]

                        if len(key_points_masked) > 0:  # if there are visual words inside the patch

                            # find the width and height of the patch
                            # this is computed only once
                            # if dims_nbr_vw_by_patch[i] is None:
                            vw_width = int(count_consecutive(key_points_masked[:, 0], key_points_masked[:, 0][0])[0])
                            vw_height = int(len(key_points_masked) / vw_width)
                            dims_nbr_vw_by_patch = (vw_height, vw_width)

                            patchs_by_scale.append(visual_words[scale][mask].reshape(dims_nbr_vw_by_patch))

                patches[page_no][y_patch][x_patch] = np.ushort(generate_histogram(patchs_by_scale, K))

        if write_image:
            draw_patches(corpus_name, page_no, images[page_no], patches[page_no],patch_sampling, H, W)

        # reshape from 3D to a 2D matrix (list of patches)
        patches[page_no] = patches[page_no].reshape(patches_2d_dim[page_no][0] * patches_2d_dim[page_no][1], 3 * K)

    folder = "/".join(['corpus', corpus_name, str(W)])
    Path(folder).mkdir(parents=True, exist_ok=True)

    gc.collect()
    print("[MEMORY USED]", psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")

    if to_pickle:
        with open(folder + "/" + 'patches_histogram.pickle', 'wb') as handle:
            pickle.dump(patches, handle)

    return patches
