import glob
import os
import re
import math
from typing import List
import seaborn as sns
import cv2
import numpy as np


def part_corpus_to_idxs(part_corpus, max_idx):
    if part_corpus:
        idxs = set()
        try:
            intervals = part_corpus.split(",")
            for inter in intervals:
                if "-" in inter:
                    start, end = inter.split("-")
                    if end < start:
                        tmp = end
                        end = start
                        start = tmp
                    for i in range(int(start), int(end) + 1):
                        idxs.add(i)
                else:
                    idxs.add(int(inter))

            return list(idxs)
        except:
            exit("part_corpus argument malformed")
    else:
        return list(range(max_idx))


def read_images(images_path: str, part_corpus=None):
    images = []
    images_ids = []
    rel_to_abs_page_no = dict()
    total_images = len([f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))])
    idxs = part_corpus_to_idxs(part_corpus, total_images)

    idx = 0
    for image_file in list(sorted(glob.iglob(os.path.join(images_path, "*")))):
        if os.path.isfile(image_file):
            if idx in idxs:
                images.append(cv2.imread(image_file))
                filename = re.findall(r'[^\/]+(?=\.)', image_file)[0]
                rel_to_abs_page_no[filename] = idx
                images_ids.append(filename)

            idx += 1

    return images, images_ids, rel_to_abs_page_no


def read_templates_filename(templates_folder):
    templates = []
    for template_file in list(sorted(glob.iglob(os.path.join(templates_folder, "*")))):
        if os.path.isfile(template_file):
            templates.append(template_file.split("/")[-1])

    return templates


def draw_patches(corpus_name, page_no, image, patches, patch_sampling, height, W):
    image_to_draw = image.copy()

    # patches_2d = patches.reshape(patches_2d_dim)

    count = 0
    for y in range(patches.shape[0]):
        pos_y = y * int(patch_sampling)
        for x in range(patches.shape[1]):
            pos_x = x * int(patch_sampling)
            cv2.rectangle(image_to_draw, (pos_x, pos_y), (pos_x + W, pos_y + height),
                          (0, 0, 255), 1)

            cv2.putText(image_to_draw, str(count % 10),
                        (pos_x, int(pos_y + height)),  # bottom left
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.25,
                        (255, 0, 0),
                        1,
                        2)
            count += 1

    results_folders = "/".join(['results', corpus_name])
    cv2.imwrite(results_folders + "/" + str(page_no) + "_" + str(W) + "_patches.png", image_to_draw)


def draw_key_points(corpus_name, page_no, image, key_points, scale, filename_prefix):
    image_to_draw = image.copy()

    for i in range(key_points.shape[0]):
        pos_y, pos_x = int(key_points[i, 0]), int(key_points[i, 1])
        try:
            cv2.rectangle(image_to_draw, (pos_x - int(scale / 2), pos_y - int(scale / 2)),
                          (pos_x + int(scale / 2), pos_y + int(scale / 2)), (255, 0, 0), 1)
        except Exception as e:
            print("error")

    results_folders = "/".join(['results', corpus_name])
    cv2.imwrite(results_folders + "/" + filename_prefix + str(page_no) + "_descriptor.png", image_to_draw)


def draw_visual_words(corpus_name, page_no, image, cells, key_points, scale, sift_step, K, filename_prefix):
    palette = list(reversed([(int(t[0] * 256), int(t[1] * 256), int(t[2] * 256)) for t in
                             sns.color_palette("Spectral_r", min(256, K))]))
    image_to_draw = image.copy()

    for y in range(key_points.shape[0]):
        for x in range(key_points.shape[1]):
            pos_y, pos_x = int(key_points[y, x, 0]), int(key_points[y, x, 1])
            # do not write the flagged visual words
            if cells[y][x] != K:
                try:
                    cv2.rectangle(image_to_draw, (pos_x, pos_y),
                                  (pos_x + sift_step, pos_y + sift_step),
                                  palette[cells[y][x] % len(palette)], 1)
                except Exception as e:
                    print("error")

    results_folders = "/".join(['results', corpus_name])
    cv2.imwrite(results_folders + "/" + filename_prefix + str(page_no) + "_visual_words.png", image_to_draw)


def generate_histogram(patches_by_scale, K):
    sub_histograms = []
    for patch in patches_by_scale:
        freq_patch_global = np.zeros(K + 1, dtype=np.ushort)
        unique, counts = np.unique(patch, return_counts=True)
        freq_patch_global[unique] = counts

        patch_left, patch_right = np.array_split(patch, 2, axis=1)

        freq_patch_right = np.zeros(K + 1, dtype=np.ushort)
        unique, counts = np.unique(patch_right, return_counts=True)
        freq_patch_right[unique] = counts

        freq_patch_left = np.zeros(K + 1, dtype=np.ushort)
        unique, counts = np.unique(patch_left, return_counts=True)
        freq_patch_left[unique] = counts

        sub_histograms.append(np.concatenate(
            # remove the last column of the histogram representing the flagged visual words
            (
                freq_patch_global[:-1],
                2 * freq_patch_left[:-1],
                2 * freq_patch_right[:-1]
            )
        ))

    return sum(sub_histograms)


def convert_orig_coord_to_dsift_space(x, y, bin_size, step, num_bin, ceiled=True):
    numFramesX = (x - (num_bin - 1) * bin_size) / step
    numFramesY = (y - (num_bin - 1) * bin_size) / step

    if ceiled:
        return math.ceil(numFramesY), math.ceil(numFramesX)
    else:
        return numFramesY, numFramesX


def find_local_maxima(array2d):
    return (
            (array2d >= np.roll(array2d, -1, -1)) &
            (array2d >= np.roll(array2d, 0, -1)) &
            (array2d >= np.roll(array2d, 1, -1)) &
            (array2d >= np.roll(array2d, -1, 0)) &
            (array2d >= np.roll(array2d, 1, 0)) &
            (array2d >= np.roll(array2d, -1, 1)) &
            (array2d >= np.roll(array2d, 0, 1)) &
            (array2d >= np.roll(array2d, 1, 1))
    )


def get_indices_of_n_largest(matrix, n):
    # Use argpartition to get the indices of the N largest values in the matrix
    indices = np.argpartition(matrix.flatten(), -n)[-n:]

    # Convert the indices to row and column indices
    rows, cols = np.unravel_index(indices, matrix.shape)
    return rows, cols


def find_candidates_new(sim, page_no, H, W, patch_sampling, threshold):
    voting_space = sim.copy()

    # find the local maxima indices
    local_maxima = np.where(find_local_maxima(voting_space))
    peaks = list()

    # construct the list of peak with x, y, similarity and page_no
    for i in range(len(local_maxima[0])):
        d = dict()
        sim = voting_space[local_maxima[0][i]][local_maxima[1][i]]
        d["sim"] = sim
        d["x"] = local_maxima[1][i] * patch_sampling
        d["y"] = local_maxima[0][i] * patch_sampling
        d["page_no"] = page_no
        peaks.append(d)

    peaks = group_overlapping_rectangles(peaks, W, H)

    return peaks, voting_space


def min_max_normalization(voting_spaces: List):
    total_max = 1
    total_min = -1

    # min-max normalization of each voting space
    for i in range(len(voting_spaces)):
        voting_spaces[i] = ((voting_spaces[i] - total_min) * (255 / (total_max - total_min))).astype(np.uint8)

    return voting_spaces


def group_overlapping_rectangles(patches: List, max_distance_x, max_distance_y):
    # sort the patches by coordinate ! Important ! Otherwise, algo not working
    patches = sorted(patches, key=lambda p: (p["x"], p["y"]))
    # add the first patch to the list of groups
    groups = [[patches[0]]]
    for i in range(1, len(patches)):
        grouped = False
        for group in groups:
            for p in group:
                if abs(p["x"] - patches[i]["x"]) < max_distance_x and abs(p["y"] - patches[i]["y"]) < max_distance_y:
                    group.append(patches[i])
                    grouped = True
                    break

            if grouped:
                break
        if not grouped:
            groups.append([patches[i]])

    assert sum(map(lambda x: len(x), groups)) == len(patches)
    # return groups
    return [max(group, key=lambda p: p["sim"]) for group in groups]


def get_most_convenient_patch_width(query_width, widths):
    return min(widths, key=lambda x: abs(x - query_width))


def cosine_similarity_raw__(u, v):
    v_norm = np.linalg.norm(v, axis=1)
    u_norm = np.linalg.norm(u)
    return np.nan_to_num((v @ u) / (v_norm * u_norm), nan=-1)


def flatten(l):
    return [item for sublist in l for item in sublist]


def to_gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def count_consecutive(arr, n):
    # pad a with False at both sides for edge cases when array starts or ends with n
    d = np.diff(np.concatenate(([False], arr == n, [False])).astype(int))
    # subtract indices when value changes from False to True from indices where value changes from True to False
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


# ['only-from-corpus', 'only-from-non-corpus', 'intersection', 'all']
def choose_templates_to_evaluate(templates_splitted, strategy):
    if strategy == "only-from-corpus":
        return templates_splitted["corpus"]
    elif strategy == "only-from-non-corpus":
        return templates_splitted["query"]
    elif strategy == "intersection":
        return [item for item in templates_splitted["query"] if
                item["word"] in [x["word"] for x in templates_splitted["corpus"]]]
    else:
        return templates_splitted["query"] + templates_splitted["corpus"]
