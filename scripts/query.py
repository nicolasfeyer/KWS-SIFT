import math
import cv2
import numpy as np
from sklearn import preprocessing

from scripts.sift import get_and_labelize_sift_descriptors
from scripts.utils import generate_histogram, cosine_similarity_raw__, find_candidates_new, min_max_normalization, \
    get_most_convenient_patch_width


def query(centroids, corpus_name, patches, images, sift_step,
          feature_sizes, magnitude_thresholds, H, K, available_width, patch_sampling, template,
          idf, X, templates_folder, gt_parser, draw_heatmap=False):
    if gt_parser:
        query_path = gt_parser.get_template_file_path(templates_folder, template)
    else:
        query_path = templates_folder + "/" + template["template_id"]

    if X is not None and idf is None:
        print("Cannot use LSA without TF-IDF")
        return

    query_image = cv2.imread(query_path)

    target_width = get_most_convenient_patch_width(query_image.shape[1], available_width)

    print("[QUERY]  Patch width selected:", target_width)

    labelled_cells, key_points = get_and_labelize_sift_descriptors(corpus_name, query_image, sift_step,
                                                                   feature_sizes, magnitude_thresholds, K, centroids,
                                                                   to_pickle=False, write_descr=False, write_vw=False,
                                                                   filename_prefix=template["template_id"])

    patch = generate_histogram(list(labelled_cells[0].values()), K)

    if idf is not None:
        print("[QUERY]  Weighting Visual Words with TF-IDF")

        patch = preprocessing.normalize(patch.reshape(1, K * 3) * np.hstack([idf] * 3), norm='l2')

        if X is not None:
            print("[QUERY]  LSA Transform")
            #
            # This setup has obtained better experimental results than applying the LSA technique directly to
            # the local patch descriptor fj.By separately transforming each sub-vector, the spatial
            # information encoded by the SPM scheme is maintained.

            patch = np.hstack((patch[:, :K].dot(X),
                               patch[:, K:2 * K].dot(X),
                               patch[:, 2 * K:3 * K].dot(X)))

        # patch = np.float32(preprocessing.normalize(patch, norm='l2'))[0]
        patch = np.float32(patch)[0]

    print("[QUERY]  Find candidates")

    # Compute similarities between query and patches

    sims = {}

    for page_no, patches_ in patches.items():
        patches_2d_dim = (
            math.floor((images[page_no].shape[0] - H) / patch_sampling),
            math.floor((images[page_no].shape[1] - target_width) / patch_sampling),
        )
        sims[page_no] = cosine_similarity_raw__(np.float32(patch), np.float32(patches_)) \
            .reshape((patches_2d_dim[0], patches_2d_dim[1]))

    candidates_l = list()
    voting_spaces = list()

    # find candidates
    for page_no, sim in sims.items():
        peaks, voting_space = find_candidates_new(sim, page_no, H, target_width, patch_sampling, threshold=0)
        candidates_l.extend(peaks)
        voting_spaces.append(voting_space)

    voting_spaces = min_max_normalization(voting_spaces)

    ## sort and keep the first 10'000 candidates
    # candidates_l = list(sorted(candidates_l, key=lambda x: max(x, key=lambda p: p["sim"])["sim"], reverse=True))
    candidates_l = list(sorted(candidates_l, key=lambda x: x["sim"], reverse=True))
    worst_candidates = candidates_l[-5:]
    # candidates_l = candidates_l[0:min(top_n_candidates, len(candidates_l))]

    if draw_heatmap:
        for i_m in range(len(images)):
            # #### HEATMAP
            voting_space_draw = voting_spaces[i_m].copy()
            voting_space_draw = np.lib.pad(voting_space_draw,
                                           ((math.floor((H / patch_sampling) / 2),
                                             math.ceil((H / patch_sampling) / 2)),
                                            (math.floor((target_width / patch_sampling) / 2),
                                             math.ceil((target_width / patch_sampling) / 2))),
                                           'constant', constant_values=(0))

            heatmapshow = cv2.resize(voting_space_draw, (images[i_m].shape[1], images[i_m].shape[0]), cv2.INTER_LINEAR)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_TURBO)

            alpha = 0.5
            beta = (1.0 - alpha)
            heatmapshow = cv2.addWeighted(images[i_m], alpha, heatmapshow, beta, 0.0)

            # cv2.imwrite('res/' + word_image_id + '_heatmap_page' + str(i_m) + ".jpg", heatmapshow)

            for ic in range(len(candidates_l)):
                if candidates_l[ic]["page_no"] == i_m:
                    top = candidates_l[ic]

                    page_no_c, x, y, _ = top["page_no"], top["x"], top["y"], top["sim"]
                    cv2.rectangle(heatmapshow,
                                  (x, y),
                                  (x + target_width, y + H),
                                  (0, 0, 255),
                                  3)

                    cv2.putText(heatmapshow, str(ic),
                                (int(x + target_width / 2),
                                 int(y + H / 2)
                                 ),  # bottom left
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 255),
                                2,
                                2)

                    # for jc in range(len(cdnt[1:])):
                    #     page_no_c, x, y, _ = cdnt[jc]["page_no"], cdnt[jc]["x"], cdnt[jc]["y"], cdnt[jc]["sim"]
                    #     cv2.rectangle(heatmapshow,
                    #                   (x, y),
                    #                   (x + target_width, y + H),
                    #                   (255, 0, 0),
                    #                   3)
                    #
                    #     cv2.putText(heatmapshow, str(ic) + "-" + str(jc),
                    #                 (int(x + target_width / 2),
                    #                  int(y + H / 2)
                    #                  ),  # bottom left
                    #                 cv2.FONT_HERSHEY_SIMPLEX,
                    #                 1,
                    #                 (255, 0, 0),
                    #                 2,
                    #                 2)
            results_folders = "/".join(['results', corpus_name])
            cv2.imwrite(results_folders + '/' + template["template_id"] + '_candidates_page' + str(i_m) + ".jpg",
                        heatmapshow)

    return candidates_l, worst_candidates
