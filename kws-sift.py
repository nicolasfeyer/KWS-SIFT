import gc
import pickle
from pathlib import Path

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm
import cv2
import numpy as np
import random

from scripts.args_parser import parse_args
from scripts.config import write_config, read_config
from scripts.evaluation import do_evaluate, split_train_test, filter_by_word_size
from scripts.gt_parser.standard_gt_parser import StandardGTParser
from scripts.gt_parser.washington_gt_parser import WashingtonGTParser
from scripts.lsa import lsa_transform
from scripts.patch_grouping import group_cells_by_patches
from scripts.query import query
from scripts.sift import get_and_labelize_sift_descriptors
from scripts.tf_idf import weight_with_tf_idf
from scripts.utils import read_images, \
    get_most_convenient_patch_width, choose_templates_to_evaluate, read_templates_filename, part_corpus_to_idxs

random.seed(42)
gc.set_debug(gc.DEBUG_UNCOLLECTABLE)

if __name__ == "__main__":

    args = parse_args()

    action = args.action
    corpus_name = args.corpus_name

    if action == "generate":

        images_folder = args.images_folder
        part_corpus = args.part_corpus
        sift_step = args.sift_step
        scales = args.bin_sizes
        magnitude_thresholds = args.magnitude_thresholds
        K = args.codebook_size
        H = args.patch_height
        Ws = args.patch_widths
        patch_sampling = args.patch_sampling
        T = args.topics

        Path("corpus").mkdir(parents=True, exist_ok=True)
        Path("corpus/" + corpus_name).mkdir(parents=True, exist_ok=True)
        Path("results").mkdir(parents=True, exist_ok=True)
        Path("results/" + corpus_name).mkdir(parents=True, exist_ok=True)

        write_config(args)

        images, image_ids, rel_to_abs_page_no = read_images(images_folder, part_corpus=part_corpus)

        labelled_cells, key_points = get_and_labelize_sift_descriptors(corpus_name, images, sift_step,
                                                                       scales, magnitude_thresholds, K,
                                                                       to_pickle=True, write_descr=True,
                                                                       write_vw=False)

        for W in Ws:
            dims_nbr_vw_by_patch = (int(H / sift_step), int(W / sift_step))

            print("[CORPUS] Group by Patch for width {}".format(W))
            patches = group_cells_by_patches(corpus_name, None, None, images, H, W, K, patch_sampling, scales,
                                             from_pickle=True, to_pickle=False, write_image=False)

            print("[CORPUS] Weighting Visual Words with TF-IDF {}".format(W))
            patches_tf_idf = weight_with_tf_idf(corpus_name, patches, W, K, from_pickle=False, to_pickle=False)

            print("[CORPUS] LSA Transform {}".format(W))
            lsa_transform(corpus_name, patches_tf_idf, W, K, T, patches_from_pickle=False, x_from_pickle=False,
                          to_pickle=True)

    else:

        templates_folder = args.templates_folder
        strategy = args.strategy
        ground_truth_folder = args.ground_truth_folder
        template_text_limit = args.template_text_limit

        config = read_config(corpus_name)

        images_folder = config["images_folder"]
        part_corpus = config["part_corpus"] if config["part_corpus"] else None
        sift_step = int(config["sift_step"])
        scales = [int(s) for s in config["bin_sizes"].split(",")]
        magnitude_thresholds = [float(s) for s in config["magnitude_thresholds"].split(",")]
        K = int(config["codebook_size"])
        H = int(config["patch_height"])
        Ws = [int(s) for s in config["patch_widths"].split(",")]
        patch_sampling = int(config["patch_sampling"])
        T = int(config["topics"])

        if not part_corpus and strategy != "union":
            exit("[ERROR] Considering that the corpus was generate with all images, the strategy must be 'union'")

        images, image_ids, rel_to_abs_page_no = read_images(images_folder)

        folder_corpus = "/".join(['corpus', corpus_name])
        with open(folder_corpus + '/centroids.pickle', 'rb') as handle:
            centroids = pickle.load(handle)

        if ground_truth_folder:
            # gt_parser_o = WashingtonGTParser()
            gt_parser_o = StandardGTParser()

            templates_l = gt_parser_o.read_ground_truth(ground_truth_folder)

            # word_coords, words_dict, template_dict = StandardGTParser() \
            #     .read_ground_truth(ground_truth_folder)

            # templates_l = PinkasGTParser() \
            #     .read_ground_truth(ground_truth_folder)

            images_splitted, templates_splitted = split_train_test(images, templates_l, strategy,
                                                                   part_corpus=part_corpus)
            if template_text_limit:
                for key, tmpls in templates_splitted.items():
                    templates_splitted[key] = filter_by_word_size(tmpls, template_text_limit)

            # word_image_ids = choose_templates_to_evaluate(templates_splitted, strategy)

            wid_by_width = {}

            # order the words by their width to optimize the reading of the corpus
            for tplt in templates_splitted["query"]:
                query_path = gt_parser_o.get_template_file_path(templates_folder, tplt)
                try:
                    query_width = cv2.imread(query_path).shape[1]
                    target_width = get_most_convenient_patch_width(query_width, Ws)
                    if target_width not in wid_by_width:
                        wid_by_width[target_width] = []
                    wid_by_width[target_width].append(tplt)
                except AttributeError as ae:
                    print("[WARNING] Template", query_path, "not found")

            aps = {}
            roc_aucs = {}
            nbr_char = {}
            precisions = {}
            recalls = {}

            for target_width, wids in wid_by_width.items():
                folder_width = "/".join(['corpus', corpus_name, str(target_width)])

                with open(folder_width + '/patches_lsa.pickle', 'rb') as handle:
                    patches = pickle.load(handle)

                with open(folder_width + '/lsa_X.pickle', 'rb') as handle:
                    X = pickle.load(handle)

                with open(folder_width + '/idf.pickle', 'rb') as handle:
                    idf = pickle.load(handle)

                aps[target_width] = []
                roc_aucs[target_width] = []
                recalls[target_width] = []
                precisions[target_width] = []
                nbr_char[target_width] = []

                for template in tqdm(wids):
                    print("----------------", template["template_id"], "------------------------")

                    candidates, worst_candidates = query(centroids, corpus_name, patches, images_splitted["corpus"],
                                                         sift_step, scales, magnitude_thresholds, H, K, Ws,
                                                         patch_sampling,
                                                         template, idf, X, templates_folder, gt_parser_o,
                                                         draw_heatmap=False)

                    print("[QUERY]  Evaluation")

                    # find the word to find in the train set
                    word = template["word"]

                    # find all the template showing the same word as the queried one
                    gt_words = list(filter(lambda x: x["word"] == word, templates_splitted["corpus"]))

                    precision, recall, ap, roc_auc, nbr_chars = do_evaluate(images,
                                                                            gt_words, word,
                                                                            candidates, template, H, Ws,
                                                                            target_width,
                                                                            worst_candidates,
                                                                            corpus_name, draw_results=False)

                    print("Precision   ", precision)
                    print("Recall      ", recall)
                    print("AP          ", ap)
                    print("ROC-AUC     ", roc_auc)

                    aps[target_width].append(ap)
                    roc_aucs[target_width].append(roc_auc)
                    precisions[target_width].append(precision)
                    recalls[target_width].append(recall)

                    gc.collect()

            print("================ RESULTS ================")
            global_aps = []
            global_roc_aucs = []
            global_precision = []
            global_recall = []

            for tw in Ws:
                if tw not in aps:
                    continue
                print("--------------", tw, "----------------")

                mean_ap = np.mean(aps[tw])
                print("Mean AP: ", mean_ap)
                median_ap = np.median(aps[tw])
                print("Median AP: ", median_ap)
                print()

                global_aps.extend(aps[tw])

                mean_roc_auc = np.mean(roc_aucs[tw])
                print("Mean ROC-AUC: ", mean_roc_auc)
                median_roc_auc = np.median(roc_aucs[tw])
                print("Median ROC-AUC: ", median_roc_auc)
                print()

                global_roc_aucs.extend(roc_aucs[tw])

                mean_precision = np.mean(precisions[tw])
                print("Mean precision: ", mean_precision)
                median_precision = np.median(precisions[tw])
                print("Median precision: ", median_precision)
                print()

                global_precision.extend(precisions[tw])

                mean_recall = np.mean(recalls[tw])
                print("Mean recall: ", mean_recall)
                median_recall = np.median(recalls[tw])
                print("Median recall: ", median_recall)
                print()

                global_recall.extend(recalls[tw])

            print("------------ Global --------------")
            print("mean AP: ", np.mean(global_aps))
            print("median AP: ", np.median(global_aps))
            print()
            print("mean ROC-AUC: ", np.mean(global_roc_aucs))
            print("median ROC-AUC: ", np.median(global_roc_aucs))
            print()
            print("mean Precisions: ", np.mean(global_precision))
            print("median Precisions: ", np.median(global_precision))
            print()
            print("mean Recalls: ", np.mean(global_recall))
            print("median Recalls: ", np.median(global_recall))

        else:
            idxs = part_corpus_to_idxs(part_corpus, len(images))
            images_selected = [images[i] for i in idxs]
            word_image_ids = read_templates_filename(templates_folder)
            wid_by_width = {}

            # order the words by their width to optimize the reading of the corpus
            for tplt in word_image_ids:
                query_path = templates_folder + "/" + tplt
                try:
                    query_width = cv2.imread(query_path).shape[1]
                    target_width = get_most_convenient_patch_width(query_width, Ws)
                    if target_width not in wid_by_width:
                        wid_by_width[target_width] = []
                    wid_by_width[target_width].append({"template_id": tplt})
                except AttributeError as ae:
                    print("[WARNING] Template", query_path, "not found")

            for target_width, wids in wid_by_width.items():
                folder_width = "/".join(['corpus', corpus_name, str(target_width)])

                with open(folder_width + '/patches_lsa.pickle', 'rb') as handle:
                    patches = pickle.load(handle)

                with open(folder_width + '/lsa_X.pickle', 'rb') as handle:
                    X = pickle.load(handle)

                with open(folder_width + '/idf.pickle', 'rb') as handle:
                    idf = pickle.load(handle)

                for template in tqdm(wids):
                    print("----------------", template["template_id"], "------------------------")
                    candidates_l, worst_candidates = query(centroids, corpus_name, patches, images_selected,
                                                           sift_step, scales, magnitude_thresholds, H, K, Ws,
                                                           patch_sampling,
                                                           template, idf, X, templates_folder, None,
                                                           draw_heatmap=True)

                    fig = plt.figure(figsize=(22, 1.6))
                    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                     nrows_ncols=(1, 11),  # creates 2x2 grid of axes
                                     axes_pad=0.5,  # pad between axes in inch.
                                     )

                    sub_img = cv2.imread(templates_folder + "/" + template["template_id"])
                    grid[0].imshow(sub_img, cmap='gray')
                    grid[0].set_title("Template", fontsize=16)
                    grid[0].axis('off')

                    for i in range(1, min(len(candidates_l), 11)):
                        c = candidates_l[i]
                        sub_img = images_selected[c["page_no"]][c["y"]:c["y"] + H, c["x"]:c["x"] + target_width]
                        grid[i].imshow(sub_img, cmap='gray')
                        grid[i].set_title(str(i) + ") " + str(round(c["sim"], 3)), fontsize=16)
                        grid[i].axis('off')

                    # plt.show()
                    results_folders = "/".join(['results', corpus_name])
                    fig.savefig(results_folders + "/top9_results_" + template["template_id"] + ".png")
                    plt.draw()
                    plt.clf()
                    plt.close("all")
