import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from scripts.utils import part_corpus_to_idxs


def compute_iou(gt_coord, c, target_width, H):
    gt_polygon = Polygon(gt_coord)
    # minx, miny, maxx, maxy = Polygon(gt_coord).bounds
    # gt_polygon = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
    c_polygon = Polygon(
        [[c["x"], c["y"]],
         [c["x"] + target_width, c["y"]],
         [c["x"] + target_width, c["y"] + H],
         [c["x"], c["y"] + H]])
    # Calculate Intersection and union, and tne IOU
    try:
        polygon_intersection = gt_polygon.intersection(c_polygon).area
        polygon_union = gt_polygon.union(c_polygon).area
        return polygon_intersection / polygon_union
    except Exception as e:
        print(e)
        return 0


def draw_blank(axs, x_plot, H, target_width, y_offset_sim):
    blank = np.zeros((H, target_width, 3), np.uint8)
    blank[:] = (255, 255, 255)
    axs[x_plot].axis('off')
    axs[x_plot].imshow(blank, interpolation='nearest', cmap='Greys_r')
    axs[x_plot].text(0.5, y_offset_sim, "...", size=12, ha="center",
                     transform=axs[x_plot].transAxes)


def do_evaluate(images, gt_words, word, candidates, template, H, widths,
                target_width, worst_candidates, corpus_name, draw_results=False):
    gt_coords = [t["coords"] for t in gt_words]

    i = 1
    i_last_found = 0
    nbr_found = 0
    for c in candidates:
        c["found"] = False
        c["rank"] = i
        c["IOU"] = 0
        for gt_coord in gt_coords:
            c["IOU"] = max(compute_iou(gt_coord, c, target_width, H), c["IOU"])
            if c["IOU"] > 0.5:
                i_last_found = i
                nbr_found += 1
                c["found"] = True
                break

        if nbr_found == len(gt_coords):
            break
        i += 1

    if nbr_found == 0:
        return 0, 0, 0, 0, 0
    else:

        # cut the candidates list to the last found candidate
        candidates = candidates[:i_last_found]

        # Extract the predicted probabilities and ground truth labels from the candidates
        probs = [candidate['sim'] for candidate in candidates]
        labels = [candidate['found'] for candidate in candidates]
        ap = average_precision_score(labels, probs)

        # Compute the AUC
        if sum(labels) == 0:
            roc_auc = 0
        elif sum(labels) == len(labels):
            roc_auc = 1
        else:
            roc_auc = roc_auc_score(labels, probs)

        n_retrieved_relevant = sum(1 for candidate in candidates if candidate['found'])

        precision = n_retrieved_relevant / len(candidates)
        recall = n_retrieved_relevant / len(gt_coords)

        if recall > 1:
            recall = 1

        if draw_results:
            NBR_TEMPLATE = 1
            NBR_OF_CANDIDATES = min(len(candidates), 10)
            NBR_BLANK_SPACE = 1
            NBR_WORST = max(len(worst_candidates), 1)

            fig, axs = plt.subplots(1,
                                    NBR_TEMPLATE + NBR_OF_CANDIDATES + NBR_BLANK_SPACE + NBR_WORST,
                                    figsize=((NBR_TEMPLATE + NBR_OF_CANDIDATES + NBR_BLANK_SPACE + NBR_WORST) * 1.5, 3))

            fig.suptitle(
                'Page ' + template["page_name"]
                + " \"" + word + "\" : "
                + str(len(gt_words)) + " instance(s). "
                + str(sum(map(lambda x: x["found"], candidates))) + " found over the first " + str(
                    len(candidates)) + " candidates. "
                + str(round(ap * 100, 2)) + "% AP", fontsize=16)

            fig.tight_layout(pad=10.0)
            plt.axis('off')

            minx, miny, maxx, maxy = Polygon(template["coords"]).bounds

            y_offset_sim = -0.8 if widths.index(target_width) > 1 else -0.5
            y_offset_iou = 1.1 if widths.index(target_width) > 1 else 1.1
            y_offset_page_no = 1.5 if widths.index(target_width) > 1 else 1.5

            # Draw the template first
            axs[0].imshow(images[template["page_no"]][
                          int(miny):int(maxy),
                          int(minx):int(maxx)])
            axs[0].text(0.5, y_offset_page_no, "Page", size=12, ha="center", transform=axs[0].transAxes)
            axs[0].text(0.5, y_offset_sim, "Sim", size=12, ha="center", transform=axs[0].transAxes)
            axs[0].text(0.5, y_offset_iou, "IOU", size=12, ha="center", transform=axs[0].transAxes)
            axs[0].axis('off')

            for i in range(min(len(candidates), 10)):
                cdnt = candidates[i]
                page_no, x, y, sim, found, IOU = cdnt["page_no"], cdnt["x"], cdnt["y"], cdnt["sim"], \
                    cdnt["found"] if "found" in cdnt else False, cdnt["IOU"] if "IOU" in cdnt else 0

                to_draw = cv2.copyMakeBorder(
                    images[page_no][int(y):int(y + H), int(x):int(x + target_width)],
                    top=3,
                    bottom=3,
                    left=3,
                    right=3,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 255, 0] if found else [255, 0, 0]
                )

                axs[i + NBR_TEMPLATE].axis('off')
                axs[i + NBR_TEMPLATE].imshow(to_draw, interpolation='nearest', cmap='Greys_r')

                axs[i + NBR_TEMPLATE].text(0.5, y_offset_sim, str(round(sim, 3)), size=12, ha="center",
                                           transform=axs[i + 1].transAxes)

                axs[i + NBR_TEMPLATE].text(0.5, y_offset_iou, str(round(IOU, 3)), size=12, ha="center",
                                           transform=axs[i + 1].transAxes)

                # only write the page number for the choosen candidates (first row)
                axs[i + NBR_TEMPLATE].text(0.5, y_offset_page_no, str(page_no), size=12,
                                           ha="center",
                                           transform=axs[i + 1].transAxes)

            # Draw blank
            draw_blank(axs, NBR_TEMPLATE + NBR_OF_CANDIDATES, H, target_width, y_offset_sim)

            for i in range(len(worst_candidates)):
                worst_candidate = worst_candidates[i]
                page_no, x, y, sim = worst_candidate["page_no"], worst_candidate["x"], worst_candidate["y"], \
                    worst_candidate["sim"]

                to_draw = images[page_no][int(y):int(y + H),
                          int(x):int(x + target_width)]

                axs[NBR_TEMPLATE + NBR_OF_CANDIDATES + NBR_BLANK_SPACE + i].axis('off')
                try:

                    axs[NBR_TEMPLATE + NBR_OF_CANDIDATES + NBR_BLANK_SPACE + i].imshow(to_draw)
                except:
                    print("worst_candidates oups")

                axs[NBR_TEMPLATE + NBR_OF_CANDIDATES + NBR_BLANK_SPACE + i].text(0.5, y_offset_sim,
                                                                                 str(round(sim, 3)), size=12,
                                                                                 ha="center",
                                                                                 transform=axs[
                                                                                     NBR_TEMPLATE + NBR_OF_CANDIDATES + NBR_BLANK_SPACE + i].transAxes)

            results_folders = "/".join(['results', corpus_name])
            fig.savefig(results_folders + '/' + template["template_id"] + "_results_top10" + '.png')
            plt.draw()
            plt.clf()
            plt.close("all")

    return precision, recall, ap, roc_auc, len(word)


## ['only-from-corpus', 'only-from-non-corpus', 'intersection', 'all']
def split_train_test(images, templates, strategy, part_corpus=None):
    if part_corpus and strategy:

        idxs_corpus_images = part_corpus_to_idxs(part_corpus, len(images))
        idxs_query_images = list(set(list(range(0, len(images)))) - set(idxs_corpus_images))

        images_splitted = {"corpus": [], "query": []}
        templates_splitted = {"corpus": [], "query": []}

        if strategy == "only-from-corpus":
            for template in templates:
                if template["page_no"] in idxs_corpus_images:
                    templates_splitted["corpus"].append(template)
                    templates_splitted["query"].append(template)

            for ii in range(len(images)):
                if ii in idxs_corpus_images:
                    images_splitted["corpus"].append(images[ii])
                    images_splitted["query"].append(images[ii])

        elif strategy == 'only-from-non-corpus':
            for template in templates:
                if template["page_no"] in idxs_query_images:
                    templates_splitted["corpus"].append(template)
                    templates_splitted["query"].append(template)

            for ii in range(len(images)):
                if ii in idxs_query_images:
                    images_splitted["corpus"].append(images[ii])
                    images_splitted["query"].append(images[ii])

        elif strategy == "intersection":
            for template in templates:
                if template["page_no"] in idxs_corpus_images:
                    templates_splitted["corpus"].append(template)
                else:
                    templates_splitted["query"].append(template)

            for ii in range(len(images)):
                if ii in idxs_corpus_images:
                    images_splitted["corpus"].append(images[ii])
                else:
                    images_splitted["query"].append(images[ii])

        else:
            images_splitted = {"corpus": images, "query": images}
            templates_splitted = {"corpus": templates, "query": templates}
    else:
        images_splitted = {"corpus": images, "query": images}
        templates_splitted = {"corpus": templates, "query": templates}

    return images_splitted, templates_splitted


def filter_by_word_size(templates, template_text_limit):
    return list(filter(lambda x: len(x["word"]) > template_text_limit, templates))
