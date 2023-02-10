import glob
import os

from scripts.gt_parser.ground_truth_parser import GroundTruthParser


class StandardGTParser(GroundTruthParser):

    def read_ground_truth(self, ground_truth_folder):
        folder = "/".join([ground_truth_folder, '/*.tsv'])

        templates = list()

        for tsv_file in list(sorted(glob.iglob(os.path.join(folder)))):
            with open(tsv_file, 'r') as f:
                for line in f:
                    template_id, page_no, word, x, y, width, height = line.strip().split('\t')

                    templates.append(
                        {"template_id": template_id,
                         "word": word,
                         "coords": [[float(x), float(y)], [float(x) + float(width), float(y)],
                                    [float(x) + float(width), float(y) + float(height)],
                                    [float(x), float(y) + float(height)]],
                         "page_no": page_no})

        return templates

    def get_template_file_path(self, templates_folder, template_d):
        return templates_folder + "/" + template_d["template_id"] + ".jpg"
