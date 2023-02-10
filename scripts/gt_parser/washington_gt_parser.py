import glob
import os
import string

from bs4 import BeautifulSoup
from svgpath2mpl import parse_path

from scripts.gt_parser.ground_truth_parser import GroundTruthParser


class WashingtonGTParser(GroundTruthParser):

    def read_words(self, ground_truth_folder):
        lines = list(open("/".join([ground_truth_folder, 'Words.txt']), 'r').readlines())

        template_id_word = dict()

        for l in lines:
            id_, word = l.strip().split(" ")

            assert len(id_) == 9

            word = word.replace("s_pt", ".")
            word = word.replace("s_cm", ",")
            word = word.replace("s_sq", ";")
            word = word.replace("s_qo", ":")
            word = word.replace("s_mi", "$")
            word = word.replace("s_", "")  # digit prefix
            word = word.replace("_", " ")
            word = word.replace("-", "")
            word = word.replace("$", "-")
            word = word.translate(str.maketrans('', '', string.punctuation))

            if word == "":
                continue

            template_id_word[id_] = word

        return template_id_word

    def read_ground_truth(self, ground_truth_folder):
        folder = "/".join([ground_truth_folder, '*.svg'])

        templates = list()

        template_id_word = self.read_words(ground_truth_folder)

        page_no = 0
        for svg_file in list(sorted(glob.iglob(os.path.join(folder)))):
            for path in BeautifulSoup(open(svg_file), 'html.parser').find_all("path"):
                svg_d = path["d"]
                id_template = path["id"]

                if id_template in template_id_word:
                    if id_template and svg_d and id_template != "null":
                        mpl_path = parse_path(svg_d)
                        coordinates = mpl_path.to_polygons()

                        templates.append(
                            {"template_id": path["id"], "word": template_id_word[id_template], "coords": coordinates[0],
                             "page_name": svg_file.split("/")[-1].split(".")[0] + ".jpg",
                             "page_no": page_no})

            page_no += 1

        return templates

    def get_template_file_path(self, templates_folder, template_d):
        return templates_folder + "/" + template_d["template_id"] + ".jpg"
