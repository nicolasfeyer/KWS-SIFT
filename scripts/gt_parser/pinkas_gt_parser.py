import glob
import os
import cv2
from bs4 import BeautifulSoup
from shapely.geometry import Polygon
from tqdm import tqdm

from scripts.gt_parser.ground_truth_parser import GroundTruthParser


class PinkasGTParser(GroundTruthParser):
    # try:
    #     self.crop_template(page_name, template_id, coords)
    # except:
    #     print("Error cropping template: " + template_id)
    # continue
    def crop_template(self, page_name, template_name, coords):
        im = cv2.imread("dataset/pinkas_dataset/" + page_name + ".jpg")
        gt_polygon = Polygon(coords)
        minx, miny, maxx, maxy = gt_polygon.bounds
        template_image = im[int(miny):int(maxy), int(minx):int(maxx)]
        cv2.imwrite("dataset/pinkas_dataset/templates/" + template_name + ".jpg", template_image)

    def read_ground_truth(self, ground_truth_folder):
        folder = "/".join([ground_truth_folder, '*.xml'])

        templates = list()

        page_no = 0
        for xml_file in list(sorted(glob.iglob(os.path.join(folder)))):
            print(xml_file)
            xml_ = BeautifulSoup(open(xml_file), 'xml')
            for path in tqdm(xml_.find_all("Word")):
                # list of x,y coordinates separated by space
                coords = [[float(x) for x in coord.split(",")] for coord in
                          path.findChildren("Coords", recursive=False)[0]["points"].split(" ")]
                word = path.findChildren("Unicode", recursive=True)[0].text
                page_name = xml_.find("Page")["imageFilename"]

                templates.append({"template_id": path["id"], "word": word, "coords": coords, "page_name": page_name,
                                  "page_no": page_no})

            page_no += 1

        return templates

    def get_template_file_path(self, templates_folder, template_d):
        return templates_folder + "/" + "".join(template_d["page_name"].split(".")[:-1]) + "-" + template_d[
            "template_id"] + ".jpg"
