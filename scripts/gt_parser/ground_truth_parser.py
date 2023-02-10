from abc import ABC, abstractmethod


class GroundTruthParser(ABC):
    @abstractmethod
    def read_ground_truth(self, ground_truth_folder):
        pass

    @abstractmethod
    def get_template_file_path(self, template_folder, template_d):
        pass
