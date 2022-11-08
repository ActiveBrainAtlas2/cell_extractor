import os ,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import numpy as np
import pickle as pkl
from cell_extractor.CellDetectorBase import CellDetectorBase
import os
class MeanImageCalculator(CellDetectorBase):
    """class for calculated mean gaussian blurred image for background subtraction"""
    def __init__(self,animal, *args, **kwargs):
        super().__init__(animal, *args, **kwargs)
        self.calulate_average_cell_image()
    
    def calulate_average_cell_image(self):
        """calculate and saves the average cell image for subtraction
        """
        examples = self.load_all_examples_in_brain(label=1)
        self.average_image_ch1 = self.calculate_average_cell_images(examples,channel = 1)
        self.average_image_ch3 = self.calculate_average_cell_images(examples,channel = 3)
        average_image = dict(zip(['CH1','CH3'],[self.average_image_ch1,self.average_image_ch3]))
        pkl.dump(average_image,open(self.AVERAGE_CELL_IMAGE_DIR,'wb'))

    def calculate_average_cell_images(self,examples,channel = 3):
        """calculate average imaged for one channel"""
        images = []
        for examplei in examples:
            images.append(examplei[f'image_CH{channel}'])
        images = np.stack(images)
        average = np.average(images,axis=0)
        average = (average - average.mean())/average.std()
        return average

if __name__ == '__main__':
    calculator = MeanImageCalculator('DK79')