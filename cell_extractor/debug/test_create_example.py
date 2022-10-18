import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from cell_extractor.ExampleFinder import ExampleFinder 

def calculate_one_section(animal,section,disk,segmentation_threshold,replace):
    extractor = ExampleFinder(animal=animal,section=section,disk=disk,segmentation_threshold = segmentation_threshold,replace = replace)
    extractor.find_examples()
    extractor.save_examples()

if __name__ =='__main__':
    animal = 'DK79'
    section = 180
    disk = '/net/birdstore/Active_Atlas_Data'
    threshold=2000
    calculate_one_section(animal,section,disk=disk,segmentation_threshold = threshold,replace=True)
