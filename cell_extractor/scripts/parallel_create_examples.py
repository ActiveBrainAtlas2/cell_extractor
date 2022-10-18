import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from cell_extractor.ExampleFinder import create_examples_for_all_sections 
import argparse

if __name__ =='__main__':
    animal = 'DK79'
    create_examples_for_all_sections(animal,disk = '/net/birdstore/Active_Atlas_Data/',segmentation_threshold=2000,njobs=7)
