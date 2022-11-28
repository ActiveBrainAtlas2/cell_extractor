import os
import sys
sys.path.append(os.environ['PROJECT_DIR'])
# sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from cell_extractor.ExampleFinder import create_examples_for_one_section 
import argparse

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal', type=str, help='Animal ID')
    parser.add_argument('--section', type=int, help='Secton being processed')
    parser.add_argument('--disk', type=str, help='storage disk')
    args = parser.parse_args()
    animal = args.animal
    section = args.section
    disk = args.disk
    
    create_examples_for_one_section(animal,section,disk=disk,segmentation_threshold = 2000)
