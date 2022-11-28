import os
import sys
sys.path.append(os.environ['PROJECT_DIR'])
# sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from cell_extractor.ExampleFinder import create_examples_for_all_sections 
import argparse

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal', type=str, help='Animal ID')
    parser.add_argument('--disk', type=str, help='storage disk')
    parser.add_argument('--njobs', type=int, help='Number of parallel jobs',default=10)
    args = parser.parse_args()
    animal = args.animal
    disk = args.disk
    njobs = args.njobs
    create_examples_for_all_sections(animal,disk = disk,segmentation_threshold=2000,njobs=njobs)
