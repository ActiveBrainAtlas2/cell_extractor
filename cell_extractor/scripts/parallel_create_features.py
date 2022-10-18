import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from cell_extractor.FeatureFinder import create_features_for_all_sections 
from cell_extractor.CellDetectorBase import CellDetectorBase

if __name__ =='__main__':
    base = CellDetectorBase()
    create_features_for_all_sections('DK79',disk = '/net/birdstore/Active_Atlas_Data/',segmentation_threshold=2000)
