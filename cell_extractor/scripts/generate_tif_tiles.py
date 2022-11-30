import os 
import sys
sys.path.append(os.environ['PROJECT_DIR'])
from lib.TiffSegmentor import TiffSegmentor
import argparse
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal', type=str, help='Animal ID')
    parser.add_argument('--fluorescence_image', type=str, help='Where fluorescence images are stored')
    parser.add_argument('--nissel_stain_image', type=str, help='Where nissel stained images are stored')
    parser.add_argument('--disk', type=str, help='Storage Disk')
    parser.add_argument('--njobs', type=int, help='Number of parallel jobs',default=10)
    args = parser.parse_args()
    animal = args.animal
    fluorescence_image = args.fluorescence_image
    nissel_stain_image = args.nissel_stain_image
    disk = args.disk
    njobs = args.njobs
    
    segmentor = TiffSegmentor(animal,fluorescence_image,nissel_stain_image,disk = disk,n_workers = njobs)
    segmentor.generate_tiles()

