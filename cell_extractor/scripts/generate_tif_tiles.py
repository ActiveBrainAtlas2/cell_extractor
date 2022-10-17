import os 
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)+'/../..'))
from lib.TiffSegmentor import TiffSegmentor
# import argparse
if __name__ =='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--animal', type=str, help='Animal ID')
    # parser.add_argument('--flurescence_image', type=str, help='Where fluorescence images are stored')
    # parser.add_argument('--nissel_stain_image', type=str, help='Where nissel stained images are stored')
    # parser.add_argument('--disk', type=str, help='Storage Disk')
    # parser.add_argument('--njobs', type=int, help='Number of parallel jobs',default=10)
    # args = parser.parse_args()
    # animal = args.animal
    # flurescence_image = args.flurescence_image
    # nissel_stain_image = args.nissel_stain_image
    # disk = args.disk
    # njobs = args.njobs
    animal='DK79'
    flurescence_image = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK79/preps/CH2/full_aligned/'
    nissel_stain_image = '/net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK79/preps/CH1/full_aligned/'
    segmentor = TiffSegmentor( animal,flurescence_image,nissel_stain_image,n_workers = 10)
    segmentor.generate_tiles()
