import os
from multiprocessing.pool import Pool
from lib.utilities_process import workernoshell,get_image_dimension
from cell_extractor.CellDetectorBase import CellDetectorBase
from multiprocessing.pool import Pool
import tqdm
import shutil
import numpy as np
import pandas as pd

class TiffSegmentor(CellDetectorBase):
    def __init__(self, animal,flurescence_image_input,nissel_stain_image_intput, n_workers=10, *args, **kwargs):
        super().__init__(animal, *args, **kwargs)
        self.flurescence_image_input = flurescence_image_input
        self.nissel_stain_image_input = nissel_stain_image_intput
        self.generate_tile_information()
        self.n_workers = n_workers
        self.check_files()
        self.create_output_folders(self.fluorescence_channel_output)
        self.create_output_folders(self.cell_body_channel_output)

    def set_image_width_and_height(self):
        files = os.listdir(self.flurescence_image_input)
        filei = os.path.join(self.flurescence_image_input,files[0])
        self.width, self.height = get_image_dimension(filei)
        self.tile_height = int(self.height / self.nrow )
        self.tile_width=int(self.width/self.ncol )
        

    def generate_tile_information(self):
        if not os.path.isfile(self.TILE_INFO_DIR):
            self.set_image_width_and_height()
            self.set_tile_origins()
            ntiles = len(self.tile_origins)
            tile_information = pd.DataFrame(columns = ['id','tile_origin','ncol','nrow','width','height'])
            for tilei in range(ntiles):
                tile_informationi = pd.DataFrame(dict(
                    id = [tilei],
                    tile_origin = [self.tile_origins[tilei]],
                    ncol = [self.ncol],
                    nrow = [self.nrow],
                    width = [self.width],
                    height = [self.height]) )
                tile_information = pd.concat([tile_information,tile_informationi],ignore_index=True)
            tile_information.to_csv(self.TILE_INFO_DIR,index = False)

    def set_tile_origins(self):
        assert hasattr(self,'width')
        self.tile_origins={}
        for i in range(self.nrow*self.ncol):
            row=int(i/self.ncol)
            col=i%self.ncol
            self.tile_origins[i] = (row*self.tile_height,col*self.tile_width)

    def check_files(self):
        flurescence_images = os.listdir(self.flurescence_image_input)
        nissel_stain_images = os.listdir(self.nissel_stain_image_input)
        flurescence_image_names = ['.'.join(i.split('.')[:-1]) for i in flurescence_images]
        nissel_stain_image_names = ['.'.join(i.split('.')[:-1]) for i in nissel_stain_images]
        assert np.all([i in flurescence_image_names for i in nissel_stain_image_names])
        assert np.all([i in nissel_stain_image_names for i in flurescence_image_names])
        self.image_names = flurescence_image_names
        self.extension = '.'+flurescence_images[0].split('.')[-1]
    
    def create_output_folders(self,output_folder):
        for filei in self.image_names:
            os.makedirs(os.path.join(output_folder,filei),exist_ok=True)
    
    def generate_tiles(self):
        self.generate_tiff_segments(self.flurescence_image_input,self.fluorescence_channel_output)
        self.generate_tiff_segments(self.nissel_stain_image_input,self.cell_body_channel_output)

    def get_path_to_output_folders(self, output_folder,tif_directory):
        files = os.listdir(tif_directory)
        file_names = ['.'.join(i.split('.')[:-1]) for i in files]
        path_to_output_folders = []
        for i in range(len(files)):
            file_name = file_names[i]
            path_to_output_folder = os.path.join(output_folder, file_name)
            path_to_output_folders.append(path_to_output_folder)
        return path_to_output_folders

    def generate_tiff_segments(self, tif_directory,output_folder):
        print(f"generate segment for channel: {tif_directory}")
        commands = []
        for filei in self.image_names:
            section_folder = os.path.join(output_folder,filei)
            if len(os.listdir(section_folder)) >= 10:
                continue
            cmd = [
                f"convert",
                os.path.join(tif_directory,filei+ self.extension),
                "-compress",
                "LZW",
                "-crop",
                f"{self.ncol}x{self.nrow}-0-0@",
                "+repage",
                "+adjoin",
                f"{section_folder}/{filei}tile-%d.tif",
            ]
            commands.append(cmd)
        print(f"working on {len(commands)} sections")
        with Pool(self.n_workers) as p:
            for _ in tqdm.tqdm(p.map(workernoshell, commands), total=len(commands)):
                pass