import os
import sys
sys.path.append(os.environ['PROJECT_DIR'])
from cell_extractor.CellDetector import detect_cell
import argparse

def run_from_terminal():
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal', type=str, help='Animal ID')
    parser.add_argument('--disk', type=str, help='storage disk')
    parser.add_argument('--round', type=int, help='model version',default=2)
    args = parser.parse_args()
    braini = args.animal
    disk = args.disk
    round = args.round
    detect_cell(braini,disk = disk,round=round)

if __name__ =='__main__':
    run_from_terminal()
   