import os
from mean_shift import MeanShift
import pandas as pd
import numpy as np
import utils as utl
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError


def dir_path(path):
    if os.path.exists(path):
        return path
    else: 
        raise ArgumentTypeError(f"{path} does not exist")
    

def run_clustering(data_path, bandwidth):
    '''
    Creates cluster centers and labels data point
    inputs:
        data_path: path to csv file which contains points
        bandwidth: parameter used in gaussian kernel
    '''
    cols = ['timestamp', 'x', 'y']
    df = pd.read_csv(data_path, sep=';', names=cols, header=None)
    if not df.empty:
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
    else:
        raise ValueError("Data file is empty")
    data = np.stack((x, y), axis=1)
    ms = MeanShift()
    # perform clustering
    ms.fit(data, bandwidth)
    if ms.labels is not None:
        utl.plot(ms.labels)


def main():
    
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--bandwidth", default = 75.0, type=float, help="Bandwidth for gausian kernel")
    parser.add_argument("-p", "--path", default=os.path.abspath("Heineken.csv"), type=dir_path, help="Absolute path to csv file")
    args = vars(parser.parse_args())

    data_path = args["path"]
    bandwidth = args["bandwidth"]
    
    run_clustering(data_path, bandwidth)


if __name__ == "__main__":
    main()
