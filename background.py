import time
import numpy as np
from dataset import Dataset
import multiprocessing
from subprocess import Popen

def main(seq, sn, iu):
    cmd = ["nice -n 10", "python3", "uncertainty.py", seq, str(sn), str(iu)]
    p = Popen(cmd)
    p.wait()

if __name__ == "__main__":
    dataset = Dataset()
    print("Starting!")
    start = time.time()
    with multiprocessing.Pool() as pool:
        args = [(seq, round(sn, 2), round(iu, 1)) for seq in dataset.sequences for sn in np.arange(0, 0.101, 0.01) for iu in np.arange(1, 2.101, 0.1)]
        pool.starmap(main, args)
    end = time.time()
    print("Finished! Total running time is:", end-start)
