import numpy as np
from dataset import Dataset
import multiprocessing
from subprocess import Popen

def main(seq, sn, iu):
    cmd = ["python3", "uncertainty.py", seq, str(sn), str(iu)]
    p = Popen(cmd)
    p.wait()

if __name__ == "__main__":
    dataset = Dataset()
    with multiprocessing.Pool() as pool:
        args = [(seq, round(sn, 1), round(iu, 1)) for seq in dataset.sequences for sn in np.arange(0, 2.1, 0.1) for iu in np.arange(1, 3.1, 0.1)]
        pool.starmap(main, args)
