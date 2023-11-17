import os
import pandas as pd
from utils import Param
from dataset import Dataset
import matplotlib.pyplot as plt

dataset = Dataset(read_data=False)

for seq in dataset.sequences:
    print(seq)
    path_pickle = os.path.join(Param.path_sequence_base, f'shap_{seq}.p')

    mondict = dataset.load(path_pickle)
    df = pd.DataFrame(mondict.values(), columns=['sn', 'ip', 'po'])
    print(df.median())

    # Remove negative values
    df_filtered = df[df.apply(lambda x: (x >= 0).all(), axis=1)]

    # Compute IQR for the filtered DataFrame
    Q1 = df_filtered.quantile(0.25)
    Q3 = df_filtered.quantile(0.75)
    IQR = Q3 - Q1

    # Filter out outliers
    df = df_filtered[~((df_filtered < (Q1 - 1.5 * IQR)) | (df_filtered > (Q3 + 1.5 * IQR))).any(axis=1)]

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['sn'], df['ip'], df['po'])
    ax.set_xlabel('Sensor Noise')
    ax.set_ylabel('Initial Pose Uncertainty')
    ax.set_zlabel('Partial Overlap')
    dir = "mean_pert"
    if not os.path.exists():
        os.makedirs(dir)
    plt.savefig(f"mean_pert/{seq}.png")
