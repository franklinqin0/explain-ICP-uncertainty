# Explain ICP Uncertainty

This codebase implements Ziyuan Qin's work *Explanation of Uncertainty in Point Cloud Registration*. It uses [kernel SHAP](https://arxiv.org/abs/1705.07874) to find the associations between uncertainty sources and estimated uncertainty in ICP algorithms.

## Installation

After creating a Python virtual environment using `conda`, `pyenv virtualenv`, or `venv`, do `pip install -r requirements.txt`.

Then, clone the [fork](https://github.com/CAOR-MINES-ParisTech/libpointmatcher) of `libpointmatcher` by [3D ICP](https://arxiv.org/pdf/1909.05722.pdf).

```sh
cd 3d-icp-cov
git clone https://github.com/CAOR-MINES-ParisTech/libpointmatcher.git
cd libpointmatcher
mkdir build && cd build
cmake ..
make
sudo make install
cd ../..
```

## Data Preparation

First, then download `global_frame` and `local_frame` datasets from the *Challenging datasets* at [](https://projects.asl.ethz.ch/datasets/doku.php).

Then, create a directory `data/<sequence>/`, and put the `global_frame` and `local_frame` datasets under `data/<sequence>/global_frame/` and `data/<sequence>/local_frame/0_0/`.

Last, change variables `lpm_path`, `dir_path` and `results_base` of `class Param` in `utils.py` according to your paths.

After running ICP algorithms with different perturbations (see [next section](#run-pipeline)), the file structure would look like this:

```
├── data
│   ├── Apartment
│   │   ├── global_frame
│   │   │   ├── overlap_apartment.csv
│   │   │   ├── PointCloud<id>.csv
│   │   │   ├── pose_scanner_leica.csv
│   │   │   ├── xsens_Compass_mean.csv
│   │   │   ├── xsens_Gps_mean.csv
│   │   │   └── xsens_Gravity_mean.csv
│   │   ├── local_frame
│   │   │   ├── 0_0
│   │   │   │   ├── Hokuyo_<id>.csv
│   │   │   │   ├── <partial overlap>
│   │   │   │   │   ├── Hokuyo_<id>.csv
│   │   │   ├── <other sensor noise>
│   │   │   │   ├── <partial overlap>
│   │   │   │   │   ├── Hokuyo_<id>.csv
```

## Run Pipeline

Both experiments in the paper can be run. Experiment 1 determines the relationship between perturbation levels of uncertainty sources and uncertainty estimate for point clouds `scan_ref` and `scan_ref+1` in sequence `Apartment`. Experiment 2 determines the effects of uncertainty sources for contiguous point clouds in all sequences.

To run experiment 1, do:

```sh
python background.py <scan_ref> # run background dataset
python kernel_shap.py <scan_ref> # use kernel SHAP to generate plots
```

To run experiment 2, do:

```sh
python avg_parallel.py <seq> # the parallel version of average.py
python ks_avg_parallel.py <seq> # the parallel version of ks_avg.py
python plot_avg.py # visualize the inlier SHAP values
```
## Note

One could set `logger` in `/home/parallels/Desktop/idp/libpointmatcher/martin/config/base_config.yaml` to `NullLogger` to avoid seeing output from ICP.
