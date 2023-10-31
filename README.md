# IDP

## Installation

## Run Pipeline

### Generate Datasert

### Explanation Module

Kernel SHAP

## TODO

should run icp.mc for sensor_noise=0.0, init_unc=1.0 before anything else

can set `logger` in `base_config.yaml` to `NullLogger` to avoid seeing output from ICP

[x] kernel_shap, `f`
[x] partial overlap
    [x] method to calculate overlap ratio correspond to overlap matrix
    [x] perturb input point cloud to a particular overlap ratio
    [x] vis perturbed point cloud
    [x] calc avg overlap
    [x] generate data
    [x] add to kernel shap
    [] recompute overlap: [](https://github.com/ethz-asl/libpointmatcher/blob/master/examples/compute_overlap.cpp)
[x] write report
[x] compare 30 vs. 100, mean vs. scan_in = scan_ref
[x] fix `f`
[x] replace avg with randomly selected instance
[] summary_plot, dependence_plot
[] scan_in - scan_ref > 1
[] another ICP algorithm
<!-- [] do calculus -->
