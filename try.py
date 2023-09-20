from utils import *

Param.sensor_noise = 0.01
Param.init_unc = 1.0
Param.update()
seq = "Apartment"
Param.results_path = os.path.join(Param.results_base, seq, dec2str(0.0), dec2str(1.0))
Param.results_pert = os.path.join(Param.results_base, seq, dec2str(Param.sensor_noise), dec2str(Param.init_unc))
print("results_path:", Param.results_path)
print("results_pert:", Param.results_pert)
print("path_pc:", Param.path_pc)