using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

# raw_data_df = collect_results(datadir("benchmarks/track"))
# raw_data_df = collect_results(datadir("benchmarks/kernel"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/test"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/jetson"))
raw_data_df = collect_results(datadir("benchmarks/kernel/kernelnaming1"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/cuda-elapsed"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/belapsed"))
os ="windows"
gnss="GPSL5"
# os ="windows"
## plot comparison of all kernels sweeping num_samples
using Plots; gr()
##
processor = "CPUvGPU"; num_ants = 1; num_correlators = 3; OS=os; GNSS=gnss;
params = @dict processor num_ants num_correlators OS GNSS

pl1 = plot_min_exec_time(raw_data_df, os=os, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl1
)
processor = "GPU"; num_ants = 1; num_correlators = 3; OS=os; GNSS=gnss;
params = @dict processor num_ants num_correlators OS GNSS
pl2 = plot_min_exec_time_gpu(raw_data_df, os=os, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl1
)

##
processor = "CPUvGPU"; num_ants = 4; num_correlators = 3; os=os;
params = @dict processor num_ants num_correlators
pl1 = plot_min_exec_time(raw_data_df, os=os, num_ants = 4, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl1
)
processor = "GPU"; num_ants = 4; num_correlators = 3; os=os;
params = @dict processor num_ants num_correlators
pl2 = plot_min_exec_time_gpu(raw_data_df, os=os, num_ants = 4, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl2
)

## 
processor = "CPUvGPU"; num_ants = 1; num_correlators = 7; os=os;
params = @dict processor num_ants num_correlators
pl1 = plot_min_exec_time(raw_data_df, os=os, num_correlators = 7, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl1
)
processor = "GPU"; num_ants = 1; num_correlators = 7; os=os;
params = @dict processor num_ants num_correlators
pl2 = plot_min_exec_time_gpu(raw_data_df, os=os, num_correlators = 7, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl2
)