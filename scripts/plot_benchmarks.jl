using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

# raw_data_df = collect_results(datadir("benchmarks/track"))
# raw_data_df = collect_results(datadir("benchmarks/kernel"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/test"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/jetson"))
raw_data_df = collect_results(datadir("benchmarks/kernel/kernelnaming1"))

## plot comparison of all kernels sweeping num_samples

##
os = "windows"
processor = "CPUvGPU"; num_ants = 1; num_correlators = 3
params = @dict processor num_ants num_correlators
pl1 = plot_min_exec_time(raw_data_df)
wsave(
    plotsdir("atbenchmark", savename(params, "png")), 
    pl1
)
processor = "GPU"; num_ants = 1; num_correlators = 3
params = @dict processor num_ants num_correlators
pl2 = plot_min_exec_time_gpu(raw_data_df)
wsave(
    plotsdir("atbenchmark", savename(params, "png")), 
    pl1
)

##
processor = "CPUvGPU"; num_ants = 4; num_correlators = 3
params = @dict processor num_ants num_correlators
pl1 = plot_min_exec_time(raw_data_df, num_ants = 4)
wsave(
    plotsdir("atbenchmark", savename(params, "png")), 
    pl1
)
processor = "GPU"; num_ants = 4; num_correlators = 3
params = @dict processor num_ants num_correlators
pl2 = plot_min_exec_time_gpu(raw_data_df, num_ants = 4)
wsave(
    plotsdir("atbenchmark", savename(params, "png")), 
    pl2
)

## 
processor = "CPUvGPU"; num_ants = 1; num_correlators = 7
params = @dict processor num_ants num_correlators
pl1 = plot_min_exec_time(raw_data_df, num_correlators = 7)
wsave(
    plotsdir("atbenchmark", savename(params, "png")), 
    pl1
)
processor = "GPU"; num_ants = 1; num_correlators = 7
params = @dict processor num_ants num_correlators
pl2 = plot_min_exec_time_gpu(raw_data_df, num_correlators = 7)
wsave(
    plotsdir("atbenchmark", savename(params, "png")), 
    pl2
)