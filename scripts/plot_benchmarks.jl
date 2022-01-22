using DrWatson, PrettyTables, Query, DataFrames
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, DataFrames, Plots, PGFPlotsX

# df = collect_results(datadir("benchmarks/track"))
# df = collect_results(datadir("benchmarks/kernel"))
# df = collect_results(datadir("benchmarks/kernel/test"))
raw_data_df = collect_results(datadir("benchmarks/kernel/jetson"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/kernelnaming1"))

## plot comparison of all kernels sweeping num_samples
plot_min_exec_time(raw_data_df)
plot_min_exec_time_gpu(raw_data_df)

## plot comparison of all kernels sweeping num_samples
# num_ants = 4, num_correlators = 3
plot_min_exec_time(raw_data_df, num_ants = 4)
plot_min_exec_time_gpu(raw_data_df, num_ants = 4)

## plot comparison of all kernels sweeping num_samples
# num_ants = 1, num_correlators = 7
plot_min_exec_time(raw_data_df, num_correlators = 7)
plot_min_exec_time_gpu(raw_data_df, num_correlators = 7)