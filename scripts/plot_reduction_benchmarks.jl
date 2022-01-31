using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

raw_data_df = collect_results(datadir("benchmarks/reduction"))

num_ants = 4; num_correlators = 3;

fig_bar = plot_reduction_benchmark(raw_data_df, num_ants, num_correlators)
save( plotsdir("reduction.pdf"), fig_bar)    