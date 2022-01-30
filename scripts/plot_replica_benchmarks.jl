using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

raw_data_df = collect_results(datadir("benchmarks/codereplica"))

fig = plot_replica_benchmark(raw_data_df)

save( plotsdir("benchmark_textmem.pdf"), fig)