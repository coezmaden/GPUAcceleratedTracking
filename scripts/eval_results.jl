using GPUAcceleratedTracking, DrWatson, PrettyTables, Query, DataFrames
@quickactivate "GPUAcceleratedTracking"

table = eval_results(collect_results(datadir("benchmarks/kernel/kernelnaming1")))