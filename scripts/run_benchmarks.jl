using GPUAcceleratedTracking, DrWatson, Tracking, GNSSSignals, StructArrays
@quickactivate "GPUAcceleratedTracking"


allparams = Dict(
    "processor"   => ["GPU"],
    "GNSS"  => ["GPSL1"],
    "num_samples" => [5000, 50000, 500000],
    "num_ants" => [1],
    "num_correlators" => [3],
    "algorithm" => [5]
)

dicts = dict_list(allparams)

# for (_, d) in enumerate(dicts)
#     benchmark_results = run_track_benchmark(d)
#     @tagsave(datadir("benchmarks/track", savename("TrackFunctionBenchmark", d, "jld2")), benchmark_results)
# end

for (_, d) in enumerate(dicts)
    benchmark_results = run_kernel_benchmark(d)
    @tagsave(
        datadir("benchmarks/kernel/test", savename("KernelBenchmark", d, "jld2")), 
        benchmark_results
    )
end