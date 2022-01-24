using GPUAcceleratedTracking, DrWatson, Tracking, GNSSSignals, StructArrays, ProgressMeter
@quickactivate "GPUAcceleratedTracking"


allparams = Dict(
    "processor"   => ["CPU"],
    "GNSS"  => ["GalieoE1B"],
    "num_samples" => 2 .^ (11:17),
    "num_ants" => [1, 4],
    "num_correlators" => [3, 7],
    "algorithm" => [
        "1_4_cplx_multi_textmem",
        "2_4_cplx_multi_textmem",
        "3_4_cplx_multi_textmem",
        "4_4_cplx_multi_textmem",
        "5_4_cplx_multi_textmem"
    ]
)

dicts = dict_list(allparams)

# for (_, d) in enumerate(dicts)
#     benchmark_results = run_track_benchmark(d)
#     @tagsave(datadir("benchmarks/track", savename("TrackFunctionBenchmark", d, "jld2")), benchmark_results)
# end

@showprogress 1 "Benchmarking kernel algorithms" for (_, d) in enumerate(dicts)
    benchmark_results = run_kernel_benchmark(d)
    @tagsave(
        datadir("benchmarks/kernel/kernelnaming1", savename("KernelBenchmark", d, "jld2")), 
        benchmark_results
    )
end