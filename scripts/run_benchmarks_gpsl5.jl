using GPUAcceleratedTracking, DrWatson, Tracking, GNSSSignals, StructArrays, ProgressMeter
@quickactivate "GPUAcceleratedTracking"


allparams = Dict(
    "processor"   => ["GPU"],
    "GNSS"  => ["GPSL5"],
    "num_samples" => 2 .^ (15:18),
    "num_ants" => [1],
    "num_correlators" => [7],
    "algorithm" => [
        "1_4_cplx_multi_textmem",
        "2_4_cplx_multi_textmem",
        "3_4_cplx_multi_textmem",
        "4_4_cplx_multi_textmem",
        "5_4_cplx_multi_textmem"
    ]
)

dicts = dict_list(allparams)

@showprogress 1 "Benchmarking kernel algorithms" for (_, d) in enumerate(dicts)
    benchmark_results = run_kernel_benchmark(d)
    @tagsave(
        datadir("benchmarks/kernel/kernelnaming1", savename("KernelBenchmark", d, "jld2")), 
        benchmark_results
    )
end