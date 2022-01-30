using GPUAcceleratedTracking, DrWatson, ProgressMeter
@quickactivate "GPUAcceleratedTracking"

allparams = Dict(
    "num_samples" => 2 .^ (11:18),
    "algorithm" => ["gmem", "textmem"]
)

dicts = dict_list(allparams)

@showprogress 0.5 "Benchmarking reduction algorithms (pure vs cplx vs cplx_multi)" for (_, d) in enumerate(dicts)
    benchmark_results = run_replica_benchmark(d)
    @tagsave(
        datadir("benchmarks/codereplica", savename("CodeReplica", d, "jld2")), 
        benchmark_results
    )
end