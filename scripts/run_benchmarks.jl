using DrWatson
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, GNSSSignals, CUDA, Tracking, StructArrays, BenchmarkTools
using Tracking: Hz, ms
using Unitful

os_name = @static Sys.iswindows() ? "windows" : (@static Sys.isapple() ? "macos" : @static Sys.islinux() ? "linux" : @static Sys.isunix() ? "generic_unix" : throw("Can't determine OS name"))

allparams = Dict(
    "processor"   => ["CPU", "GPU"],
    "GNSS"  => ["GPSL1"],
    "num_samples" => [2500, 50000],
    "num_ants" => [1],
    "num_correlators" => [3],
    "OS" => os_name,
    "Algorithm" => [1]
)

dicts = dict_list(allparams)

for (_, d) in enumerate(dicts)
    benchmark_results = do_track_benchmark(d)
    @tagsave(datadir("benchmarks/track", savename("TrackFunctionBenchmark", d, "jld2")), benchmark_results)
end

for (_, d) in enumerate(dicts)
    benchmark_results = do_kernel_wrapper_benchmark(d)
    @tagsave(datadir("benchmarks/kernel/nowrapper", savename("KernelBenchmark", d, "jld2")), benchmark_results)
end