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
    "num_samples" => collect(2500:2500:50000),
    "OS" => os_name
)

dicts = dict_list(allparams)

for (_, d) in enumerate(dicts)
    benchmark_results = do_track_benchmark(d)
    @tagsave(datadir("benchmarks", savename("TrackFunctionBenchmark", d, "jld2")), benchmark_results)
end