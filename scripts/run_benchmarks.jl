using DrWatson
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, GNSSSignals, CUDA, Tracking, StructArrays, BenchmarkTools
using Tracking: Hz, ms
using Unitful
num_samples = 2500
carrier_doppler = 1500Hz
code_phase = 0
prn = 1
gpsl1_gpu = GPSL1(use_gpu = Val(true))
gpsl1_cpu = GPSL1()

allparams = Dict(
    "processor"   => ["CPU", "GPU"],
    "GNSS"  => [Val(GPSL1)],
    "num_samples" => [2500, 50000]
)

dicts = dict_list(allparams)

for (_, d) in enumerate(dicts)
    f = do_benchmark(d)
    wsave(datadir("benchmarks", savename(d, "jld2")), f)
end

function valtostring(gnss::Val(S)) where {S}
    String(typeof(gnss))
end