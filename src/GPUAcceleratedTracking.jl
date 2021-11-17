module GPUAcceleratedTracking

using
    BenchmarkTools,
    CUDA,
    GNSSSignals,
    StructArrays,
    Parameters,
    Tracking
    
import Unitful: MHz, kHz, Hz, s, ms, dBHz, ustrip, NoUnits
import Tracking: TrackingState, NumAnts, NumAccumulators

include("gen_signal.jl")
include("benchmark_loop.jl")

export 
    gen_signal, 
    do_benchmark
end