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
include("benchmark_functions.jl")

const GNSSDICT = Dict(
    "GPSL1" => GPSL1,
    "GPSL5" => GPSL5,
    "GalileoE1B" => GalileoE1B
)

export 
    gen_signal, 
    do_track_benchmark

end