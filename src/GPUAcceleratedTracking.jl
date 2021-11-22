module GPUAcceleratedTracking

using
    BenchmarkTools,
    CUDA,
    GNSSSignals,
    StructArrays,
    Parameters,
    Tracking,
    Statistics,
    Dates,
    DataFrames,
    Query,
    Plots,
    StatsPlots,
    PGFPlotsX
    
import Unitful: MHz, kHz, Hz, s, ms, dBHz, ustrip, NoUnits
import Tracking: TrackingState, NumAnts, NumAccumulators

include("gen_signal.jl")
include("benchmarks.jl")
include("plots.jl")


const GNSSDICT = Dict(
    "GPSL1" => GPSL1,
    "GPSL5" => GPSL5,
    "GalileoE1B" => GalileoE1B
)

export 
    gen_signal, 
    do_track_benchmark,
    do_kernel_wrapper_benchmark,
    do_kernel_nowrapper_benchmark
    plot_min_exec_time

end