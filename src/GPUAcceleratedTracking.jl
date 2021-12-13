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

struct KernelAlgorithm{x}
end

KernelAlgorithm(x) = KernelAlgorithm{x}()

const GNSSDICT = Dict(
    "GPSL1" => GPSL1,
    "GPSL5" => GPSL5,
    "GalileoE1B" => GalileoE1B
)

include("gen_signal.jl")
include("benchmarks.jl")
include("plots.jl")

export 
    gen_signal, 
    run_track_benchmark,
    run_kernel_benchmark,
    plot_min_exec_time,
    KernelAlgorithm

end