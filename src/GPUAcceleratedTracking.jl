module GPUAcceleratedTracking

using
    BenchmarkTools,
    CUDA,
    GNSSSignals,
    StructArrays,
    StaticArrays,
    Parameters,
    Tracking,
    Statistics,
    Dates,
    DataFrames,
    Query,
    PrettyTables,
    Plots,
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

include("algorithms.jl")
include("gen_signal.jl")
include("benchmarks.jl")
include("plots.jl")

export 
    gen_signal, 
    run_track_benchmark,
    run_kernel_benchmark,
    plot_min_exec_time,
    reduce_3,
    reduce_cplx_3,
    gen_code_replica_kernel!,
    gen_code_replica_strided_kernel!,
    gen_code_replica_texture_mem_kernel!,
    gen_code_replica_texture_mem_strided_kernel!,
    downconvert_strided_kernel!,
    downconvert_and_accumulate_strided_kernel!,
    downconvert_and_correlate_kernel_1!,
    downconvert_and_correlate_kernel_2!,
    downconvert_and_correlate_kernel_3!,
    downconvert_and_correlate_kernel_4!,
    downconvert_and_correlate_kernel_5!,
    downconvert_and_correlate_kernel_6!,
    downconvert_and_correlate_strided_kernel_2!
    # downconvert_and_correlate_isolated_kernel_5!,
    downconvert_and_correlate_strided_kernel_5!,
    cpu_reduce_partial_sum,
    cuda_reduce_partial_sum,
    kernel_algorithm,
    KernelAlgorithm

end