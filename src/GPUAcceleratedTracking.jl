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

struct ReductionAlgorithm{x}
end

ReductionAlgorithm(x) = ReductionAlgorithm{x}()

const GNSSDICT = Dict(
    "GPSL1" => GPSL1,
    "GPSL5" => GPSL5,
    "GalileoE1B" => GalileoE1B
)

const ALGODICT = Dict(
    # "1_3_pure"                  => 1300,
    # "1_3_pure_textmem"          => 1301,
    # "1_3_cplx"                  => 1320,
    # "1_3_cplx_textmem"          => 1321,
    "1_3_cplx_multi"            => 1330,
    "1_3_cplx_multi_textmem"    => 1331,
    # "1_4_cplx_multi"            => 1430,
    "1_4_cplx_multi_textmem"    => 1431,
    "2_3_cplx_multi"            => 2330,
    "2_3_cplx_multi_textmem"    => 2331,
    "2_4_cplx_multi"            => 2430,
    "2_4_cplx_multi_textmem"    => 2431,
    "3_4_cplx_multi"            => 3430,
    "3_4_cplx_multi_textmem"    => 3431,
    "4_4_cplx_multi_textmem"    => 4431,
    "5_4_cplx_multi_textmem"    => 5431
)

const REDDICT = Dict(
    "pure" => ReductionAlgorithm(1),
    "cplx" => ReductionAlgorithm(2),
    "cplx_multi" => ReductionAlgorithm(3)
)

const ALGODICTINV = Dict(
    1300 => "1_3_pure"                   ,
    1301 => "1_3_pure_textmem"           ,
    1320 => "1_3_cplx"                   ,
    1331 => "1_3_cplx_textmem"           ,
    1330 => "1_3_cplx_multi"             ,
    1331 => "1_3_cplx_multi_textmem"     ,
    1430 => "1_4_cplx_multi"             ,
    1431 => "1_4_cplx_multi_textmem"     ,
    2330 => "2_3_cplx_multi"             ,
    2331 => "2_3_cplx_multi_textmem"     ,
    2430 => "2_4_cplx_multi"             ,
    2431 => "2_4_cplx_multi_textmem"     ,
    3430 => "3_4_cplx_multi"             ,
    3431 => "3_4_cplx_multi_textmem"     ,
    4431 => "4_4_cplx_multi_textmem"     , 
    5431 => "5_4_cplx_multi_textmem"    
)
include("algorithms.jl")
include("reduction.jl")
include("gen_signal.jl")
include("benchmarks.jl")
include("plots.jl")
include("results.jl")

export 
    gen_signal, 
    run_track_benchmark,
    run_kernel_benchmark,
    run_reduction_benchmark,
    add_results!,
    add_metadata!,
    eval_results,
    plot_min_exec_time,
    plot_min_exec_time_gpu,
    reduce_3,
    reduce_4,
    reduce_5,
    reduce_cplx_3,
    reduce_cplx_4,
    reduce_cplx_5,
    reduce_cplx_multi_3,
    reduce_cplx_multi_31,
    reduce_cplx_multi_4,
    reduce_cplx_multi_5,
    reduce_cplx_multi_nant_5,
    gen_code_replica_kernel!,
    gen_code_replica_strided_kernel!,
    gen_code_replica_texture_mem_kernel!,
    gen_code_replica_texture_mem_strided_kernel!,
    gen_code_replica_texture_mem_strided_nsat_kernel!,
    downconvert_strided_kernel!,
    downconvert_and_accumulate_strided_kernel!,
    downconvert_and_correlate_kernel_1330!,
    downconvert_and_correlate_kernel_1331!,
    downconvert_and_correlate_kernel_1431!,
    downconvert_and_correlate_kernel_3431!,
    downconvert_and_correlate_kernel_4431!,
    downconvert_and_correlate_kernel_5431!,
    downconvert_and_correlate_kernel_3d_4431!,
    cpu_reduce_partial_sum,
    cuda_reduce_partial_sum,
    kernel_algorithm,
    KernelAlgorithm,
    ReductionAlgorithm

end