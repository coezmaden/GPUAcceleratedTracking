# julia --project=. /scripts/nsys.jl
# nsys launch julia /path/to/nsys.jl
using GPUAcceleratedTracking, CUDA, Tracking, GNSSSignals, StructArrays
import Tracking: Hz, ms
system = GPSL1(use_gpu = Val(true))
enable_gpu = Val(true)
num_samples = 2500
num_ants = 1
num_correlators = 3
system = GPSL1(use_gpu = enable_gpu)
codes = system.codes
#convert to text_mem
codes = CuTexture(
    CuTextureArray(codes),
    address_mode = CUDA.ADDRESS_MODE_WRAP,
    interpolation = CUDA.NearestNeighbour(),
    normalized_coordinates = true
)
code_frequency = get_code_frequency(system)
code_length = get_code_length(system)
start_code_phase = 0.0f0
carrier_phase = 0.0f0
carrier_frequency = 1500Hz
prn = 1
signal, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase)
correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]
code_replica = CUDA.zeros(Float32, num_samples + num_of_shifts)
carrier_replica = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples), CUDA.zeros(Float32, num_samples)))
downconverted_signal = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples, num_ants), CUDA.zeros(Float32, num_samples, num_ants)))
threads_per_block = zeros(Int, 3)
blocks_per_grid = zeros(Int, 3)
accum = StructArray{ComplexF32}((CUDA.zeros(Float32, (num_samples, num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (num_samples, num_ants, length(correlator_sample_shifts)))))
threads_per_block[3] = 1024
blocks_per_grid[3] = cld(num_samples, threads_per_block[3])
phi = StructArray{ComplexF32}((CUDA.zeros(Float32, (blocks_per_grid[3], num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (blocks_per_grid[3], num_ants, length(correlator_sample_shifts)))))
shmem_size = sizeof(ComplexF32) * 1024 * num_correlators * num_ants
code_replica_kernel = @cuda launch=false gen_code_replica_texture_mem_strided_kernel!(
        code_replica,
        codes,
        code_frequency,
        sampling_frequency,
        start_code_phase,
        prn,
        num_samples,
        num_of_shifts,
        code_length
    )
blocks_per_grid[1], threads_per_block[1] = launch_configuration(code_replica_kernel.fun)
downconvert_and_accumulate_kernel = @cuda launch=false downconvert_and_accumulate_strided_kernel!(
    accum.re,
    accum.im,
    code_replica,
    carrier_replica.re,
    carrier_replica.im,
    downconverted_signal.re,
    downconverted_signal.im,
    signal.re,
    signal.im,
    carrier_frequency,
    sampling_frequency,
    carrier_phase,
    num_samples,
    NumAnts(num_ants),
    correlator_sample_shifts
)
blocks_per_grid[2], threads_per_block[2] = launch_configuration(downconvert_and_accumulate_kernel.fun)
algorithm = KernelAlgorithm(2431)
kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    accum.re,
    accum.im,
    phi.re,
    phi.im,
    carrier_replica.re,
    carrier_replica.im,
    downconverted_signal.re,
    downconverted_signal.im,
    signal.re,
    signal.im,
    correlator_sample_shifts,
    carrier_frequency,
    carrier_phase,
    NumAnts(num_ants),
    nothing,
    algorithm
)
CUDA.@profile NVTX.@range "kernel_algorithm" begin
    kernel_algorithm(
    threads_per_block,
    blocks_per_grid,
    shmem_size,
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length,
    accum.re,
    accum.im,
    phi.re,
    phi.im,
    carrier_replica.re,
    carrier_replica.im,
    downconverted_signal.re,
    downconverted_signal.im,
    signal.re,
    signal.im,
    correlator_sample_shifts,
    carrier_frequency,
    carrier_phase,
    NumAnts(num_ants),
    nothing,
    algorithm
)
end