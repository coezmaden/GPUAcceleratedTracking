# nvprof --profile-from-start off julia /path/to/ncu.jl
# or
# ncu --mode=launch julia /path/to/ncu.jl
using GPUAcceleratedTracking, CUDA, Tracking, GNSSSignals, StructArrays, BenchmarkTools, Test;
import Tracking: Hz, ms;
num_of_multiprocessors = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
system = GPSL1(use_gpu = Val(true));

## 1 ms signal track
# signal, sampling_frequency = gen_signal(system, 1, 1500Hz, 2500, num_ants = NumAnts(4))
# state = TrackingState(1, system, 1500Hz, 0, num_samples = 2500, num_ants = NumAnts(4))
# track(signal, state, sampling_frequency)
# CUDA.@profile track(signal, state, sampling_frequency)

## 1 ms signal downconvert and correlate
# Generate GNSS object and signal information
enable_gpu = Val(true)
num_samples = 2500
num_ants = 1
num_correlators = 3
algorithm = KernelAlgorithm(5)

system = GPSL1(use_gpu = enable_gpu)
codes = system.codes
codes_text_mem_simple = CuTexture(
    CuTextureArray(codes)
)
codes_text_mem = CuTexture(
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

# Generate the signal;
signal, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase)

# Generate correlator;
correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]

# Generate blank code and carrier replica, and downconverted signal;
code_replica = CUDA.zeros(Float32, num_samples + num_of_shifts)
code_replica_text_mem = CUDA.zeros(Float32, num_samples + num_of_shifts)
code_replica_cpu = zeros(Float32, num_samples + num_of_shifts)
carrier_replica = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples), CUDA.zeros(Float32, num_samples)))
downconverted_signal = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples, num_ants), CUDA.zeros(Float32, num_samples, num_ants)))

# Generate CUDA kernel tuning parameters;
threads_per_block = [1024, 512÷2]
blocks_per_grid = cld.(num_samples, threads_per_block)
partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, (blocks_per_grid[2], num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (blocks_per_grid[2], num_ants, length(correlator_sample_shifts)))))
shmem_size = sizeof(ComplexF32) * threads_per_block[2] * num_correlators * num_ants



code_replica_kernel = @cuda launch=false gen_code_replica_strided_kernel!(
    code_replica,
    codes, # texture memory codes
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length
)
blocks, threads = launch_configuration(code_replica_kernel.fun)
CUDA.@sync @cuda threads=threads blocks=blocks gen_code_replica_strided_kernel!(
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
@cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] shmem=shmem_size downconvert_and_correlate_kernel_2!(
    partial_sum.re,
    partial_sum.im,
    carrier_replica.re,
    carrier_replica.im,
    downconverted_signal.re,
    downconverted_signal.im,
    signal.re,
    signal.im,
    code_replica,
    correlator_sample_shifts,
    carrier_frequency,
    sampling_frequency,
    carrier_phase,
    num_samples,
    NumAnts(num_ants)
)
Tracking.cuda_reduce_partial_sum(partial_sum)