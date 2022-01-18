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
num_samples = 50000
num_ants = 1
num_correlators = 3
# algorithm = KernelAlgorithm(5)

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
threads_per_block = [768, 512÷2]
blocks_per_grid = cld.(num_samples, threads_per_block)
partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, (blocks_per_grid[2], num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (blocks_per_grid[2], num_ants, length(correlator_sample_shifts)))))
shmem_size = sizeof(ComplexF32) * threads_per_block[2] * num_correlators * num_ants

####### STRIDED VS MONOLITHIC

@cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_kernel!(
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
code_replica
code_replica_strided = CUDA.zeros(Float32, num_samples + num_of_shifts)
@cuda threads=threads_per_block[1] blocks=1 gen_code_replica_strided_kernel!(
    code_replica_strided,
    codes, # texture memory codes
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length
)
code_replica_strided
@test Array(code_replica_strided) == Array(code_replica) 

@btime CUDA.@sync @cuda threads=768 blocks=$blocks_per_grid[1] gen_code_replica_kernel!(
    $code_replica,
    $codes, # texture memory codes
    $code_frequency,
    $sampling_frequency,
    $start_code_phase,
    $prn,
    $num_samples,
    $num_of_shifts,
    $code_length
)
@btime CUDA.@sync @cuda threads=768 blocks=num_of_multiprocessors*1 gen_code_replica_strided_kernel!(
    $code_replica_strided,
    $codes, # texture memory codes
    $code_frequency,
    $sampling_frequency,
    $start_code_phase,
    $prn,
    $num_samples,
    $num_of_shifts,
    $code_length
)

#launch config
kernel = @cuda launch=false gen_code_replica_strided_kernel!(
    code_replica_strided,
    codes, # texture memory codes
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length
)
blocks, threads = launch_configuration(kernel.fun)
@btime CUDA.@sync @cuda threads=$threads blocks=blocks gen_code_replica_strided_kernel!(
    $code_replica_strided,
    $codes, # texture memory codes
    $code_frequency,
    $sampling_frequency,
    $start_code_phase,
    $prn,
    $num_samples,
    $num_of_shifts,
    $code_length
)
######## TEXTURE MEMORY SHENANIGANS

@cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_texture_mem_kernel!(
    code_replica_text_mem,
    codes_text_mem, # texture memory codes
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prn,
    num_samples,
    num_of_shifts,
    code_length
)
@cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_kernel!(
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
Tracking.gen_code_replica!(code_replica_cpu, system, code_frequency, sampling_frequency, start_code_phase, 1, num_samples, correlator_sample_shifts, prn)

[Array(code_replica) Array(code_replica_text_mem) code_replica_cpu]

@test Array(code_replica_text_mem) == Array(code_replica)
@test code_replica_cpu == Array(code_replica_text_mem) == Array(code_replica)

@btime CUDA.@sync @cuda threads=$threads_per_block[1] blocks=$blocks_per_grid[1] $gen_code_replica_texture_mem_kernel!(
    $code_replica_text_mem,
    $codes_text_mem, # texture memory codes
    $code_frequency,
    $sampling_frequency,
    $start_code_phase,
    $prn,
    $num_samples,
    $num_of_shifts,
    $code_length
)

@btime CUDA.@sync @cuda threads=$threads_per_block[1] blocks=$blocks_per_grid[1] $gen_code_replica_kernel!(
    $code_replica,
    $codes, # texture memory codes
    $code_frequency,
    $sampling_frequency,
    $start_code_phase,
    $prn,
    $num_samples,
    $num_of_shifts,
    $code_length
)

@btime CUDA.@sync @cuda threads=$threads_per_block[1] blocks=$blocks_per_grid[1] $gen_code_replica_kernel!(
    $code_replica_text_mem,
    $codes_text_mem_simple, # texture memory codes
    $code_frequency,
    $sampling_frequency,
    $start_code_phase,
    $prn,
    $num_samples,
    $num_of_shifts,
    $code_length
)


@cuda threads=threads blocks=blocks gen_code_replica_kernel!(
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

kernel = @cuda launch=false gen_code_replica_strided_kernel!(
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
blocks, threads = launch_configuration(kernel.fun)
@cuda threads=threads blocks=blocks kernel(
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
