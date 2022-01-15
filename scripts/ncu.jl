# nvprof --profile-from-start off julia /path/to/ncu.jl
# or
# ncu --mode=launch julia /path/to/ncu.jl
using CUDA
CUDA.version()
using GPUAcceleratedTracking, CUDA, Tracking, GNSSSignals, StructArrays
import Tracking: Hz, ms

system = GPSL1(use_gpu = Val(true));

## 1 ms signal track
# signal, sampling_frequency = gen_signal(system, 1, 1500Hz, 2500, num_ants = NumAnts(4))
# state = TrackingState(1, system, 1500Hz, 0, num_samples = 2500, num_ants = NumAnts(4))
# track(signal, state, sampling_frequency)
# CUDA.@profile track(signal, state, sampling_frequency)

## 1 ms signal downconvert and correlate
# Generate GNSS object and signal information
enable_gpu = Val(true);
num_samples = 50000;
num_ants = 1;
num_correlators = 3;
algorithm = KernelAlgorithm(5);

system = GPSL1(use_gpu = enable_gpu);
codes = CuTexture(CuTextureArray(system.codes));
code_frequency = get_code_frequency(system);
code_length = get_code_length(system);
start_code_phase = 0.0f0;
carrier_phase = 0.0f0;
carrier_frequency = 1500Hz;
prn = 1;

# Generate the signal;
signal, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase);

# Generate correlator;
correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators));
correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5);
num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1];

# Generate blank code and carrier replica, and downconverted signal;
code_replica = CUDA.zeros(Float32, num_samples + num_of_shifts);
carrier_replica = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples), CUDA.zeros(Float32, num_samples)));
downconverted_signal = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples, num_ants), CUDA.zeros(Float32, num_samples, num_ants)));

# Generate CUDA kernel tuning parameters;
threads_per_block = [1024, 512÷2];
blocks_per_grid = cld.(num_samples, threads_per_block);
partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, (blocks_per_grid[2], num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (blocks_per_grid[2], num_ants, length(correlator_sample_shifts)))));
shmem_size = sizeof(ComplexF32) * threads_per_block[2] * num_correlators * num_ants;

CUDA.@profile @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_kernel!(
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