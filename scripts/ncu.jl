# nvprof --profile-from-start off julia /path/to/ncu.jl
# or
# ncu --mode=launch julia /path/to/ncu.jl
using GPUAcceleratedTracking, CUDA, Tracking, GNSSSignals, StructArrays;
import Tracking: Hz, ms;
enable_gpu = Val(true)
num_samples = 50000
sampling_frequency = num_samples / 1ms |> Hz
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
prn = 1
correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]
code_replica = CUDA.zeros(Float32, num_samples + num_of_shifts)
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
blocks_per_grid, threads_per_block = launch_configuration(code_replica_kernel.fun)
CUDA.@profile @cuda threads=threads_per_block blocks=blocks_per_grid gen_code_replica_texture_mem_strided_kernel!(
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