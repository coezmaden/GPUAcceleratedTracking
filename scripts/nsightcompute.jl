using GPUAcceleratedTracking, CUDA, Tracking, GNSSSignals, StructArrays;
import Tracking: Hz, ms;
enable_gpu = Val(true);
num_samples = 2048;
num_ants = 4;
num_correlators = 3;
system = GPSL1(use_gpu = enable_gpu);
codes = system.codes;
#convert to text_mem;
codes = CuTexture(
    CuTextureArray(codes),
    address_mode = CUDA.ADDRESS_MODE_WRAP,
    interpolation = CUDA.NearestNeighbour(),
    normalized_coordinates = true
);
code_frequency = get_code_frequency(system);
code_length = get_code_length(system);
start_code_phase = 0.0f0;
carrier_phase = 0.0f0;
carrier_frequency = 1500Hz;
prns = collect(1:4);
num_sats = length(prns);
signal, sampling_frequency = gen_signal(system, prns, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase);
correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators));
correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5);
num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1];
code_replica = CUDA.zeros(Float32, (num_samples + num_of_shifts, num_sats));
prnlist_gpu = CuArray(prns);
code_replica_kernel = @cuda launch=false gen_code_replica_texture_mem_strided_nsat_kernel!(
    code_replica,
    codes,
    code_frequency,
    sampling_frequency,
    start_code_phase,
    prnlist_gpu,
    num_samples,
    num_of_shifts,
    code_length
);
blocks_per_grid_1, threads_per_block_1 = launch_configuration(code_replica_kernel.fun);
correlator_sample_shifts_unrolled = CuArray(correlator_sample_shifts .- correlator_sample_shifts[end]);
threads_per_block_y = num_ants;
# threads_per_block_x = CUDA.attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) ÷ threads_per_block_y;
threads_per_block_x = 256 ÷ threads_per_block_y;
blocks_per_grid_x = cld(cld(num_samples, threads_per_block_x), 2);
blocks_per_grid_y = num_correlators;
blocks_per_grid_z = num_sats;
threads_per_block = (threads_per_block_x, threads_per_block_y);
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z);
shmem_size = sizeof(ComplexF32) * threads_per_block_x * threads_per_block_y;
accum = StructArray{ComplexF32}(
    (
        CUDA.zeros(Float32, (num_ants, num_correlators, num_sats)),
        CUDA.zeros(Float32, (num_ants, num_correlators, num_sats))
    )
);
#warmup
@cuda threads=threads_per_block_1 blocks=(blocks_per_grid_1, num_sats) gen_code_replica_texture_mem_strided_nsat_kernel!(
        code_replica,
        codes,
        code_frequency,
        sampling_frequency,
        start_code_phase,
        prnlist_gpu,
        num_samples,
        num_of_shifts,
        code_length
    )
@cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size downconvert_and_correlate_kernel_3d_4431!(
        accum.re,
        accum.im,
        signal.re,
        signal.im,
        code_replica,
        correlator_sample_shifts_unrolled,
        carrier_frequency,
        sampling_frequency,
        carrier_phase,
        num_samples,
        NumAnts(num_ants)
    )
CUDA.@profile begin
    NVTX.@range "gen_code_replica_texture_mem_strided_nsat_kernel!" begin
        @cuda threads=threads_per_block_1 blocks=(blocks_per_grid_1, num_sats) gen_code_replica_texture_mem_strided_nsat_kernel!(
            code_replica,
            codes,
            code_frequency,
            sampling_frequency,
            start_code_phase,
            prnlist_gpu,
            num_samples,
            num_of_shifts,
            code_length
        )
    end
    NVTX.@range "Direct CUDA.@elapsed" begin
        CUDA.@elapsed @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size downconvert_and_correlate_kernel_3d_4431!(
            accum.re,
            accum.im,
            signal.re,
            signal.im,
            code_replica,
            correlator_sample_shifts_unrolled,
            carrier_frequency,
            sampling_frequency,
            carrier_phase,
            num_samples,
            NumAnts(num_ants)
        )
    end
    NVTX.@range "Function CUDA.@elapsed" begin
        bench(
            threads_per_block,
            blocks_per_grid,
            accum,
            signal,
            code_replica,
            correlator_sample_shifts_unrolled,
            carrier_frequency,
            sampling_frequency,
            carrier_phase,
            num_samples,
            NumAnts(num_ants)
        )
    end
    NVTX.@range "downconvert_and_correlate_kernel_3d_4431!" begin
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size downconvert_and_correlate_kernel_3d_4431!(
            accum.re,
            accum.im,
            signal.re,
            signal.im,
            code_replica,
            correlator_sample_shifts_unrolled,
            carrier_frequency,
            sampling_frequency,
            carrier_phase,
            num_samples,
            NumAnts(num_ants)
        )
    end
end

# CUDA.@profile @cuda threads=threads_per_block_1 blocks=(blocks_per_grid_1, num_sats) gen_code_replica_texture_mem_strided_nsat_kernel!(
#     code_replica,
#     codes,
#     code_frequency,
#     sampling_frequency,
#     start_code_phase,
#     prnlist_gpu,
#     num_samples,
#     num_of_shifts,
#     code_length
# )
# @benchmark CUDA.@sync @cuda threads=$threads_per_block_1 blocks=$(blocks_per_grid_1, num_sats) $gen_code_replica_texture_mem_strided_nsat_kernel!(
#     $code_replica,
#     $codes,
#     $code_frequency,
#     $sampling_frequency,
#     $start_code_phase,
#     $prnlist_gpu,
#     $num_samples,
#     $num_of_shifts,
#     $code_length
# )
# CUDA.@profile @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size downconvert_and_correlate_kernel_3d_4431!(
#     accum.re,
#     accum.im,
#     signal.re,
#     signal.im,
#     code_replica,
#     correlator_sample_shifts_unrolled,
#     carrier_frequency,
#     sampling_frequency,
#     carrier_phase,
#     num_samples,
#     NumAnts(num_ants)
# )