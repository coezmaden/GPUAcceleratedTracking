@testset "Kernel Algorithm #1" begin
    enable_gpu = Val(true)
    num_samples = 2500
    num_ants = 1
    num_correlators = 3
    system = GPSL1(use_gpu = enable_gpu)
    codes = system.codes
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
    block_dim_z = num_correlators
    block_dim_y = num_ants
    # keep num_corrs and num_ants in seperate dimensions, truncate num_samples accordingly to fit
    block_dim_x = prevpow(2, 512 ÷ block_dim_y ÷ block_dim_z)
    threads_per_block = (block_dim_x, block_dim_y, block_dim_z)
    blocks_per_grid = cld(num_samples, block_dim_x)
    partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z), CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z)))
    shmem_size = sizeof(ComplexF32) * block_dim_x * block_dim_y * block_dim_z
    @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size downconvert_and_correlate_kernel_1!(
        partial_sum.re,
        partial_sum.im,
        signal.re,
        signal.im,
        codes,
        code_frequency,
        correlator_sample_shifts,
        carrier_frequency,
        sampling_frequency,
        start_code_phase,
        carrier_phase,
        code_length,
        prn,
        num_samples,
        num_ants,
        num_correlators
    )
    accumulators = vec(sum(Array(partial_sum), dims=1))
    accumulators_true = ComplexF32.([1476.0f0; 2500.0f0; 1476.0f0])
    @test accumulators ≈ accumulators_true
end