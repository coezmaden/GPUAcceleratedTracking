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

@testset "Kernel Algorithm #2" begin
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
    code_replica = CUDA.zeros(Float32, num_samples + num_of_shifts)
    carrier_replica = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples), CUDA.zeros(Float32, num_samples)))
    downconverted_signal = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples, num_ants), CUDA.zeros(Float32, num_samples, num_ants)))
    threads_per_block = [1024, 512]
    blocks_per_grid = cld.(num_samples, threads_per_block)
    partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, (blocks_per_grid[2], num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (blocks_per_grid[2], num_ants, length(correlator_sample_shifts)))))
    shmem_size = sizeof(ComplexF32) * threads_per_block[2] * num_correlators * num_ants
    @cuda threads=1024 blocks=6 gen_code_replica_strided_kernel!(
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
    accumulators = vec(sum(Array(partial_sum), dims=1))
    accumulators_true = ComplexF32.([1476.0f0; 2500.0f0; 1476.0f0])
    @test accumulators ≈ accumulators_true
end


@tetsset "Downconvert Kernel" begin
    enable_gpu = Val(true)
    num_samples = 2500
    num_ants = 1
    system = GPSL1(use_gpu = enable_gpu)
    codes = system.codes
    code_frequency = get_code_frequency(system)
    code_length = get_code_length(system)
    start_code_phase = 0.0f0
    carrier_phase = 0.0f0
    carrier_frequency = 1500Hz
    prn = 1
    signal, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase)
    carrier_replica = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples), CUDA.zeros(Float32, num_samples)))
    downconverted_signal = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples, num_ants), CUDA.zeros(Float32, num_samples, num_ants)))
    code_replica = CUDA.zeros(Float32, num_samples + 2)
    @cuda threads=1024 blocks=6 gen_code_replica_strided_kernel!(
                code_replica,
                codes,
                code_frequency,
                sampling_frequency,
                start_code_phase,
                prn,
                num_samples,
                2,
                code_length
    )
    kernel = @cuda launch=false downconvert_strided_kernel!(
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
        NumAnts(num_ants)
    )
    blocks, threads = launch_configuration(kernel.fun)
    @cuda threads=threads blocks=blocks downconvert_strided_kernel!(
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
        NumAnts(num_ants)
    ) 
    @test Array(downconverted_signal) ≈ ones(ComplexF32, num_samples) .* Array(code_replica)[2:2501]
end