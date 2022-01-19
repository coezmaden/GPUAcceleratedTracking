@testset "Kernel Algorithm 1_3_cplx_multi" begin
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
    threads_per_block = [(block_dim_x, block_dim_y, block_dim_z), 512]
    blocks_per_grid = cld(num_samples, block_dim_x)
    partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z), CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z)))
    shmem_size = [sizeof(ComplexF32) * block_dim_x * block_dim_y * block_dim_z
                sizeof(ComplexF32) * 512 * num_ants * num_correlators]
    algorithm = KernelAlgorithm(1330)
    
    # @cuda threads=threads_per_block[1] blocks=blocks_per_grid shmem=shmem_size downconvert_and_correlate_kernel_1!(
    #     partial_sum.re,
    #     partial_sum.im,
    #     signal.re,
    #     signal.im,
    #     codes,
    #     code_frequency,
    #     correlator_sample_shifts,
    #     carrier_frequency,
    #     sampling_frequency,
    #     start_code_phase,
    #     carrier_phase,
    #     code_length,
    #     prn,
    #     num_samples,
    #     num_ants,
    #     num_correlators
    # )
    # @cuda threads=512 blocks=1 shmem=sizeof(ComplexF32)*512*num_ants*num_correlators reduce_cplx_multi_3(
    #     partial_sum.re,
    #     partial_sum.im,
    #     partial_sum.re,
    #     partial_sum.im,
    #     blocks_per_grid,
    #     NumAnts(num_ants),
    #     correlator_sample_shifts
    # )
    kernel_algorithm(
        threads_per_block,
        blocks_per_grid,
        shmem_size,
        nothing,
        codes,
        code_frequency,
        sampling_frequency,
        start_code_phase,
        prn,
        num_samples,
        num_of_shifts,
        code_length,
        partial_sum,
        nothing,
        nothing,
        nothing,
        nothing,
        signal.re,
        signal.im,
        correlator_sample_shifts,
        carrier_frequency,
        carrier_phase,
        NumAnts(num_ants),
        nothing,
        algorithm
    )    
    CUDA.@allowscalar begin 
        accumulators = vec(Array(partial_sum))
        accumulators_true = ComplexF32.([1476.0f0; 2500.0f0; 1476.0f0])
        @test accumulators ≈ accumulators_true
    end
end

@testset "Kernel Algorithm 1_3_cplx_multi_textmem" begin
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
    block_dim_z = num_correlators
    block_dim_y = num_ants
    # keep num_corrs and num_ants in seperate dimensions, truncate num_samples accordingly to fit
    block_dim_x = prevpow(2, 512 ÷ block_dim_y ÷ block_dim_z)
    threads_per_block = [(block_dim_x, block_dim_y, block_dim_z), 512]
    blocks_per_grid = cld(num_samples, block_dim_x)
    partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z), CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z)))
    shmem_size = [sizeof(ComplexF32) * block_dim_x * block_dim_y * block_dim_z
                sizeof(ComplexF32) * 512 * num_ants * num_correlators]
    algorithm = KernelAlgorithm(1331)
    
    # @cuda threads=threads_per_block[1] blocks=blocks_per_grid shmem=shmem_size[1] downconvert_and_correlate_kernel_1331!(
    #     partial_sum.re,
    #     partial_sum.im,
    #     signal.re,
    #     signal.im,
    #     codes,
    #     code_frequency,
    #     correlator_sample_shifts,
    #     carrier_frequency,
    #     sampling_frequency,
    #     start_code_phase,
    #     carrier_phase,
    #     code_length,
    #     prn,
    #     num_samples,
    #     num_ants,
    #     num_correlators
    # )
    # # reduction_kernel = @cuda launch=false reduce_cplx_multi_3(
    # #     partial_sum.re,
    # #     partial_sum.im,
    # #     partial_sum.re,
    # #     partial_sum.im,
    # #     blocks_per_grid,
    # #     NumAnts(num_ants),
    # #     correlator_sample_shifts
    # # )
    # # threads, blocks = launch_configuration(reduction_kernel.fun)
    # @cuda threads=1024 blocks=1 shmem=sizeof(ComplexF32)*1024*num_ants*num_correlators reduce_cplx_multi_3(
    #     partial_sum.re,
    #     partial_sum.im,
    #     partial_sum.re,
    #     partial_sum.im,
    #     blocks_per_grid,
    #     NumAnts(num_ants),
    #     correlator_sample_shifts
    # )
    kernel_algorithm(
        threads_per_block,
        blocks_per_grid,
        shmem_size,
        nothing,
        codes,
        code_frequency,
        sampling_frequency,
        start_code_phase,
        prn,
        num_samples,
        num_of_shifts,
        code_length,
        partial_sum,
        nothing,
        nothing,
        nothing,
        nothing,
        signal.re,
        signal.im,
        correlator_sample_shifts,
        carrier_frequency,
        carrier_phase,
        NumAnts(num_ants),
        nothing,
        algorithm
    )    
    CUDA.@allowscalar begin 
        accumulators = Array(partial_sum)[1 ,: ,:]
        accumulators_true = ComplexF32.([1476.0f0 2500.0f0 1476.0f0])
        @test skip=true accumulators[1] ≈ accumulators_true[1] #text mem fail
        @test accumulators[2] ≈ accumulators_true[2]
        @test accumulators[3] ≈ accumulators_true[3]
        @test accumulators ≈ accumulators_true
    end
end

@testset "Kernel Algorithm 1_4_cplx_multi_textmem" begin
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
    block_dim_z = num_correlators
    block_dim_y = num_ants
    # keep num_corrs and num_ants in seperate dimensions, truncate num_samples accordingly to fit
    block_dim_x = prevpow(2, 512 ÷ block_dim_y ÷ block_dim_z)
    threads_per_block = [(block_dim_x, block_dim_y, block_dim_z), 1024]
    blocks_per_grid = cld(num_samples, block_dim_x) ÷ 2 # launch with half the grid
    partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z), CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z)))
    shmem_size = [sizeof(ComplexF32) * block_dim_x * block_dim_y * block_dim_z
                sizeof(ComplexF32) * threads_per_block[2] * num_ants * num_correlators]
    algorithm = KernelAlgorithm(1431)
    
    # @cuda threads=threads_per_block[1] blocks=blocks_per_grid shmem=shmem_size[1] downconvert_and_correlate_kernel_1431!(
    #     partial_sum.re,
    #     partial_sum.im,
    #     signal.re,
    #     signal.im,
    #     codes,
    #     code_frequency,
    #     correlator_sample_shifts,
    #     carrier_frequency,
    #     sampling_frequency,
    #     start_code_phase,
    #     carrier_phase,
    #     code_length,
    #     prn,
    #     num_samples,
    #     num_ants,
    #     num_correlators
    # )
    # # reduction_kernel = @cuda launch=false reduce_cplx_multi_3(
    # #     partial_sum.re,
    # #     partial_sum.im,
    # #     partial_sum.re,
    # #     partial_sum.im,
    # #     blocks_per_grid,
    # #     NumAnts(num_ants),
    # #     correlator_sample_shifts
    # # )
    # # threads, blocks = launch_configuration(reduction_kernel.fun)
    # @cuda threads=threads_per_block[2] blocks=1 shmem=shmem_size[2] reduce_cplx_multi_3(
    #     partial_sum.re,
    #     partial_sum.im,
    #     partial_sum.re,
    #     partial_sum.im,
    #     blocks_per_grid,
    #     NumAnts(num_ants),
    #     correlator_sample_shifts
    # )
    kernel_algorithm(
        threads_per_block,
        blocks_per_grid,
        shmem_size,
        nothing,
        codes,
        code_frequency,
        sampling_frequency,
        start_code_phase,
        prn,
        num_samples,
        num_of_shifts,
        code_length,
        partial_sum,
        nothing,
        nothing,
        nothing,
        nothing,
        signal.re,
        signal.im,
        correlator_sample_shifts,
        carrier_frequency,
        carrier_phase,
        NumAnts(num_ants),
        nothing,
        algorithm
    )    
    CUDA.@allowscalar begin 
        accumulators = Array(partial_sum)[1 ,: ,:]
        accumulators_true = ComplexF32.([1476.0f0 2500.0f0 1476.0f0])
        @test accumulators[1] ≈ accumulators_true[1]
        @test accumulators[2] ≈ accumulators_true[2]
        @test accumulators[3] ≈ accumulators_true[3]
        @test accumulators ≈ accumulators_true
    end
end

@testset "Kernel Algorithm 2_3_cplx_multi" begin
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
    threads_per_block = zeros(Int, 3)
    blocks_per_grid = zeros(Int, 3)
    accum = StructArray{ComplexF32}((CUDA.zeros(Float32, (num_samples, num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (num_samples, num_ants, length(correlator_sample_shifts)))))
    threads_per_block[3] = 1024
    blocks_per_grid[3] = cld(num_samples, threads_per_block[3])
    phi = StructArray{ComplexF32}((CUDA.zeros(Float32, (blocks_per_grid[3], num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (blocks_per_grid[3], num_ants, length(correlator_sample_shifts)))))
    shmem_size = sizeof(ComplexF32) * 1024 * num_correlators * num_ants
    code_replica_kernel = @cuda launch=false gen_code_replica_strided_kernel!(
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
    algorithm = KernelAlgorithm(2330)
    # @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_strided_kernel!(
    #     code_replica,
    #     codes,
    #     code_frequency,
    #     sampling_frequency,
    #     start_code_phase,
    #     prn,
    #     num_samples,
    #     num_of_shifts,
    #     code_length
    # )
    # @cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] downconvert_and_accumulate_strided_kernel!(
    #     accum.re,
    #     accum.im,
    #     code_replica,
    #     carrier_replica.re,
    #     carrier_replica.im,
    #     downconverted_signal.re,
    #     downconverted_signal.im,
    #     signal.re,
    #     signal.im,
    #     carrier_frequency,
    #     sampling_frequency,
    #     carrier_phase,
    #     num_samples,
    #     NumAnts(num_ants),
    #     correlator_sample_shifts
    # )
    # @cuda threads=threads_per_block[3] blocks=blocks_per_grid[3] shmem=shmem_size reduce_cplx_multi_3(
    #     sum.re,
    #     sum.im,
    #     accum.re,
    #     accum.im,
    #     num_samples,
    #     NumAnts(num_ants),
    #     correlator_sample_shifts
    # )
    # @cuda threads=threads_per_block[3] blocks=1 shmem=shmem_size reduce_cplx_multi_3(
    #     sum.re,
    #     sum.im,
    #     sum.re,
    #     sum.im,
    #     blocks_per_grid[3],
    #     NumAnts(num_ants),
    #     correlator_sample_shifts
    # )
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
    CUDA.@allowscalar begin 
        accumulators = Array(phi)[1, :, :]
        accumulators_true = ComplexF32.([1476.0f0 2500.0f0 1476.0f0])
        @test accumulators ≈ accumulators_true
    end
end

@testset "Kernel Algorithm 2_3_cplx_multi_textmem" begin
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
    algorithm = KernelAlgorithm(2331)
    @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_texture_mem_strided_kernel!(
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
    @cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] downconvert_and_accumulate_strided_kernel!(
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
    @cuda threads=threads_per_block[3] blocks=blocks_per_grid[3] shmem=shmem_size reduce_cplx_multi_3(
        phi.re,
        phi.im,
        accum.re,
        accum.im,
        num_samples,
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    Array(phi)
    @cuda threads=threads_per_block[3] blocks=1 shmem=shmem_size reduce_cplx_multi_3(
        phi.re,
        phi.im,
        phi.re,
        phi.im,
        blocks_per_grid[3],
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    Array(phi)
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
    CUDA.@allowscalar begin 
        accumulators = Array(phi)[1, :, :]
        accumulators_true = ComplexF32.([1476.0f0 2500.0f0 1476.0f0])
        @test accumulators ≈ accumulators_true
    end
end

@testset "Kernel Algorithm 2_4_cplx_multi" begin
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
    threads_per_block = zeros(Int, 3)
    blocks_per_grid = zeros(Int, 3)
    accum = StructArray{ComplexF32}((CUDA.zeros(Float32, (num_samples, num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (num_samples, num_ants, length(correlator_sample_shifts)))))
    threads_per_block[3] = 1024
    blocks_per_grid[3] = cld(num_samples, threads_per_block[3])
    phi = StructArray{ComplexF32}((CUDA.zeros(Float32, (blocks_per_grid[3], num_ants, length(correlator_sample_shifts))),CUDA.zeros(Float32, (blocks_per_grid[3], num_ants, length(correlator_sample_shifts)))))
    shmem_size = sizeof(ComplexF32) * 1024 * num_correlators * num_ants
    code_replica_kernel = @cuda launch=false gen_code_replica_strided_kernel!(
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
    algorithm = KernelAlgorithm(2331)
    @cuda threads=threads_per_block[1] blocks=blocks_per_grid[1] gen_code_replica_strided_kernel!(
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
    @cuda threads=threads_per_block[2] blocks=blocks_per_grid[2] downconvert_and_accumulate_strided_kernel!(
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
    @cuda threads=threads_per_block[3] blocks=cld(blocks_per_grid[3], 2) shmem=shmem_size reduce_cplx_multi_4(
        phi.re,
        phi.im,
        accum.re,
        accum.im,
        num_samples,
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    @cuda threads=threads_per_block[3] blocks=1 shmem=shmem_size reduce_cplx_multi_4(
        phi.re,
        phi.im,
        phi.re,
        phi.im,
        blocks_per_grid[3],
        NumAnts(num_ants),
        correlator_sample_shifts
    )
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
    CUDA.@allowscalar begin 
        accumulators = Array(phi)[1, :, :]
        accumulators_true = ComplexF32.([1476.0f0 2500.0f0 1476.0f0])
        @test accumulators ≈ accumulators_true
    end
end

# @testset "Kernel Algorithm 2_4_cplx_multi_textmem" begin
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
    @benchmark CUDA.@sync @cuda threads=$threads_per_block[1] blocks=$blocks_per_grid[1] $gen_code_replica_texture_mem_strided_kernel!(
        $code_replica,
        $codes,
        $code_frequency,
        $sampling_frequency,
        $start_code_phase,
        $prn,
        $num_samples,
        $num_of_shifts,
        $code_length
    )
    @benchmark CUDA.@sync @cuda threads=$threads_per_block[2] blocks=$blocks_per_grid[2] $downconvert_and_accumulate_strided_kernel!(
        $accum.re,
        $accum.im,
        $code_replica,
        $carrier_replica.re,
        $carrier_replica.im,
        $downconverted_signal.re,
        $downconverted_signal.im,
        $signal.re,
        $signal.im,
        $carrier_frequency,
        $sampling_frequency,
        $carrier_phase,
        $num_samples,
        $NumAnts(num_ants),
        $correlator_sample_shifts
    )
    @benchmark CUDA.@sync @cuda threads=$threads_per_block[3] blocks=$cld(blocks_per_grid[3], 2) shmem=$shmem_size reduce_cplx_multi_4(
        $phi.re,
        $phi.im,
        $accum.re,
        $accum.im,
        $num_samples,
        $NumAnts(num_ants),
        $correlator_sample_shifts
    )
    # Array(phi)
    @cuda threads=threads_per_block[3] blocks=1 shmem=shmem_size reduce_cplx_multi_4(
        phi.re,
        phi.im,
        phi.re,
        phi.im,
        blocks_per_grid[3],
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    # Array(phi)
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
    CUDA.@allowscalar begin 
        accumulators = Array(phi)[1, :, :]
        accumulators_true = ComplexF32.([1476.0f0 2500.0f0 1476.0f0])
        @test accumulators ≈ accumulators_true
    end
end

@testset "Downconvert Kernel" begin
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
    @cuda threads=512 blocks=6 gen_code_replica_strided_kernel!(
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
    CUDA.@allowscalar begin
        @test Array(downconverted_signal) ≈ ones(ComplexF32, num_samples) .* Array(code_replica)[2:2501]
    end
end

@testset "Downconvert and Accumulate Kernel" begin
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
    accum = StructArray{ComplexF32}(
        (
        CUDA.zeros(Float32, (num_samples, num_ants, num_correlators)),
        CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    code_replica = CUDA.zeros(Float32, num_samples + 2)
    @cuda threads=512 blocks=6 gen_code_replica_strided_kernel!(
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
    kernel = @cuda launch=false downconvert_and_accumulate_strided_kernel!(
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
    blocks, threads = launch_configuration(kernel.fun)
    @cuda threads=threads blocks=blocks downconvert_and_accumulate_strided_kernel!(
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
    CUDA.@allowscalar begin
        ϕ_hat = vec(sum(Array(accum), dims=1))
        ϕ = ComplexF32.([1476.0f0; 2500.0f0; 1476.0f0])
        @test Array(accum)[:, :, 2] ≈ ones(ComplexF32, num_samples)
        @test ϕ_hat ≈ ϕ
    end
    # @benchmark CUDA.@sync @cuda threads=$threads blocks=$blocks $downconvert_and_accumulate_strided_kernel!(
    #     $accum.re,
    #     $accum.im,
    #     $code_replica,
    #     $carrier_replica.re,
    #     $carrier_replica.im,
    #     $downconverted_signal.re,
    #     $downconverted_signal.im,
    #     $signal.re,
    #     $signal.im,
    #     $carrier_frequency,
    #     $sampling_frequency,
    #     $carrier_phase,
    #     $num_samples,
    #     $NumAnts(num_ants),
    #     $correlator_sample_shifts
    # )
end