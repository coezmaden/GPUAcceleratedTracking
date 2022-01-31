function _bench_replica(
    num_samples,
    algorithm_obj::ReplicaAlgorithm{1}
)
    # num_samples = 2_048
    num_ants = 1
    num_correlators = 3
    enable_gpu = Val(true)
    system = GPSL1(use_gpu = Val(true));
    # system_h = GPSL1(use_gpu = Val(false));
    codes = system.codes
    # codes_text_mem_simple = CuTexture(
        # CuTextureArray(codes)
    # )
    # codes_text_mem = CuTexture(
    #     CuTextureArray(codes),
    #     address_mode = CUDA.ADDRESS_MODE_WRAP,
    #     interpolation = CUDA.NearestNeighbour(),
    #     normalized_coordinates = true
    # )
    code_frequency = get_code_frequency(system)
    code_length = get_code_length(system)
    start_code_phase = 0.0f0
    carrier_phase = 0.0f0
    carrier_frequency = 1500Hz
    prn = 1

    # Generate the signal;
    _, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase)

    # Generate correlator;
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
    num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]

    # Generate blank code and carrier replica, and downconverted signal;
    # code_replica_cpu = zeros(Float32, num_samples + num_of_shifts)
    code_replica = CUDA.zeros(Float32, num_samples + num_of_shifts)

    # Generate CUDA kernel tuning parameters;
    threads_per_block = 768
    blocks_per_grid = cld.(num_samples, threads_per_block)

    # @cuda threads=threads_per_block blocks=blocks_per_grid gen_code_replica_kernel!(
    #     code_replica,
    #     codes_text_mem, # texture memory codes
    #     code_frequency,
    #     sampling_frequency,
    #     start_code_phase,
    #     prn,
    #     num_samples + num_of_shifts,
    #     correlator_sample_shifts[1],
    #     code_length
    # )
    @benchmark CUDA.@sync @cuda threads=$threads_per_block blocks=$blocks_per_grid $gen_code_replica_kernel!(
        $code_replica,
        $codes, # texture memory codes
        $code_frequency,
        $sampling_frequency,
        $start_code_phase,
        $prn,
        $num_samples + $num_of_shifts,
        $correlator_sample_shifts[1],
        $code_length
    )
end

function _bench_replica(
    num_samples,
    algorithm_obj::ReplicaAlgorithm{2}
)
    # num_samples = 2_048
    num_ants = 1
    num_correlators = 3
    enable_gpu = Val(true)
    system = GPSL1(use_gpu = Val(true));
    # system_h = GPSL1(use_gpu = Val(false));
    codes = system.codes
    # codes_text_mem_simple = CuTexture(
        # CuTextureArray(codes)
    # )
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
    _, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase)

    # Generate correlator;
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
    num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]

    # Generate blank code and carrier replica, and downconverted signal;
    # code_replica_cpu = zeros(Float32, num_samples + num_of_shifts)
    code_replica = CUDA.zeros(Float32, num_samples + num_of_shifts)

    # Generate CUDA kernel tuning parameters;
    threads_per_block = 768
    blocks_per_grid = cld.(num_samples, threads_per_block)

    # @cuda threads=threads_per_block blocks=blocks_per_grid gen_code_replica_texture_mem_kernel!(
    #     code_replica,
    #     codes_text_mem, # texture memory codes
    #     code_frequency,
    #     sampling_frequency,
    #     start_code_phase,
    #     prn,
    #     num_samples + num_of_shifts,
    #     correlator_sample_shifts[1],
    #     code_length
    # )
    @benchmark CUDA.@sync @cuda threads=$threads_per_block blocks=$blocks_per_grid $gen_code_replica_texture_mem_kernel!(
        $code_replica,
        $codes_text_mem, # texture memory codes
        $code_frequency,
        $sampling_frequency,
        $start_code_phase,
        $prn,
        $num_samples + $num_of_shifts,
        $correlator_sample_shifts[1],
        $code_length
    )
end



function run_replica_benchmark(benchmark_params::Dict)
    @unpack num_samples, algorithm = benchmark_params
    algorithm_obj = MEMDICT[algorithm]
    benchmark_results = _bench_replica(num_samples, algorithm_obj)
    benchmark_results_w_params = copy(benchmark_params)
    benchmark_results_w_params["Minimum"] = minimum(benchmark_results).time
    benchmark_results_w_params["Mean"] = mean(benchmark_results).time
    benchmark_results_w_params["Median"] = median(benchmark_results).time
    benchmark_results_w_params["Std"] = std(benchmark_results).time
    return benchmark_results_w_params
end