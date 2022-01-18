function add_results!(benchmark_results_w_params, benchmark_results)
    benchmark_results_w_params["TrialObj"] = benchmark_results
    benchmark_results_w_params["RawTimes"] = benchmark_results.times
    benchmark_results_w_params["Minimum"] = minimum(benchmark_results).time
    benchmark_results_w_params["Median"] = median(benchmark_results).time
    benchmark_results_w_params["Mean"] = mean(benchmark_results).time
    benchmark_results_w_params["σ"] = std(benchmark_results).time
    benchmark_results_w_params["Maximum"] = maximum(benchmark_results).time
end

function add_metadata!(benchmark_results_w_params, processor, algorithm::KernelAlgorithm{ALGN}) where ALGN
    # Get OS info
    os_name = @static Sys.iswindows() ? "windows" : (@static Sys.isapple() ? "macos" : @static Sys.islinux() ? "linux" : @static Sys.isunix() ? "generic_unix" : throw("Can't determine OS name"))
    
    # Get CPU info
    cpu_name = Sys.cpu_info()[1].model
    # Workaround for NVIDIA Jetson CPU
    cpu_name == "unknown" ? "NVIDIA ARMv8" : cpu_name

    # Get GPU name if GPU is selected, if not use cpu_name
    processor_name = processor == "GPU" ? name(CUDA.CuDevice(0)) : cpu_name

    # Add metadata to the results
    benchmark_results_w_params["os"] = os_name
    benchmark_results_w_params[processor * " model"] = processor_name
    benchmark_results_w_params["algorithm"] = ALGN
end

# CPU Benchmark
function _run_kernel_benchmark(
    gnss,
    enable_gpu::Val{false},
    num_samples,
    num_ants,
    num_correlators,
    algorithm
)
    system = gnss(use_gpu = enable_gpu)
    start_code_phase = 0.0
    carrier_phase = 0.0
    carrier_frequency = 1500Hz
    prn = 1
    code_frequency = get_code_frequency(system)

    signal, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants=NumAnts(num_ants))
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), num_correlators)
    correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
    
    state = TrackingState(prn, system, carrier_frequency, start_code_phase, num_samples=num_samples, num_ants=NumAnts(num_ants), correlator=correlator)

    code_replica = Tracking.get_code(state)
    Tracking.resize!(code_replica, num_samples + correlator_sample_shifts[end] - correlator_sample_shifts[1])
    carrier_replica = Tracking.get_carrier(state)
    Tracking.resize!(Tracking.choose(carrier_replica, signal), num_samples)
    downconverted_signal_temp = Tracking.get_downconverted_signal(state)
    downconverted_signal = Tracking.resize!(downconverted_signal_temp, size(signal, 1), signal)

    @benchmark Tracking.downconvert_and_correlate!(
        $system,
        $signal,
        $correlator,
        $code_replica,
        $start_code_phase,
        $carrier_replica,
        $carrier_phase,
        $downconverted_signal,
        $code_frequency,
        $correlator_sample_shifts,
        $carrier_frequency,
        $sampling_frequency,
        1,
        $num_samples,
        $prn
    )
end

# GPU Kernel Benchmark for KernelAlgorithm 1_3_cplx_multi
function _run_kernel_benchmark(
    gnss,
    enable_gpu::Val{true},
    num_samples,
    num_ants,
    num_correlators,
    algorithm::KernelAlgorithm{1330}
)   
    # Generate GNSS object and signal information
    system = gnss(use_gpu = enable_gpu)
    codes = system.codes
    code_frequency = get_code_frequency(system)
    code_length = get_code_length(system)
    start_code_phase = 0.0f0
    carrier_phase = 0.0f0
    carrier_frequency = 1500Hz
    prn = 1

    # Generate the signal
    signal, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase)
    
    # Generate correlator
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
    num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]

    # Generate CUDA kernel tuning parameters
    block_dim_z = num_correlators
    block_dim_y = num_ants
    # keep num_corrs and num_ants in seperate dimensions, truncate num_samples accordingly to fit
    block_dim_x = prevpow(2, 512 ÷ block_dim_y ÷ block_dim_z)
    threads_per_block = [(block_dim_x, block_dim_y, block_dim_z), 512]
    blocks_per_grid = cld(num_samples, block_dim_x)
    partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z), CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z)))
    shmem_size = [sizeof(ComplexF32) * block_dim_x * block_dim_y * block_dim_z
                sizeof(ComplexF32) * 1024 * num_ants * num_correlators]
    Num_Ants = NumAnts(num_ants)
    @benchmark CUDA.@sync $kernel_algorithm(
        $threads_per_block,
        $blocks_per_grid,
        $shmem_size,
        nothing,
        $codes,
        $code_frequency,
        $sampling_frequency,
        $start_code_phase,
        $prn,
        $num_samples,
        $num_of_shifts,
        $code_length,
        $partial_sum,
        nothing,
        nothing,
        nothing,
        nothing,
        $signal.re,
        $signal.im,
        $correlator_sample_shifts,
        $carrier_frequency,
        $carrier_phase,
        $Num_Ants,
        nothing,
        $algorithm
    )
end

# GPU Kernel Benchmark for KernelAlgorithm 1_3_cplx_multi_texmem
function _run_kernel_benchmark(
    gnss,
    enable_gpu::Val{true},
    num_samples,
    num_ants,
    num_correlators,
    algorithm::KernelAlgorithm{1331}
)   
    # Generate GNSS object and signal information
    system = gnss(use_gpu = enable_gpu)
    codes = system.codes
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

    # Generate the signal
    signal, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase)
    
    # Generate correlator
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
    num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]

    # Generate CUDA kernel tuning parameters
    block_dim_z = num_correlators
    block_dim_y = num_ants
    # keep num_corrs and num_ants in seperate dimensions, truncate num_samples accordingly to fit
    block_dim_x = prevpow(2, 512 ÷ block_dim_y ÷ block_dim_z)
    threads_per_block = [(block_dim_x, block_dim_y, block_dim_z), 1024]
    blocks_per_grid = cld(num_samples, block_dim_x)
    partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z), CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z)))
    shmem_size = [sizeof(ComplexF32) * block_dim_x * block_dim_y * block_dim_z
                sizeof(ComplexF32) * 1024 * num_ants * num_correlators]
    Num_Ants = NumAnts(num_ants)
    @benchmark CUDA.@sync $kernel_algorithm(
        $threads_per_block,
        $blocks_per_grid,
        $shmem_size,
        nothing,
        $codes,
        $code_frequency,
        $sampling_frequency,
        $start_code_phase,
        $prn,
        $num_samples,
        $num_of_shifts,
        $code_length,
        $partial_sum,
        nothing,
        nothing,
        nothing,
        nothing,
        $signal.re,
        $signal.im,
        $correlator_sample_shifts,
        $carrier_frequency,
        $carrier_phase,
        $Num_Ants,
        nothing,
        $algorithm
    )
end

# GPU Kernel Benchmark for KernelAlgorithm 1_4_cplx_multi_texmem
function _run_kernel_benchmark(
    gnss,
    enable_gpu::Val{true},
    num_samples,
    num_ants,
    num_correlators,
    algorithm::KernelAlgorithm{1431}
)   
    # Generate GNSS object and signal information
    system = gnss(use_gpu = enable_gpu)
    codes = system.codes
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

    # Generate the signal
    signal, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase)
    
    # Generate correlator
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
    num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]

    # Generate CUDA kernel tuning parameters
    block_dim_z = num_correlators
    block_dim_y = num_ants
    # keep num_corrs and num_ants in seperate dimensions, truncate num_samples accordingly to fit
    block_dim_x = prevpow(2, 512 ÷ block_dim_y ÷ block_dim_z)
    threads_per_block = [(block_dim_x, block_dim_y, block_dim_z), 1024]
    blocks_per_grid = cld(num_samples, block_dim_x)
    partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z), CUDA.zeros(Float32, blocks_per_grid, block_dim_y, block_dim_z)))
    shmem_size = [sizeof(ComplexF32) * block_dim_x * block_dim_y * block_dim_z
                sizeof(ComplexF32) * threads_per_block[2] * num_ants * num_correlators]
    Num_Ants = NumAnts(num_ants)
    @benchmark CUDA.@sync $kernel_algorithm(
        $threads_per_block,
        $blocks_per_grid,
        $shmem_size,
        nothing,
        $codes,
        $code_frequency,
        $sampling_frequency,
        $start_code_phase,
        $prn,
        $num_samples,
        $num_of_shifts,
        $code_length,
        $partial_sum,
        nothing,
        nothing,
        nothing,
        nothing,
        $signal.re,
        $signal.im,
        $correlator_sample_shifts,
        $carrier_frequency,
        $carrier_phase,
        $Num_Ants,
        nothing,
        $algorithm
    )
end

function run_kernel_benchmark(benchmark_params::Dict)
    @unpack GNSS, num_samples, num_ants, num_correlators, processor, algorithm = benchmark_params
    enable_gpu = (processor == "GPU" ? Val(true) : Val(false))
    algorithm = KernelAlgorithm(algorithm)
    benchmark_results = _run_kernel_benchmark(
        GNSSDICT[GNSS], 
        enable_gpu,
        num_samples,
        num_ants,
        num_correlators,
        algorithm
    )
    benchmark_results_w_params = copy(benchmark_params)
    add_results!(benchmark_results_w_params, benchmark_results)
    add_metadata!(benchmark_results_w_params, processor, algorithm)
    return benchmark_results_w_params
end