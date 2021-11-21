function _run_track_benchmark(
    gnss,
    enable_gpu, 
    num_samples::Int,
    num_ants::Int,
    num_correlators::Int,
) where S
    system = gnss(use_gpu = enable_gpu)
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), num_correlators)
    state = TrackingState(1, system, 1500Hz, 0, num_samples=num_samples, num_ants=NumAnts(num_ants), correlator=correlator)
    signal, sampling_frequency = gen_signal(system, 1, 1500Hz, num_samples, num_ants=NumAnts(num_ants))
    @benchmark track($signal, $state, $sampling_frequency)
end

# GPU kernel benchmark
function _run_kernel_wrapper_benchmark(
    gnss,
    enable_gpu::Val{true}, 
    num_samples::Int,
    num_ants::Int,
    num_correlators::Int,
)
    system = gnss(use_gpu = enable_gpu)
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), num_correlators)
    state = TrackingState(1, system, 1500Hz, 0, num_samples=num_samples, num_ants=NumAnts(num_ants), correlator=correlator)
    signal, sampling_frequency = gen_signal(system, 1, 1500Hz, num_samples, num_ants=NumAnts(num_ants))
    @benchmark CUDA.@sync Tracking.downconvert_and_correlate!(
        $system,
        $signal,
        $correlator,
        nothing,
        0,
        nothing,
        0.0,
        nothing,
        $get_code_frequency(system),
        $get_correlator_sample_shifts(system, correlator, sampling_frequency, 0),
        1500Hz,
        sampling_frequency,
        1,
        num_samples,
        1
    )
end

# CPU benchmark eqv to kernel benchmark
function _run_kernel_wrapper_benchmark(
    gnss,
    enable_gpu::Val{false}, 
    num_samples::Int,
    num_ants::Int,
    num_correlators::Int,
)
    system = gnss(use_gpu = enable_gpu)
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), num_correlators)
    state = TrackingState(1, system, 1500Hz, 0, num_samples=num_samples, num_ants=NumAnts(num_ants), correlator=correlator)
    signal, sampling_frequency = gen_signal(system, 1, 1500Hz, num_samples, num_ants=NumAnts(num_ants))
    code_replica = Vector{Int8}(undef, num_samples)
    carrier_replica = StructArray{ComplexF32}((Array{Float32}(undef, num_samples),Array{Float32}(undef, num_samples)))
    downconverted_signal = similar(carrier_replica)
    @benchmark Tracking.downconvert_and_correlate!(
        $system,
        $signal,
        $correlator,
        code_replica,
        0,
        carrier_replica,
        0.0,
        downconverted_signal,
        $get_code_frequency(system),
        $get_correlator_sample_shifts(correlator),
        1500Hz,
        sampling_frequency,
        1,
        num_samples,
        1
    )
end

function _run_kernel_nowrapper_benchmark(
    gnss,
    enable_gpu::Val{true}, 
    num_samples::Int,
    num_ants::Int,
    num_correlators::Int,
)
    system = gnss(use_gpu = enable_gpu)
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), num_correlators)
    state = TrackingState(1, system, 1500Hz, 0, num_samples=num_samples, num_ants=NumAnts(num_ants), correlator=correlator)
    signal, sampling_frequency = gen_signal(system, 1, 1500Hz, num_samples, num_ants=NumAnts(num_ants))
    correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0)
    block_dim_z = num_correlators
    block_dim_y = num_ants
    # keep num_corrs and num_ants in seperate dimensions, truncate num_samples accordingly to fit
    block_dim_x = prevpow(2, 1024 ÷ block_dim_y ÷ block_dim_z)
    threads = (block_dim_x, block_dim_y, block_dim_z)
    blocks = cld(size(signal, 1), block_dim_x)
    res_re = CUDA.zeros(Float32, blocks, block_dim_y, block_dim_z)
    res_im = CUDA.zeros(Float32, blocks, block_dim_y, block_dim_z)
    shmem_size = sizeof(ComplexF32)*block_dim_x*block_dim_y*block_dim_z
    @benchmark CUDA.@sync @cuda threads=threads blocks=blocks shmem=shmem_size Tracking.downconvert_and_correlate_kernel(
        $res_re, 
        $res_im, 
        $signal.re, 
        $signal.im,
        $system.codes,
        $Float32(get_code_frequency(system)),
        $correlator_sample_shifts,
        $Float32(1500Hz),
        $Float32(sampling_frequency),
        $Float32(0),
        $Float32(0.0),
        $get_code_length(system),
        1,
        $num_samples, 
        $num_ants,
        $num_correlators
    )
end

function do_track_benchmark(benchmark_params::Dict)
    @unpack GNSS, num_samples, num_ants, num_correlators, processor, OS = benchmark_params
    @debug "[$(Dates.Time(Dates.now()))] Benchmarking: $(GNSS), $(num_samples) samples,  $(num_ants) antenna,  $(num_correlators) correlators $(processor)"
    enable_gpu = (processor == "GPU" ? Val(true) : Val(false))
    cpu_name = Sys.cpu_info()[1].model
    cpu_name == "unkown" ? "NVIDIA ARMv8" : cpu_name
    processor_name = processor == "GPU" ? name(CUDA.CuDevice(0)) : cpu_name
    benchmark_results = _run_track_benchmark(
        GNSSDICT[GNSS], 
        enable_gpu,
        num_samples,
        num_ants,
        num_correlators
    )
    benchmark_results_w_params = copy(benchmark_params)
    benchmark_results_w_params["TrialObj"] = benchmark_results
    benchmark_results_w_params["RawTimes"] = benchmark_results.times
    benchmark_results_w_params["Minimum"] = minimum(benchmark_results).time
    benchmark_results_w_params["Median"] = median(benchmark_results).time
    benchmark_results_w_params["Mean"] = mean(benchmark_results).time
    benchmark_results_w_params["σ"] = std(benchmark_results).time
    benchmark_results_w_params["Maximum"] = maximum(benchmark_results).time
    benchmark_results_w_params[processor * " model"] = processor_name
    return benchmark_results_w_params
end

function do_kernel_wrapper_benchmark(benchmark_params::Dict)
    @unpack GNSS, num_samples, num_ants, num_correlators, processor, OS = benchmark_params
    @debug "[$(Dates.Time(Dates.now()))] Benchmarking: $(GNSS), $(num_samples) samples,  $(num_ants) antenna,  $(num_correlators) correlators $(processor)"
    enable_gpu = (processor == "GPU" ? Val(true) : Val(false))
    cpu_name = Sys.cpu_info()[1].model
    cpu_name == "unkown" ? "NVIDIA ARMv8" : cpu_name
    processor_name = processor == "GPU" ? name(CUDA.CuDevice(0)) : cpu_name
    benchmark_results = _run_kernel_wrapper_benchmark(
        GNSSDICT[GNSS], 
        enable_gpu,
        num_samples,
        num_ants,
        num_correlators
    )
    benchmark_results_w_params = copy(benchmark_params)
    benchmark_results_w_params["TrialObj"] = benchmark_results
    benchmark_results_w_params["RawTimes"] = benchmark_results.times
    benchmark_results_w_params["Minimum"] = minimum(benchmark_results).time
    benchmark_results_w_params["Median"] = median(benchmark_results).time
    benchmark_results_w_params["Mean"] = mean(benchmark_results).time
    benchmark_results_w_params["σ"] = std(benchmark_results).time
    benchmark_results_w_params["Maximum"] = maximum(benchmark_results).time
    benchmark_results_w_params[processor * " model"] = processor_name
    return benchmark_results_w_params
end