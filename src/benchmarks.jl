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
    cpu_name = Sys.cpu_info()[1].model
    cpu_name == "unknown" ? "NVIDIA ARMv8" : cpu_name
    processor_name = processor == "GPU" ? name(CUDA.CuDevice(0)) : cpu_name
    benchmark_results_w_params[processor * " model"] = processor_name
    benchmark_results_w_params[algorithm] = ALGN
end

function _run_track_benchmark(
    gnss,
    enable_gpu, 
    num_samples::Int,
    num_ants::Int,
    num_correlators::Int,
) where S
    system = gnss(use_gpu = enable_gpu)
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), num_correlators)
    state = TrackingState(1, system, 1500Hz, 0,  num_samples=num_samples, num_ants=NumAnts(num_ants), correlator=correlator)
    signal, sampling_frequency = gen_signal(system, 1, 1500Hz, num_samples, num_ants=NumAnts(num_ants))
    @benchmark track($signal, $state, $sampling_frequency)
end

# GPU kernel benchmark
function _run_kernel_benchmark(
    gnss,
    enable_gpu::Val{true}, 
    num_samples::Int,
    num_ants::NumAnts{NANT},
    num_correlators::NumAccumulators{NCOR},
    algorithm::KernelAlgorithm{2}
) where {NANT, NCOR}
    system = gnss(use_gpu = enable_gpu)
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), num_correlators)
    state = TrackingState(1, system, 1500Hz, 0, num_samples=num_samples, num_ants=NumAnts(num_ants), correlator=correlator)
    signal, sampling_frequency = gen_signal(system, 1, 1500Hz, num_samples, num_ants=NumAnts(num_ants))
    num_of_shifts = get_num_accumulators(correlator) - 1
    code_replica = CUDA.zeros(Float32, num_samples + num_of_shifts)
    carrier_replica = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples), CUDA.zeros(Float32, num_samples)))
    downconverted_signal = StructArray{ComplexF32}((CUDA.zeros(Float32, num_samples, num_ants), CUDA.zeros(Float32, num_samples, num_ants)))
    threads_per_block = [1024, 512]
    blocks_per_grid = cld.(num_samples, threads_per_block)
    partial_sum = StructArray{ComplexF32}((CUDA.zeros(Float32, (cld(num_samples, threads_per_block[2]), num_ants, num_correlators)),CUDA.zeros(Float32, (cld(num_samples, threads_per_block[2]), num_ants, num_correlators))))

    kernel_algorithm(
        code_replica,
        system.codes,
        get_code_frequency(system),
        sampling_frequency,
        0.0f0,
        1,
        num_samples,
        num_of_shifts,
        get_code_length(system),
        partial_sum.re,
        partial_sum.im,
        carrier_replica.re,
        carrier_replica.im,
        downconverted_signal.re,
        downconverted_signal.im,
        signal.re,
        signal.im,
        get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5),
        1500Hz,
        0.0f0,
        num_ants,
        algorithm
    )

end

# CPU benchmark eqv to kernel benchmark
function _run_kernel_benchmark(
    gnss,
    enable_gpu::Val{false}, 
    num_samples::Int,
    num_ants::NumAnts{NANT},
    num_correlators::NumAccumulators{NCOR},
    algortihm
) where {NANT, NCOR}
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
        $code_replica,
        0,
        $carrier_replica,
        0.0,
        $downconverted_signal,
        $get_code_frequency(system),
        $get_correlator_sample_shifts(system, correlator, sampling_frequency, 0),
        1500Hz,
        $sampling_frequency,
        1,
        $num_samples,
        1
    )
end

function run_track_benchmark(benchmark_params::Dict)
    @unpack GNSS, num_samples, num_ants, num_correlators, processor, OS, algorithm = benchmark_params
    @debug "[$(Dates.Time(Dates.now()))] Benchmarking: $(GNSS), $(num_samples) samples,  $(num_ants) antenna,  $(num_correlators) correlators $(processor) w/ Algorithm:$(algorithm)"
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
    add_results!(benchmark_results_w_params, benchmark_results)
    add_metadata!(benchmark_results_w_params, processor, algorithm)
    return benchmark_results_w_params
end

function run_kernel_benchmark(benchmark_params::Dict)
    @unpack GNSS, num_samples, num_ants, num_correlators, processor, OS, algorithm = benchmark_params
    # @debug "[$(Dates.Time(Dates.now()))] Benchmarking: $(GNSS), $(num_samples) samples,  $(num_ants) antenna,  $(num_correlators) correlators $(processor) w/ Algorithm:$(algorithm)"
    enable_gpu = (processor == "GPU" ? Val(true) : Val(false))
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