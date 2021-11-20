function _run_track_benchmark(
    gnss,
    enable_gpu::Bool, 
    num_samples::Int
) where S
    # system = eval(gnss)(use_gpu = Val(enable_gpu))
    system = gnss(use_gpu = Val(enable_gpu))
    state = TrackingState(1, system, 1500Hz, 0, num_samples=num_samples)
    signal, sampling_frequency = gen_signal(system, 1, 1500Hz, num_samples)
    @benchmark track($signal, $state, $sampling_frequency)
end

function do_track_benchmark(benchmark_params::Dict)
    @unpack GNSS, num_samples, processor, OS = benchmark_params
    @debug "[$(Dates.Time(Dates.now()))] Benchmarking: $(GNSS), $(num_samples) samples, $(processor)"
    enable_gpu = (processor == "GPU" ? true : false)
    cpu_name = Sys.cpu_info()[1].model
    cpu_name == "unkown" ? "NVIDIA ARMv8" : cpu_name
    processor_name = enable_gpu ? name(CUDA.CuDevice(0)) : cpu_name
    benchmark_results = _run_track_benchmark(
        GNSSDICT[GNSS], 
        enable_gpu,
        num_samples
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