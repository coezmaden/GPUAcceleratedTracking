function _run_benchmark(
    gnss::Val{S},
    enable_gpu::Bool, 
    num_samples::Int
) where {S, T}
    # system = eval(gnss)(use_gpu = Val(enable_gpu))
    system = S(use_gpu = Val(enable_gpu))
    state = TrackingState(1, system, 1500Hz, 0, num_samples=num_samples)
    signal, sampling_frequency = gen_signal(system, 1, 1500Hz, num_samples)
    @benchmark track($signal, $state, $sampling_frequency)
end

function do_benchmark(benchmark_params::Dict)
    @unpack GNSS, num_samples, processor = benchmark_params 
    benchmark_results = _run_benchmark(
        GNSS, 
        processor == "GPU" ? true : false,
        num_samples
        )
    benchmark_results_w_params = copy(benchmark_params)
    benchmark_results_w_params["Trial"] = benchmark_results
    return benchmark_results_w_params
end