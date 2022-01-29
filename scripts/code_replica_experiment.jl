using DrWatson, GPUAcceleratedTracking, CUDA, Tracking, GNSSSignals, StructArrays, ProgressMeter;
import Tracking: Hz, ms;
@quickactivate "GPUAcceleratedTracking"

N = 2048:32:262_144
err_rel  = zeros(length(N))
@showprogress 0.5 for (idx, num_samples) in enumerate(N)
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
    signal, sampling_frequency = gen_signal(system, prn, carrier_frequency, num_samples, num_ants = NumAnts(num_ants), start_code_phase = start_code_phase, start_carrier_phase = carrier_phase)
    
    # Generate correlator;
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(system, correlator, sampling_frequency, 0.5)
    num_of_shifts = correlator_sample_shifts[end] - correlator_sample_shifts[1]
    
    # Generate blank code and carrier replica, and downconverted signal;
    # code_replica_cpu = zeros(Float32, num_samples + num_of_shifts)
    code_replica = CUDA.zeros(Float32, num_samples + num_of_shifts)
    code_replica_text_mem = CUDA.zeros(Float32, num_samples + num_of_shifts)

    # Generate CUDA kernel tuning parameters;
    threads_per_block = 768
    blocks_per_grid = cld.(num_samples, threads_per_block)
    
    @cuda threads=threads_per_block blocks=blocks_per_grid gen_code_replica_texture_mem_kernel!(
        code_replica_text_mem,
        codes_text_mem, # texture memory codes
        code_frequency,
        sampling_frequency,
        start_code_phase,
        prn,
        num_samples + num_of_shifts,
        correlator_sample_shifts[1],
        code_length
    )
    @cuda threads=threads_per_block blocks=blocks_per_grid gen_code_replica_kernel!(
        code_replica,
        codes, # texture memory codes
        code_frequency,
        sampling_frequency,
        start_code_phase,
        prn,
        num_samples + num_of_shifts,
        correlator_sample_shifts[1],
        code_length
    )
    # Tracking.gen_code_replica!(code_replica_cpu, system, code_frequency, sampling_frequency, start_code_phase, 1, num_samples, correlator_sample_shifts, prn)
    
    # code_replica_h = Array(code_replica)
    # code_replica_text_mem_h = Array(code_replica_text_mem)
    # signal = StructArray{ComplexF32}((ones(Float32, num_samples), zeros(Float32, num_samples) ))
    # code_phases = get_code_frequency(system) / sampling_frequency .* (0:num_samples-1) .+ start_code_phase
    # spread_signal = StructArray(signal .* system_h.codes[1 .+ mod.(floor.(Int, code_phases), get_code_length(system)), prn])
    # accums_true = Tracking.correlate(correlator, spread_signal, code_replica_cpu, correlator_sample_shifts, 1, num_samples)
    # accums = Tracking.correlate(correlator, spread_signal, code_replica_h, correlator_sample_shifts, 1, num_samples)
    # accums_text_mem = Tracking.correlate(correlator, spread_signal, code_replica_text_mem_h, correlator_sample_shifts, 1, num_samples)

    # err_rel = sum(abs.(code_replica - code_replica_text_mem)) / num_samples
    err_rel[idx] = sum(abs.(code_replica - code_replica_text_mem)) / num_samples
end


x = vec(collect(N / 0.001)) # convert to Hz
data = vec(100 .* err_rel)

using CairoMakie
fig = Figure(font = "Times New Roman")
ax = Axis(
    fig,
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Relative Code Phase Error [%]",
    xscale = log10,
    title = "Relative code phase error",
    xlim = [0 400_000],
    xmajorgridvisible = true,
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    xticks = [10^6, 10^7, 10^8, 10^9]
)


lines!(ax, x, 100 .* err_rel)
fig[1,1] = ax
fig

save(plotsdir("code_phase_error.pdf"), fig)