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
data_bar = mean(data)
data_med = median(data)
data_max = maximum(data)

using CairoMakie
fig = Figure(font = "Times New Roman")
ax = Axis(
    fig,
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Relative Code Phase Error [%]",
    xscale = log10,
    title = "Relative code phase error of the texture memory code replica generation for 1 ms GPS L1 C/A signal",
    # xlim = [0 400_000],
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    # yticks = (10.0 .^(-5:1:-3)),
        # yticks = (10.0 .^(-5:1:-3)),

    xticklabelsize = 18,
    yticklabelsize = 18
)
xlims!(ax, 10^6, 5*10^8)
# string_data_bar = "$(round(data_bar, sigdigits=3))%"
# string_data_max = "$(round(data_max, sigdigits=3))%"
# string_data_med = "$(round(data_med, sigdigits=3))%"

# # textmu = "μ = " * string_data_bar 
# # textmax = "max = " * string_data_max
# textmed = "median = " * string_data_med
lines!(ax, x,data)
# hlines!(ax, data_bar, color = :dimgrey, linestyle = :dash)
# hlines!(ax, data_med, color = :dimgrey, linestyle = :dash)
# hlines!(ax, data_max, color = :dimgrey, linestyle = :dash)
# text!(textmu, position = (9*10^8, 0.1+data_bar), align = (:right, :baseline))
# text!(textmax, position = (9*10^8, data_max - 0.2), align = (:right, :baseline))
# text!(textmed, position = (5*10^8, 0.5+data_med), align = (:center, :baseline))

fig[1,1] = ax
fig

@quickactivate "GPUAcceleratedTracking"

raw_data_df = collect_results(datadir("benchmarks/codereplica"))

sort!(raw_data_df, :num_samples)
samples = unique(Vector{Int64}(raw_data_df[!, :num_samples]))
algorithm_names = unique(Vector{String}(raw_data_df[!, :algorithm]))
samples = unique(Vector{Int64}(raw_data_df[!, :num_samples]))
x = samples ./ 0.001 # convert to Hz
algorithm_names = unique(Vector{String}(raw_data_df[!, :algorithm]))

# fig = Figure(
#     # resolution = (1000, 700),
#     font = "Times New Roman"
# )
ax2 = Axis(
    fig,
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Generation Time [s]",
    xscale = log10,
    yscale = log10,
    title = "Comparison between global memory and texture memory code replica generation for 1 ms GPS L1 C/A signal",
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    # yticks = (10.0 .^(-5:1:-3)),
    xticklabelsize = 18,
    yticklabelsize = 18

)
xlims!(ax2, 10^6, 5*10^8)
ylims!(ax2, 1.0e-5, 3.0e-3)


lin = Array{Lines}(undef, length(algorithm_names)); 
sca = Array{Scatter}(undef, length(algorithm_names));
markers = [:circle, :rect]
for (idx, name) = enumerate(algorithm_names)
    time = 10 ^ (-9) * vec((
        raw_data_df |>
        @filter(
            _.algorithm == name
        ) |> DataFrame
    ).Minimum)

    lin[idx] = lines!(
        ax2,
        x,
        time
    )
    sca[idx] = scatter!(
        ax2,
        x,
        time,
        marker = markers[idx],
        markersize = 15
    )
end
realtime = lines!(ax2, [10^6, 5 * 10^8], [10 ^ (-3), 10 ^ (-3)], color=:grey, linestyle=:dashdot)



fig[2,1] = ax2
fig
elements = [[lin[1] sca[1]], [lin[2] sca[2]]]
labels = ["Global Memory", "Texture Memory"]
axislegend(ax2, elements, labels, "Code Replication Algorithms",  position = :lt)
fig
# save( plotsdir("benchmark_textmem.pdf"), fig)

save(plotsdir("code_phase.pdf"), fig)