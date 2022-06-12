using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

raw_data_df = collect_results(datadir("benchmarks/kernel/atbenchmark"))

using Query, CairoMakie

gnss_string = Dict(
    "GPSL1" => "GPS L1 C/A",
    "GPSL5" => "GPS L5 P"
)
os = "linux"
   
fig = Figure(
    resolution = (1000, 1000),
    # font = "Times New Roman",
    # fontsize = 10,
)

sort!(raw_data_df)

samples = unique(Vector{Int64}(raw_data_df[!, :num_samples]))
sort!(samples)
freqs_gpsl1 = samples / 0.001
freqs_gpsl5 = samples[5:end] / 0.001
algorithm_names = unique(Vector{String}(raw_data_df[!, :algorithm]))
sort!(algorithm_names)
# algorithm_names = algorithm_names[2:end] #disregard 1 alg

# PLOT GPS L1 C/A
gnss = "GPSL1"
num_ants = 1; num_correlators = 3

gpsl1_1ms_1_3_ax = Axis(fig, 
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Processing Time [s]",
    xscale = log10,
    yscale = log10,
    xmajorgridvisible = true,
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    xticks = collect(10 .^ (6:9)),
    # yticks = collect(10.0 .^ (-6:-2)),
    title = "$(gnss_string[gnss]) T=1 ms, M=$num_ants, L=$num_correlators"
)
# xlims!(gpsl1_1ms_1_3_ax, [10^(6),  3 * 10^(8)])
# ylims!(gpsl1_1ms_1_3_ax, [10^(-6), 2*10^(-3)])
markers = [:rect, :utriangle, :circle, :diamond, :dtriangle]
lin = Array{Lines}(undef, length(algorithm_names)); 
sca = Array{Scatter}(undef, length(algorithm_names));
for (idx, algorithm) in enumerate(algorithm_names)
    times = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "GPU" &&
                    _.algorithm == algorithm_names[idx] &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
    lin[idx] = lines!(
        gpsl1_1ms_1_3_ax, 
        freqs_gpsl1, 
        times .* 10 ^ (-9)
    )
    sca[idx] = scatter!(
        gpsl1_1ms_1_3_ax, 
        freqs_gpsl1, 
        times .* 10 ^ (-9),
        marker = markers[idx],
        markersize = 15
    )
end
cputimes = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "CPU" &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
cpulin = lines!(
    gpsl1_1ms_1_3_ax,
    freqs_gpsl1,
    cputimes .* 10 ^ (-9),
)
cpusca = scatter!(
    gpsl1_1ms_1_3_ax,
    freqs_gpsl1,
    cputimes .* 10 ^ (-9),
    marker = :star5,
    markersize = 15
)
realtime = lines!(gpsl1_1ms_1_3_ax, [10^(6), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dashdot, color =:dimgrey )
fig[1,1] = gpsl1_1ms_1_3_ax
fig
# PLOT GPS L1 C/A
gnss = "GPSL1"
num_ants = 4; num_correlators = 3

gpsl1_1ms_4_3_ax = Axis(fig, 
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Processing Time [s]",
    xscale = log10,
    yscale = log10,
    xmajorgridvisible = true,
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    xticks = collect(10 .^ (6:9)),
    # yticks = collect(10.0 .^ (-6:-2)),
    title = "$(gnss_string[gnss]) T=1 ms, M=$num_ants, L=$num_correlators"
)
# xlims!(gpsl1_1ms_4_3_ax, [10^(6),  3 * 10^(8)])
# ylims!(gpsl1_1ms_4_3_ax, [10^(-6), 2*10^(-3)])
markers = [:rect, :utriangle, :circle, :diamond, :dtriangle]
lin = Array{Lines}(undef, length(algorithm_names)); 
sca = Array{Scatter}(undef, length(algorithm_names));
for (idx, algorithm) in enumerate(algorithm_names)
    times = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "GPU" &&
                    _.algorithm == algorithm_names[idx] &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
    lin[idx] = lines!(
        gpsl1_1ms_4_3_ax, 
        freqs_gpsl1, 
        times .* 10 ^ (-9)
    )
    sca[idx] = scatter!(
        gpsl1_1ms_4_3_ax, 
        freqs_gpsl1, 
        times .* 10 ^ (-9),
        marker = markers[idx],
        markersize = 15
    )
end
cputimes = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "CPU" &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
cpulin = lines!(
    gpsl1_1ms_4_3_ax,
    freqs_gpsl1,
    cputimes .* 10 ^ (-9),
)
cpusca = scatter!(
    gpsl1_1ms_4_3_ax,
    freqs_gpsl1,
    cputimes .* 10 ^ (-9),
    marker = :star5,
    markersize = 15
)
realtime = lines!(gpsl1_1ms_4_3_ax, [10^(6), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash, color = :dimgrey )
fig[1,2] = gpsl1_1ms_4_3_ax
fig
# PLOT GPS L1 C/A
gnss = "GPSL1"
num_ants = 4; num_correlators = 7

gpsl1_1ms_4_7_ax = Axis(fig, 
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Processing Time [s]",
    xscale = log10,
    yscale = log10,
    xmajorgridvisible = true,
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    xticks = collect(10 .^ (6:9)),
    # yticks = collect(10.0 .^ (-6:-2)),
    title = "$(gnss_string[gnss]) T=1 ms, M=$num_ants, L=$num_correlators"
)
# xlims!(gpsl1_1ms_4_7_ax, [10^(6),  3 * 10^(8)])
# ylims!(gpsl1_1ms_4_7_ax, [10^(-6), 2*10^(-3)])
markers = [:rect, :utriangle, :circle, :diamond, :dtriangle]
lin = Array{Lines}(undef, length(algorithm_names)); 
sca = Array{Scatter}(undef, length(algorithm_names));
for (idx, algorithm) in enumerate(algorithm_names)
    times = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "GPU" &&
                    _.algorithm == algorithm_names[idx] &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
    lin[idx] = lines!(
        gpsl1_1ms_4_7_ax, 
        freqs_gpsl1, 
        times .* 10 ^ (-9)
    )
    sca[idx] = scatter!(
        gpsl1_1ms_4_7_ax, 
        freqs_gpsl1, 
        times .* 10 ^ (-9),
        marker = markers[idx],
        markersize = 15
    )
end
cputimes = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "CPU" &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
cpulin = lines!(
    gpsl1_1ms_4_7_ax,
    freqs_gpsl1,
    cputimes .* 10 ^ (-9),
)
cpusca = scatter!(
    gpsl1_1ms_4_7_ax,
    freqs_gpsl1,
    cputimes .* 10 ^ (-9),
    marker = :star5,
    markersize = 15
)
realtime = lines!(gpsl1_1ms_4_7_ax, [10^(6), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash, color = :dimgrey )
fig[1,3] = gpsl1_1ms_4_7_ax
fig

# PLOT GPS L5 C/A
gnss = "GPSL5"
num_ants = 1; num_correlators = 3
algorithm_names = unique(Vector{String}(raw_data_df[!, :algorithm])
)
sort!(algorithm_names)
gpsl5_1ms_1_3_ax = Axis(fig, 
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Processing Time [s]",
    xscale = log10,
    yscale = log10,
    xmajorgridvisible = true,
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    # xticks = collect(10 .^ (6:9)),
    # yticks = collect(10.0 .^ (-6:-2)),
    title = "$(gnss_string[gnss]) T=1 ms, M=$num_ants, L=$num_correlators"
)
# xlims!(gpsl5_1ms_1_3_ax, [10^(7), 3 * 10^(8)])
# ylims!(gpsl5_1ms_1_3_ax,[1e-6, 5e-3])
markers = [:rect, :utriangle, :circle, :diamond, :dtriangle]
lin = Array{Lines}(undef, length(algorithm_names)); 
sca = Array{Scatter}(undef, length(algorithm_names));
for (idx, algorithm) in enumerate(algorithm_names)
    times = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "GPU" &&
                    _.algorithm == algorithm_names[idx] &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
    lin[idx] = lines!(
        gpsl5_1ms_1_3_ax, 
        freqs_gpsl5, 
        times .* 10 ^ (-9)
    )
    sca[idx] = scatter!(
        gpsl5_1ms_1_3_ax, 
        freqs_gpsl5, 
        times .* 10 ^ (-9),
        marker = markers[idx],
        markersize = 15
    )
end
cputimes = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "CPU" &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
cpulin = lines!(
    gpsl5_1ms_1_3_ax,
    freqs_gpsl5,
    cputimes .* 10 ^ (-9),
)
cpusca = scatter!(
    gpsl5_1ms_1_3_ax,
    freqs_gpsl5,
    cputimes .* 10 ^ (-9),
    marker = :star5,
    markersize = 15
)
realtime = lines!(gpsl5_1ms_1_3_ax, [2 * 10^(7), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash, color = :dimgrey )
fig[2,1] = gpsl5_1ms_1_3_ax
fig

# PLOT GPS L5 C/A
gnss = "GPSL5"
num_ants = 4; num_correlators = 3

gpsl5_1ms_4_3_ax = Axis(fig, 
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Processing Time [s]",
    xscale = log10,
    yscale = log10,
    xmajorgridvisible = true,
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    # xticks = collect(10 .^ (6:9)),
    # yticks = collect(10.0 .^ (-6:-2)),
    title = "$(gnss_string[gnss]) T=1 ms, M=$num_ants, L=$num_correlators"
)
# xlims!(gpsl5_1ms_4_3_ax, [10^(7), 3 * 10^(8)])
# ylims!(gpsl5_1ms_4_3_ax,[1e-6, 5e-3])
markers = [:rect, :utriangle, :circle, :diamond, :dtriangle]
lin = Array{Lines}(undef, length(algorithm_names)); 
sca = Array{Scatter}(undef, length(algorithm_names));
for (idx, algorithm) in enumerate(algorithm_names)
    times = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "GPU" &&
                    _.algorithm == algorithm_names[idx] &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
    lin[idx] = lines!(
        gpsl5_1ms_4_3_ax, 
        freqs_gpsl5, 
        times .* 10 ^ (-9)
    )
    sca[idx] = scatter!(
        gpsl5_1ms_4_3_ax, 
        freqs_gpsl5, 
        times .* 10 ^ (-9),
        marker = markers[idx],
        markersize = 15
    )
end
cputimes = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "CPU" &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
cpulin = lines!(
    gpsl5_1ms_4_3_ax,
    freqs_gpsl5,
    cputimes .* 10 ^ (-9),
)
cpusca = scatter!(
    gpsl5_1ms_4_3_ax,
    freqs_gpsl5,
    cputimes .* 10 ^ (-9),
    marker = :star5,
    markersize = 15
)    
realtime = lines!(gpsl5_1ms_4_3_ax, [2 * 10^(7), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash, color = :dimgrey )
fig[2,2] = gpsl5_1ms_4_3_ax
fig

# PLOT GPS L5 C/A
gnss = "GPSL5"
num_ants = 4; num_correlators = 7

gpsl5_1ms_4_7_ax = Axis(fig, 
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Processing Time [s]",
    xscale = log10,
    yscale = log10,
    xmajorgridvisible = true,
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    # xticks = collect(10 .^ (6:9)),
    # yticks = collect(10.0 .^ (-6:-2)),
    title = "$(gnss_string[gnss]) T=1 ms, M=$num_ants, L=$num_correlators"
)
# xlims!(gpsl5_1ms_4_7_ax, [10^(7), 3 * 10^(8)])
# ylims!(gpsl5_1ms_4_7_ax,[1e-6, 5e-3])
markers = [:rect, :utriangle, :circle, :diamond, :dtriangle]
lin = Array{Lines}(undef, length(algorithm_names)); 
sca = Array{Scatter}(undef, length(algorithm_names));
for (idx, algorithm) in enumerate(algorithm_names)
    times = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "GPU" &&
                    _.algorithm == algorithm_names[idx] &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
    lin[idx] = lines!(
        gpsl5_1ms_4_7_ax, 
        freqs_gpsl5, 
        times .* 10 ^ (-9)
    )
    sca[idx] = scatter!(
        gpsl5_1ms_4_7_ax, 
        freqs_gpsl5, 
        times .* 10 ^ (-9),
        marker = markers[idx],
        markersize = 15
    )
end
cputimes = Float64.(
        (
            raw_data_df |>
                @filter(
                    _.processor == "CPU" &&
                    _.num_ants == num_ants &&
                    _.num_correlators == num_correlators &&
                    _.GNSS == gnss
                ) |> 
                @map(
                    {
                        _.Minimum
                    }
                ) |> DataFrame
        ).Minimum
    )
cpulin = lines!(
    gpsl5_1ms_4_7_ax,
    freqs_gpsl5,
    cputimes .* 10 ^ (-9),
)
cpusca = scatter!(
    gpsl5_1ms_4_7_ax,
    freqs_gpsl5,
    cputimes .* 10 ^ (-9),
    marker = :star5,
    markersize = 15
)    
realtime = lines!(gpsl5_1ms_4_7_ax, [2 * 10^(7), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash, color = :dimgrey )
fig[2,3] = gpsl5_1ms_4_7_ax
fig

labels = [algorithm_names; "SIMD CPU"]
entries = [];
for entry in 1:6
    entry == 6 ? entries = vcat(entries, [cpulin; cpusca]) : entries = vcat(entries, [lin[entry]; sca[entry]])
end

lfig = 
Legend(
    fig[3,1:3], 
    [
        [lin[1], sca[1]],
        [lin[2], sca[2]],
        [lin[3], sca[3]],
        [lin[4], sca[4]],
        [lin[5], sca[5]],
        [cpulin, cpusca],
    ], 
    labels, 
    framevisible = false
)


supertitle = Label(fig[0, :], 
"Desktop PC, Intel Core i5-9600K @ 3.70 GHz, NVIDIA GeForce GTX 1050 Ti (Pascal)")
fig
save(plotsdir("desktop_allplots.pdf"), fig)