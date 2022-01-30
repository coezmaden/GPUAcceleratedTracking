using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

raw_data_df = collect_results(datadir("benchmarks/kernel/jetson"))


using Query, CairoMakie, Makie

gnss_string = Dict(
    "GPSL1" => "GPS L1 C/A",
    "GPSL5" => "GPS L5 P"
)
os = "linux"

fig = plot_all(raw_data_df)

function plot_all(raw_data_df)
    fig = Figure(
        resolution = (1000, 700),
        font = "Times New Roman"
    )

    sort!(raw_data_df)

    samples = unique(Vector{Int64}(raw_data_df[!, :num_samples]))
    sort!(samples)
    x = samples / 0.001
    algorithm_names = unique(Vector{String}(raw_data_df[!, :algorithm])
    )
    sort!(algorithm_names)
    algorithm_names = algorithm_names[2:end] #disregard 1 alg

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
    xlims!(gpsl1_1ms_1_3_ax, [10^(6),  3 * 10^(8)])
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
            x, 
            times .* 10 ^ (-9)
        )
        sca[idx] = scatter!(
            gpsl1_1ms_1_3_ax, 
            x, 
            times .* 10 ^ (-9),
            marker = markers[idx],
            markersize = 15
        )
    end
    realtime = lines!(gpsl1_1ms_1_3_ax, [10^(6), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash )
    fig[1,1] = gpsl1_1ms_1_3_ax

    # PLOT GPS L1 C/A
    os = "windows"
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
    xlims!(gpsl1_1ms_4_3_ax, [10^(6),  3 * 10^(8)])
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
            x, 
            times .* 10 ^ (-9)
        )
        sca[idx] = scatter!(
            gpsl1_1ms_4_3_ax, 
            x, 
            times .* 10 ^ (-9),
            marker = markers[idx],
            markersize = 15
        )
    end
    realtime = lines!(gpsl1_1ms_4_3_ax, [10^(6), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash )
    fig[2,1] = gpsl1_1ms_4_3_ax


    # PLOT GPS L5 C/A
    gnss = "GPSL5"
    num_ants = 1; num_correlators = 3

    gpsl5_1ms_1_3_ax = Axis(fig, 
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
    xlims!(gpsl5_1ms_1_3_ax, [10^(6), 3 * 10^(8)])
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
            x, 
            times .* 10 ^ (-9)
        )
        sca[idx] = scatter!(
            gpsl5_1ms_1_3_ax, 
            x, 
            times .* 10 ^ (-9),
            marker = markers[idx],
            markersize = 15
        )
    end
    realtime = lines!(gpsl5_1ms_1_3_ax, [10^(6), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash )
    fig[1,2] = gpsl5_1ms_1_3_ax


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
        xticks = collect(10 .^ (6:9)),
        # yticks = collect(10.0 .^ (-6:-2)),
        title = "$(gnss_string[gnss]) T=1 ms, M=$num_ants, L=$num_correlators"
    )
    xlims!(gpsl5_1ms_4_3_ax, [10^(6), 3 * 10^(8)])
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
            x, 
            times .* 10 ^ (-9)
        )
        sca[idx] = scatter!(
            gpsl5_1ms_4_3_ax, 
            x, 
            times .* 10 ^ (-9),
            marker = markers[idx],
            markersize = 15
        )
    end
    realtime = lines!(gpsl5_1ms_4_3_ax, [10^(6), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash )
    fig[2,2] = gpsl5_1ms_4_3_ax
    

    labels = algorithm_names
    # Legend(fig[2,1], lin, algorithm_names, "Reduction Algorithms", framevisible = false)
    # Legend(fig, "Algorithms", algorithm_names)
    return fig
end