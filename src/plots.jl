# function plot_replica_benchmark(raw_data_df)
    using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

raw_data_df = collect_results(datadir("benchmarks/codereplica"))
    sort!(raw_data_df, :num_samples)
    samples = unique(Vector{Int64}(raw_data_df[!, :num_samples]))
    algorithm_names = unique(Vector{String}(raw_data_df[!, :algorithm]))
    samples = unique(Vector{Int64}(raw_data_df[!, :num_samples]))
    x = samples ./ 0.001 # convert to Hz
    algorithm_names = unique(Vector{String}(raw_data_df[!, :algorithm]))

    fig = Figure(
        # resolution = (1000, 700),
        font = "Times New Roman"
    )
    ax = Axis(
        fig,
        xlabel = "Sampling Frequency [Hz]",
        ylabel = "Generation Time [s]",
        xscale = log10,
        yscale = log10,
        title = "Comparison between global memory and texture memory code replica generation for 1 ms signal",
        xminorgridvisible = true,
        xminorticksvisible = true,
        xminorticks = IntervalsBetween(9),
        # yticks = (10.0 .^(-5:1:-3)),
        xticklabelsize = 18,
        yticklabelsize = 18

    )
    xlims!(ax, 10^6, 5*10^8)
    ylims!(ax, 1.0e-5, 3.0e-3)


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
            ax,
            x,
            time
        )
        sca[idx] = scatter!(
            ax,
            x,
            time,
            marker = markers[idx],
            markersize = 15
        )
    end
    realtime = lines!(ax, [10^6, 5 * 10^8], [10 ^ (-3), 10 ^ (-3)], color=:grey, linestyle=:dashdot)
    
    
    
    fig[1,1] = ax
    elements = [[lin[1] sca[1]], [lin[2] sca[2]]]
    labels = ["Global Memory", "Texture Memory"]
    axislegend(ax, elements, labels, "Code Replication Algorithms",  position = :rb)
    return fig
end

function plot_reduction_benchmark(raw_data_df, num_ants, num_correlators)
# using GPUAcceleratedTracking, DrWatson, DataFrames
# @quickactivate "GPUAcceleratedTracking"
# raw_data_df = collect_results(datadir("benchmarks/reduction"))
# num_ants = 4; num_correlators = 3;
    elapsed_times = raw_data_df |> 
    @filter(
        _.num_ants          == num_ants         &&
        _.num_correlators   == num_correlators    
        ) |>
    @map(
        {
            _.num_samples,
            _.algorithm,
            _.Minimum,
            _.Median,
            _.Mean,
            _.Std
        }
    ) |> DataFrame

    sort!(elapsed_times, :num_samples)
    samples = unique(Vector{Int64}(elapsed_times[!, :num_samples]))
    algorithm_names = unique(Vector{String}(elapsed_times[!, :algorithm]))

    ## BAR PLOT
    # get the data to plot
    samples = unique(Vector{Int64}(elapsed_times[!, :num_samples]))
    mindata = reshape(elapsed_times.Minimum, (length(algorithm_names), length(samples)))
    mediandata = reshape(elapsed_times.Median, (length(algorithm_names), length(samples)))
    meandata = reshape(elapsed_times.Mean, (length(algorithm_names), length(samples)))
    stddata = reshape(elapsed_times.Std, (length(algorithm_names), length(samples)))
    algorithm_names = unique(Vector{String}(elapsed_times[!, :algorithm]))

    # defines figure and axes    
    fig_bar = Figure(font = "Times New Roman")
    ax_bar = Axis(
        fig_bar,
        xlabel = "Number of elements",
        ylabel = "Processing Time [s]",
        xscale = log10,
        # yscale = log10,
        title = "Reduction Algorithm Comparison for M = 4, L = 3",
        xticks = (samples, ["2048", "4096", "8192", "16384", "32768"]),
        xticklabelrotation = pi/8,
        xminorticksvisible = false
    )

    colors = [:blue, :yellow, :green]

    # plot the data to the axes
    barplot!(
        ax_bar,
        vec(repeat(samples, inner=3)),
        vec(reshape(mindata, (15, 1))),
        dodge = vec(repeat([1,2,3], 5)),
        color = vec(repeat(colors, 5)),
        strokewidth = 0.1,
        width = 300 .* (2 .^ ((repeat(1:length(samples), inner=3))))
        )
    labels = ["cplx_multi", "cplx", "pure"]
    title = "Algorithms"
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]

    Legend(fig_bar[1,2], elements, labels, "Reduction Algorithms", position = :tr)
    fig_bar[1,1] = ax_bar
    fig_bar
end