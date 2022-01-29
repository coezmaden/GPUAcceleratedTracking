using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

raw_data_df = collect_results(datadir("benchmarks/reduction"))

num_ants = 4; num_correlators = 3;

using Query, Makie, CairoMakie, ColorSchemes
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

colors = colorschemes[:Set2_3]

# plot the data to the axes
barplot!(
    ax_bar,
    vec(repeat(samples, inner=3)),
    vec(reshape(mindata, (15, 1))),
    dodge = vec(repeat([1,2,3], 5)),
    color = colors[vec(repeat([1,2,3], 5))],
    strokewidth = 0.1,
    width = 300 .* (2 .^ ((repeat(1:length(samples), inner=3))))
    )
# barplot!(
#     ax_bar,
#     vec(repeat(samples, inner=3)),
#     vec(reshape(meandata, (15, 1))),
#     dodge = vec(repeat([1,2,3], 5)),
#     color = vec(repeat(colors, 5))
#     )
# plot the mean transparent
labels = ["cplx_multi", "cplx", "pure"]
title = "Algorithms"
elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]

Legend(fig_bar[1,2], elements, labels, "Reduction Algorithms", position = :tr)
fig_bar[1,1] = ax_bar
fig_bar
save( plotsdir("reduction.pdf"), fig_bar)    