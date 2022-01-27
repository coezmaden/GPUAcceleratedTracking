using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

raw_data_df = collect_results(datadir("benchmarks/reduction"))

num_ants = 4; num_correlators = 3;

using Query;
elapsed_min_times = raw_data_df |> 
@filter(
    _.num_ants          == num_ants         &&
    _.num_correlators   == num_correlators    
    ) |>
@map(
    {
        _.num_samples,
        _.algorithm,
        _.Minimum,
    }
) |> DataFrame

sort!(elapsed_min_times, :num_samples)
samples = unique(Vector{Int64}(elapsed_min_times[!, :num_samples]))
algorithm_names = unique(Vector{String}(elapsed_min_times[!, :algorithm]))

using Makie, CairoMakie
using CairoMakie
let
    x = LinRange(-10,10,200)
    fig = Figure(resolution = (700, 450))
    ax = Axis(fig, xlabel = "x", ylabel = "y")
    # filled curve 1
    band!(x, sin.(x), sin.(x) .+ 1; color = ("#E69F00", 0.2))
    # filled curve 2
    band!(x, cos.(x), 1 .+ cos.(x); color = (:red, 0.2))
    fig[1,1] = ax
    fig
    #save("Bands.png"), fig, px_per_unit = 2.0)
end


fig = Figure(resoltion = (800, 600))
ax = Axis(fig, xlabel = "Number of elements", ylabel = "Processing Time [s]")



data = transpose(elapsed_min_times.Minimum)
xs = samples
labels = permutedims(algorithm_names)
colors = [:black :blue :red]
# pl = plot(
#         xs,
#         data,
#         color = colors,
#         label = labels,
#         legend = :bottomright,
#     )
# yaxis!("Processing Time [s]", :log10, minorgrid = true)
# xaxis!("Number of elements", :log10, minorgrid = true)