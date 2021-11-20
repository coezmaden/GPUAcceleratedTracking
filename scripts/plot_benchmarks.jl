using DrWatson
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, DataFrames, Plots, PGFPlotsX, StatsPlots
using Unitful
import Unitful: ns

df = collect_results(datadir("benchmarks"))

function plot_min_exec_time(df::DataFrame)
    gpudf = df |> @filter(_.processor=="GPU") |> @filter(_.num_ants==4) |> @filter(_.num_correlators==5) |> @map({_.num_samples, _.Minimum}) |> DataFrame
    cpudf = df |> @filter(_.processor=="CPU") |> @filter(_.num_ants==4) |> @filter(_.num_correlators==5) |> @map({_.num_samples, _.Minimum}) |> DataFrame

    num_samples = unique(Vector{Int64}(gpudf[!, :num_samples]))
    # num_ants = unique(Vector{Int64}(gpudf[!, :num_ants]))
    # num_correlators = unique(Vector{Int64}(gpudf[!, :num_correlators]))
    
    # gputimes = Array{Float64}(undef, (length(num_samples), length(num_ants), length(num_correlators)))
   
    gputimes = Vector{Float64}(gpudf[!, :Minimum])
    cputimes = Vector{Float64}(cpudf[!, :Minimum])

    plot(num_samples, 
        [gputimes, cputimes],
        title = "GPU vs CPU: $(4) Antenna, $(5) Correlators",
        label = ["GPU" "CPU"],
        ylabel = "Execution Time",
        xaxis = ("Number of Samples", :plain)
    )
end