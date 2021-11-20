using DrWatson
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, DataFrames, Plots, PGFPlotsX, StatsPlots
using Unitful
import Unitful: ns

df = collect_results(datadir("benchmarks"))

function plot_min_exec_time(df::DataFrame)
    gpudf = df |> @filter(_.processor=="GPU") |> @map({_.num_samples, _.Minimum}) |> DataFrame
    cpudf = df |> @filter(_.processor=="CPU") |> @map({_.num_samples, _.Minimum}) |> DataFrame

    num_samples = Vector{Int64}(gpudf[!, :num_samples])
    gputimes = Vector{Float64}(gpudf[!, :Minimum])
    cputimes = Vector{Float64}(cpudf[!, :Minimum])

    plot(num_samples, 
        [gputimes, cputimes],
        title = "GPU vs CPU: $(1) Antenna, $(3) Correlators",
        label = ["GPU" "CPU"],
        ylabel = "Execution Time",
        xaxis = ("Number of Samples", :plain)
    )
end