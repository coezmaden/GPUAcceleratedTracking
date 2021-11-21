using DrWatson
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, DataFrames, Plots, PGFPlotsX, StatsPlots
using Unitful
import Unitful: ns

# df = collect_results(datadir("benchmarks/track"))
df = collect_results(datadir("benchmarks/kernel"))


plot_min_exec_time(df)