using DrWatson
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, DataFrames, Plots, PGFPlotsX, StatsPlots
using Unitful
import Unitful: ns

# df = collect_results(datadir("benchmarks/track"))
# df = collect_results(datadir("benchmarks/kernel"))
df = collect_results(datadir("benchmarks/kernel/test"))

plot_min_exec_time(df)
plot_min_exec_time(df, num_ants = 16, num_correlators = 7)