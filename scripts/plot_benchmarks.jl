using DrWatson, PrettyTables, Query, DataFrames
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, DataFrames, Plots, PGFPlotsX
using Unitful
import Unitful: ns

# df = collect_results(datadir("benchmarks/track"))
# df = collect_results(datadir("benchmarks/kernel"))
# df = collect_results(datadir("benchmarks/kernel/test"))
# df = collect_results(datadir("benchmarks/kernel/jetson"))
df = collect_results(datadir("benchmarks/kernel/kernelnaming1"))

# plot_min_exec_time(df)
# plot_min_exec_time(df, num_ants = 16, num_correlators = 7)

simple_data = df  |> @map({_.processor, _.algorithm, _.num_samples, _.num_ants, _.num_correlators, _.TrialObj}) |> @orderby_descending({_.num_samples}) |> DataFrame
df_pretty = pretty_table(simple_data)