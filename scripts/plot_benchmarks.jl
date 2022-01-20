using DrWatson, PrettyTables, Query, DataFrames
@quickactivate "GPUAcceleratedTracking"

# Generate params for signals
using GPUAcceleratedTracking, DataFrames, Plots, PGFPlotsX

# df = collect_results(datadir("benchmarks/track"))
# df = collect_results(datadir("benchmarks/kernel"))
# df = collect_results(datadir("benchmarks/kernel/test"))
# df = collect_results(datadir("benchmarks/kernel/jetson"))
raw_data_df = collect_results(datadir("benchmarks/kernel/kernelnaming1"))

## plot comparison of all kernels sweeping num_samples
# filter the needed data and sort by ascending num_samples
elapsed_min_times_gpu_df = raw_data_df |> 
    @filter(
        _.processor         == "GPU"        &&
        _.os                == "windows"    &&
        _.num_ants          == 1            &&
        _.num_correlators   == 3            
        ) |>
    @map(
        {
            _.processor,
            _.num_samples,
            _.algorithm,
            _.Minimum,
        }
    ) |>
    DataFrame
elapsed_min_times_cpu_df = raw_data |> 
@filter(
    _.processor         == "CPU"        &&
    _.os                == "windows"    &&
    _.num_ants          == 1            &&
    _.num_correlators   == 3            
    ) |>
@map(
    {
        _.processor,
        _.num_samples,
        _.algorithm,
        _.Minimum,
    }
) |>
DataFrame
sort!(elapsed_min_times_cpu_df)
sort!(elapsed_min_times_gpu_df)

# get samples and algorithms
samples = unique(Vector{Int64}(elapsed_min_times_gpu_df[!, :num_samples]))
algorithm_names = unique(Vector{String}(elapsed_min_times_gpu_df[!, :algorithm]))

# put gpu data into algorithms and samples matrix
elapsed_min_times_gpu = Float64.(elapsed_min_times_gpu_df.Minimum)
elapsed_min_times_gpu = reshape(elapsed_min_times_gpu, (length(algorithms), length(samples)))

# put cpu data into matrix
elapsed_min_times_cpu = transpose(Float64.(elapsed_min_times_cpu_df.Minimum))

# define y-axis matrix
data = transpose([elapsed_min_times_gpu; elapsed_min_times_cpu]) 
data *= 10 ^ (-9) # convert to s
yline = range(10 ^ (-3), 10 ^ (-3), length(samples)) # line showing real time execution bound

# xs
xs = samples

# labeling
# labels = ["1" "2" "3" "4" "5" "CPU"]
labels = [permutedims(algorithm_names) "CPU"]

# colors
colors = distinguishable_colors(size(data, 1), [RGB(1,1,1), RGB(0,0,0)], dropseed = true)

# metadata
cpu_name = unique((raw_data_df[!, :CPU_model]))[1] # no need for indexing in the future
gpu_name = unique((raw_data_df[!, :GPU_model]))[2] # no need for indexing in the future

plot(
    xs,
    data,
    title = "Elapsed time", #on $(gpu_name) and $(cpu_name)",
    label = labels,
    legend = :bottomright,
    yaxis = (
        "Elapsed Time [s]",
        :log10,
        :grid = :all
    ),
    xaxis = (
        "Number of samples"
    ),
)