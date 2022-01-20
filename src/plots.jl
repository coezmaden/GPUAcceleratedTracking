function plot_min_exec_time(raw_data_df::DataFrame; num_ants = 1, num_correlators = 3, os = "windows")
    elapsed_min_times_gpu_df = raw_data_df |> 
        @filter(
            _.processor         == "GPU"            &&
            _.os                == os               &&
            _.num_ants          == num_ants         &&
            _.num_correlators   == num_correlators    
            ) |>
        @map(
            {
                _.processor,
                _.num_samples,
                _.algorithm,
                _.Minimum,
            }
        ) |> DataFrame
    elapsed_min_times_cpu_df = raw_data_df |> 
        @filter(
            _.processor         == "CPU"            &&
            _.os                == "windows"        &&
            _.num_ants          == num_ants         &&
            _.num_correlators   == num_correlators            
            ) |>
        @map(
            {
                _.processor,
                _.num_samples,
                _.algorithm,
                _.Minimum,
            }
        ) |> DataFrame
    sort!(elapsed_min_times_cpu_df)
    sort!(elapsed_min_times_gpu_df)

    # get samples and algorithms
    samples = unique(Vector{Int64}(elapsed_min_times_gpu_df[!, :num_samples]))
    algorithm_names = unique(Vector{String}(elapsed_min_times_gpu_df[!, :algorithm]))

    # put gpu data into algorithms and samples matrix
    elapsed_min_times_gpu = Float64.(elapsed_min_times_gpu_df.Minimum)
    elapsed_min_times_gpu = reshape(elapsed_min_times_gpu, (length(algorithm_names), length(samples)))

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
    # cpu_name = unique((raw_data_df[!, :CPU_model]))[1] # no need for indexing in the future
    # gpu_name = unique((raw_data_df[!, :GPU_model]))[2] # no need for indexing in the future

    plot(
        xs,
        data,
        title = "Elapsed time", #on $(gpu_name) and $(cpu_name)",
        label = labels,
        legend = :bottomright,
        yaxis = (
            "Elapsed Time [s]",
            :log10,
            :grid,
        ),
        xaxis = (
            "Number of samples"
        ),
    )
end

function plot_min_exec_time_gpu(df::DataFrame; num_ants = 1, num_correlators = 3)
    gpudf = df |> @filter(_.processor=="GPU") |> @filter(_.num_ants==num_ants) |> @filter(_.num_correlators==num_correlators) |> @map({_.num_samples, _.Minimum}) |> DataFrame
    cpudf = df |> @filter(_.processor=="CPU") |> @filter(_.num_ants==num_ants) |> @filter(_.num_correlators==num_correlators) |> @map({_.num_samples, _.Minimum}) |> DataFrame

    num_samples = unique(Vector{Int64}(gpudf[!, :num_samples]))
    # num_ants = unique(Vector{Int64}(gpudf[!, :num_ants]))
    # num_correlators = unique(Vector{Int64}(gpudf[!, :num_correlators]))
    
    # gputimes = Array{Float64}(undef, (length(num_samples), length(num_ants), length(num_correlators)))
   
    gputimes = Vector{Float64}(gpudf[!, :Minimum])
    cputimes = Vector{Float64}(cpudf[!, :Minimum])

    plot(num_samples, 
        [gputimes, cputimes],
        title = "GPU vs CPU: $(num_ants) Antenna, $(num_correlators) Correlators",
        label = ["GPU" "CPU"],
        ylabel = "Execution Time",
        xaxis = ("Number of Samples", :plain)
    )
end

# function plot_pgf_kernel_comparison(
#     df::DataFrame;
#     num_ants = 1,
#     num_correlators = 3;
# )
#     gpudf = df |> @filter(_.processor=="GPU") |> @filter(_.num_ants==num_ants) |> @filter(_.num_correlators==num_correlators) |> @map({_.num_samples, _.Minimum}) |> DataFrame
#     cpudf = df |> @filter(_.processor=="CPU") |> @filter(_.num_ants==num_ants) |> @filter(_.num_correlators==num_correlators) |> @map({_.num_samples, _.Minimum}) |> DataFrame

#     num_samples = unique(Vector{Int64}(gpudf[!, :num_samples]))

#     gputimes = Vector{Float64}(gpudf[!, :Minimum])
#     cputimes = Vector{Float64}(cpudf[!, :Minimum])

#     pgfplot = @pgf TikzPicture(
#         Axis(
#             {
#                 xlabel = "Samples",
#                 ylabel = "Time [μs]",
#                 ymode = "log",
#                 title = "Execution Time",
#                 xmajorgrids,
#                 ymajorgrids,
#                 scaled_ticks = "false"
#             },
#             PlotInc(
#                 {
#                     gputimes
#                 }
#             )
#         )
#     )
# end

# function plot_kernel_comparison(df::DataFrame; num_ants = 1, num_correlators = 3)
#     # Filter out CPU and GPU data
#     gpudf = df |> @filter(_.processor=="GPU") |> @filter(_.num_ants==num_ants) |> @filter(_.num_correlators==num_correlators) |> @map({_.num_samples, _.algorithm, _.Minimum}) |> @orderby(({_.num_samples})) |> DataFrame
#     cpudf = df |> @filter(_.processor=="CPU") |> @filter(_.num_ants==num_ants) |> @filter(_.num_correlators==num_correlators) |> @map({_.num_samples, _.Minimum}) |> DataFrame

#     algorithms = unique(Vector{Int64}(gpudf[!, :algorithm]))
#     samples = unique(Vector{Int64}(gpudf[!, :num_samples]))

#     gputimes = zeros(Float64, (length(algorithms), length(samples)))
#     for i = 1:length(algorithms)
#         gpualgodf = gpudf |> @filter(_.algorithm==i) |> DataFrame
#         gputimes[i, :] = gpualgodf[!, :Minimum]
#     end
#     cputimes = Vector{Float64}(cpudf[!, :Minimum])

#     data = [transpose(cputimes); gputimes]
#     labels = ["CPU" "GPU #1" "GPU #2" "GPU #3" "GPU #4" "GPU #5" "GPU #6" "GPU #7"]
#     markershapes = [:circle :diamond :octagon :heptagon :pentagon :rect :utriangle]

#     plot(
#        samples,
#        transpose(data),
#        label = labels
#        #shape = markershapes,
#     )
#     xaxis!("Number of Samples")
#     yaxis!("Execution Time [ns]", :log10)
# end

