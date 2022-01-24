function plot_min_exec_time(raw_data_df::DataFrame; num_ants = 1, num_correlators = 3, os = "windows", gnss = "GPSL1")
    elapsed_min_times_gpu_df = raw_data_df |> 
        @filter(
            _.processor         == "GPU"            &&
            _.os                == os               &&
            _.GNSS              == gnss             &&
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
            _.os                == os               &&
            _.GNSS              == gnss             &&
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
    xs = samples / 0.001 # convert to Hz

    # labeling
    # labels = ["1" "2" "3" "4" "5" "CPU"]
    labels = [permutedims(algorithm_names) "CPU"]

    # colors
    colors = permutedims(distinguishable_colors(size(data, 1), [RGB(1,1,1), RGB(0,0,0)], dropseed = true))

    # metadata
    # cpu_name = unique((raw_data_df[!, :CPU_model]))[1] # no need for indexing in the future
    # gpu_name = unique((raw_data_df[!, :GPU_model]))[2] # no need for indexing in the future

    pl = plot(
        xs,
        data,
        title = "Correlation processing time of a 1ms $(gnss) signal\nfor $(num_ants) antenna and $(num_correlators) correlators.", #on $(gpu_name) and $(cpu_name)",
        label = labels,
        legend = :bottomright,
        shape = [:circle]
    )
    plot!(size=(1000,600))
    hline!(yline, line = (:dash, :grey))
    yaxis!("Processing Time [s]", :log10, (10^(-6), 10^(-2)), minorgrid=true)
    xaxis!("Sampling Frequency [Hz]", :log10, (10^(6), 2*10^(8)), minorgrid = true)
    return pl
end

function plot_min_exec_time_gpu(raw_data_df::DataFrame; num_ants = 1, num_correlators = 3, os = "windows", gnss="GPSL1")
    elapsed_min_times_gpu_df = raw_data_df |> 
        @filter(
            _.processor         == "GPU"            &&
            _.os                == os               &&
            _.GNSS              == gnss             &&
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
    sort!(elapsed_min_times_gpu_df)

    # get samples and algorithms
    samples = unique(Vector{Int64}(elapsed_min_times_gpu_df[!, :num_samples]))
    algorithm_names = unique(Vector{String}(elapsed_min_times_gpu_df[!, :algorithm]))

    # put gpu data into algorithms and samples matrix
    elapsed_min_times_gpu = Float64.(elapsed_min_times_gpu_df.Minimum)
    elapsed_min_times_gpu = reshape(elapsed_min_times_gpu, (length(algorithm_names), length(samples)))

    # define y-axis matrix
    data = transpose(elapsed_min_times_gpu) 
    data *= 10 ^ (-9) # convert to s
    yline = range(10 ^ (-3), 10 ^ (-3), length(samples)) # line showing real time execution bound

    # xs
    xs = samples / 0.001 # convert to Hz

    # labeling
    labels = permutedims(algorithm_names)

    # colors
    colors = permutedims(distinguishable_colors(size(data, 1), [RGB(1,1,1), RGB(0,0,0)], dropseed = true))

    # markers
    # markers = [:circle :]

    # metadata
    # gpu_name = unique((raw_data_df[!, :GPU_model]))[2] # no need for indexing in the future
    
    pl = plot(
        xs,
        data,
        title = "Correlation processing time of a 1ms $(gnss) signal\nfor $(num_ants) antenna and $(num_correlators) correlators.", #on $(gpu_name) and $(cpu_name)",
        label = labels,
        legend = :bottomright,
        shape = [:circle]
    )
    plot!(size=(1000,600))
    hline!(yline, line = (:dash, :grey))
    yaxis!("Processing Time [s]", :log10, (10^(-6), 10^(-2)), minorgrid=true)
    xaxis!("Sampling Frequency [Hz]", :log10, (10^(6), 2*10^(8)), minorgrid = true)
    return pl
end