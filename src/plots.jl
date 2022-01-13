function plot_min_exec_time(df::DataFrame; num_ants = 1, num_correlators = 3)
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