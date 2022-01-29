using GPUAcceleratedTracking, DrWatson, DataFrames
@quickactivate "GPUAcceleratedTracking"

# raw_data_df = collect_results(datadir("benchmarks/track"))
# raw_data_df = collect_results(datadir("benchmarks/kernel"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/test"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/jetson"))
raw_data_df = collect_results(datadir("benchmarks/kernel/kernelnaming1"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/cuda-elapsed"))
# raw_data_df = collect_results(datadir("benchmarks/kernel/belapsed"))
os ="windows"
gnss="GPSL5"
# os ="windows"
## plot comparison of all kernels sweeping num_samples
# using Plots; gr()
##
processor = "CPUvGPU"; num_ants = 1; num_correlators = 3; OS=os; GNSS=gnss;
params = @dict processor num_ants num_correlators OS GNSS

using Query;CairoMakie; Makie;

gnss_string = Dict(
    "GPSL1" => "GPS L1 C/A",
    "GPSL5" => "GPS L5 P"
)

begin
    fig = Figure(
        resolution = (1000, 700),
        font = "Times New Roman"
    )
    elapsed_min_times_gpu_df = raw_data_df |> 
    @filter(
        _.processor         == "GPU"            &&
        _.os                == os               &&
        _.num_ants          == num_ants         &&
        _.num_correlators   == num_correlators    
        ) |>
    @map(
        {
            _.num_samples,
            _.algorithm,
            _.Minimum,
            _.GNSS
        }
    ) |> DataFrame


    sort!(elapsed_min_times_gpu_df)

    samples = unique(Vector{Int64}(elapsed_min_times_gpu_df[!, :num_samples]))
    x = samples / 0.001
    algorithm_names = unique(Vector{String}(elapsed_min_times_gpu_df[!, :algorithm])
    )
    algorithm_names = algorithm_names[2:end] #disregard 1 alg

    # PLOT GPS L1 C/A
    os = "windows"
    gnss = "GPSL1"
    num_ants = 1; num_correlators = 3

    gpsl1_1ms_1_3_ax = Axis(fig, 
        xlabel = "Sampling Frequency [Hz]",
        ylabel = "Processing Time [s]",
        xscale = log10,
        yscale = log10,
        xmajorgridvisible = true,
        xminorgridvisible = true,
        xminorticksvisible = true,
        xminorticks = IntervalsBetween(9),
        xticks = collect(10 .^ (6:9)),
        # yticks = collect(10.0 .^ (-6:-2)),
        title = "$(gnss_string[gnss]) T=1 ms, M=$num_ants, L=$num_correlators"
    )
    xlims!(gpsl1_1ms_1_3_ax, [10^(6),  3 * 10^(8)])
    # ylims!(gpsl1_1ms_1_3_ax, [10^(-6), 2*10^(-3)])
    markers = [:rect, :utriangle, :circle, :diamond, :dtriangle]
    lin = Array{Lines}(undef, length(algorithm_names)); 
    sca = Array{Scatter}(undef, length(algorithm_names));
    for (idx, algorithm) in enumerate(algorithm_names)
        times = Float64.((elapsed_min_times_gpu_df |> @filter(_.algorithm == algorithm_names[idx] && _.GNSS == gnss) |> @map({_.Minimum})|> DataFrame).Minimum)
        lin[idx] = lines!(
            gpsl1_1ms_1_3_ax, 
            x, 
            times .* 10 ^ (-9)
        )
        sca[idx] = scatter!(
            gpsl1_1ms_1_3_ax, 
            x, 
            times .* 10 ^ (-9),
            marker = markers[idx],
            markersize = 15
        )
    end
    fig[1,1] = gpsl1_1ms_1_3_ax


    # PLOT GPS L5 C/A
    os = "windows"
    gnss = "GPSL5"
    num_ants = 1; num_correlators = 3

    gpsl5_1ms_1_3_ax = Axis(fig, 
        xlabel = "Sampling Frequency [Hz]",
        ylabel = "Processing Time [s]",
        xscale = log10,
        yscale = log10,
        xmajorgridvisible = true,
        xminorgridvisible = true,
        xminorticksvisible = true,
        xminorticks = IntervalsBetween(9),
        xticks = collect(10 .^ (6:9)),
        # yticks = collect(10.0 .^ (-6:-2)),
        title = "$(gnss_string[gnss]) T=1 ms, M=$num_ants, L=$num_correlators"
    )
    xlims!(gpsl5_1ms_1_3_ax, [10^(6), 3 * 10^(8)])
    # ylims!(gpsl5_1ms_1_3_ax,[1e-6, 5e-3])
    markers = [:rect, :utriangle, :circle, :diamond, :dtriangle]
    lin = Array{Lines}(undef, length(algorithm_names)); 
    sca = Array{Scatter}(undef, length(algorithm_names));
    for (idx, algorithm) in enumerate(algorithm_names)
        times = Float64.((elapsed_min_times_gpu_df |> @filter(_.algorithm == algorithm_names[idx] && _.GNSS == gnss) |> @map({_.Minimum})|> DataFrame).Minimum)
        lin[idx] = lines!(
            gpsl5_1ms_1_3_ax, 
            x, 
            times .* 10 ^ (-9)
        )
        sca[idx] = scatter!(
            gpsl5_1ms_1_3_ax, 
            x, 
            times .* 10 ^ (-9),
            marker = markers[idx],
            markersize = 15
        )
    end
    realtime = lines!(gpsl5_1ms_1_3_ax, [10^(6), 3 * 10^(8)], [10^(-3), 10^(-3)], linestyle = :dash )
    fig[1,2] = gpsl5_1ms_1_3_ax

    labels = algorithm_names
    # Legend(fig[2,1], lin, algorithm_names, "Reduction Algorithms", framevisible = false)
    # Legend(fig, "Algorithms", algorithm_names)
    fig
end




pl1 = plot_min_exec_time(raw_data_df, os=os, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl1
)
processor = "GPU"; num_ants = 1; num_correlators = 3; OS=os; GNSS=gnss;
params = @dict processor num_ants num_correlators OS GNSS
pl2 = plot_min_exec_time_gpu(raw_data_df, os=os, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl1
)

##
processor = "CPUvGPU"; num_ants = 4; num_correlators = 3; os=os;
params = @dict processor num_ants num_correlators
pl1 = plot_min_exec_time(raw_data_df, os=os, num_ants = 4, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl1
)
processor = "GPU"; num_ants = 4; num_correlators = 3; os=os;
params = @dict processor num_ants num_correlators
pl2 = plot_min_exec_time_gpu(raw_data_df, os=os, num_ants = 4, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl2
)

## 
processor = "CPUvGPU"; num_ants = 1; num_correlators = 7; os=os;
params = @dict processor num_ants num_correlators
pl1 = plot_min_exec_time(raw_data_df, os=os, num_correlators = 7, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl1
)
processor = "GPU"; num_ants = 1; num_correlators = 7; os=os;
params = @dict processor num_ants num_correlators
pl2 = plot_min_exec_time_gpu(raw_data_df, os=os, num_correlators = 7, gnss = gnss)
wsave(
    plotsdir("kernelnaming", savename(params, "png")), 
    pl2
)