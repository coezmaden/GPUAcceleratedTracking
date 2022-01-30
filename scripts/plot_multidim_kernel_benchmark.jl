times_us = [
    299.211,
    589.400,
    1162,
    2302,
    4874,
    9065,
    18085,
    36113
]
times_s = times_us .* 10^(-6)
freqs = (2 .^ (11:18)) / 0.001 


using CairoMakie;

fig = Figure(
    resolution = (1000, 700),
    font = "Times New Roman"
)
ax = Axis(
    fig,
    xlabel = "Sampling Frequency [Hz]",
    ylabel = "Processing Time [s]",
    xscale = log10,
    yscale = log10,
    xmajorgridvisible = true,
    xminorgridvisible = true,
    xminorticksvisible = true,
    xminorticks = IntervalsBetween(9),
    xticks = collect(10 .^ (6:9)),
    title = "Multi-Dimensional Kernel"
)