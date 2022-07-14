# Tracking.jl: Accelerating multi-antenna GNSS receivers with CUDA


[![DOI](https://zenodo.org/badge/438278321.svg)](https://zenodo.org/badge/latestdoi/438278321)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5933726.svg)](https://doi.org/10.5281/zenodo.5933726)

This repository contains the source code for the paper *"Tracking.jl: Accelerating multi-antenna GNSS receivers with CUDA"*. It is submitted to be published in the [JuliaCon Proceedings](https://proceedings.juliacon.org/).
![desktop_allplots](https://user-images.githubusercontent.com/33359548/151870105-af00d1da-38bf-4e0d-aa53-7aa3c3b5a2e9.svg)

Scripts reproducing the benchmarks and figures can be found under `/scripts`, the algorithms source code under `/src`. Paper itself resides under `/paper`. 

## Data
You can download the raw data from the experiments on the two platforms specified in the paper [here](https://zenodo.org/record/5933726). 
The dataset is identified witha a DOI: 10.5281/zenodo.5933726

## How to use this repository
This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> GPUAcceleratedTracking

It is authored by Can Özmaden.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

## Algorithms Description
![algorithms](https://user-images.githubusercontent.com/33359548/151870213-4a71e15a-b288-4a75-ba19-e435a9b5296e.svg)

###  NVIDIA Jetson AGX Xavier
![jetson_allplots](https://user-images.githubusercontent.com/33359548/151870294-15d61608-64e9-4271-8ea7-e30c3af64fea.svg)

## Acknowledgements
The following packages have played a crucial role during the preparation of this paper:
* [JuliaGNSS](https://github.com/JuliaGNSS): [Tracking.jl](https://github.com/JuliaGNSS/Tracking.jl), [GNSSSignals.jl](https://github.com/JuliaGNSS/GNSSSignals.jl)
* GPU Programming: [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
* Data analysis and visualization: [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl), [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl), [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl), [Makie.jl](https://github.com/JuliaPlots/Makie.jl)
