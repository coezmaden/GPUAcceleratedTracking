# Comparison of Optimization Strategies for a GPU-Enabled Multi-Antenna Multi-Correlator GNSS SDR Module

This repository contains the source code for the paper *"Comparison of Optimization Strategies for a GPU-Enabled Multi-Antenna Multi-Correlator GNSS SDR Module"* contended for the ION GNSS+ Student Paper Award 2022.

Scripts reproducing the benchmarks and figures can be found under `/scripts`, the algorithms source code under `/src`. 

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


## Kernels Description
