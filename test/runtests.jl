using
    Test,
    Tracking,
    GNSSSignals,
    CUDA,
    StructArrays

import Tracking: Hz, ms

if CUDA.functional()
    include("algorithms.jl")
    include("reduction.jl")
end