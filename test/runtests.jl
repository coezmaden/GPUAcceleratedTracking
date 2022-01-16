using
    Test,
    Tracking,
    GNSSSignals

import Tracking: Hz, ms

if CUDA.functional()
    include("algorithms.jl")
end