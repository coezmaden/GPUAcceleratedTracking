# julia --project=. /scripts/nsys.jl
# nsys launch julia /path/to/nsys.jl
using GPUAcceleratedTracking, CUDA, Tracking, GNSSSignals
import Tracking: Hz, ms

system = GPSL1(use_gpu = Val(true))

## 1 ms signal
signal, sampling_frequency = gen_signal(system, 1, 1500Hz, 2500, num_ants = NumAnts(4))
state = TrackingState(1, system, 1500Hz, 0, num_samples = 2500, num_ants = NumAnts(4))
track(signal, state, sampling_frequency)
CUDA.@profile track(signal, state, sampling_frequency)

## 10 ms signal
#= signal, sampling_frequency = gen_signal(system, 1, 1500Hz, 2500*10, duration=10ms)
state = TrackingState(1, system, 1500Hz, 0, num_samples = 2500*10)
track(signal, state, sampling_frequency)
CUDA.@profile track(signal, state, sampling_frequency)
 =#