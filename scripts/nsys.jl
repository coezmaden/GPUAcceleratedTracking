# nsys launch julia /path/to/nsys.jl

using GPUAcceleratedTracking, CUDA, Tracking, GNSSSignals
import Tracking: Hz

system = GPSL1(use_gpu = Val(true))
signal, sampling_frequency = gen_signal(system, 1, 1500Hz, 2500)
state = TrackingState(1, system, 1500Hz, 0, num_samples = 2500)
CUDA.@profile track(signal, state, sampling_frequency)