@testset "Reduction #3 per Harris" begin
    num_samples = 2500
    num_ants = 1
    num_correlators = 3
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, 2.5e6Hz, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants, num_correlators)),
            CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    threads_per_block = 256
    blocks_per_grid = cld(num_samples, threads_per_block)
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators)),
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators))
        )
    )
    shmem_size = sizeof(Float32) * threads_per_block
    for corr_idx = 1:num_correlators
        # re samples
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_3(
            view(accum.re, :, :, corr_idx),
            view(input.re, :, :, corr_idx),
            num_samples
        )
        # im samples
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_3(
            view(accum.im, :, :, corr_idx),
            view(input.im, :, :, corr_idx),
            num_samples
        )
    end
    for corr_idx = 1:num_correlators
        # re samples
        @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_3(
            view(accum.re, :, :, corr_idx),
            view(accum.re, :, :, corr_idx),
            size(accum, 1)
        )
        # im samples
        @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_3(
            view(accum.im, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            size(accum, 1)
        )
    end
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    # @benchmark CUDA.@sync begin
    #     for corr_idx = 1:$num_correlators
    #         # re samples
    #         @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_3(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(input.re, :, :, corr_idx),
    #             $num_samples
    #         )
    #         # im samples
    #         @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_3(
    #             $view(accum.im, :, :, corr_idx),
    #             $view(input.im, :, :, corr_idx),
    #             $num_samples
    #         )
    #     end
    #     for corr_idx = 1:$num_correlators
    #         # re samples
    #         @cuda threads=$threads_per_block blocks=1 shmem=$shmem_size $reduce_3(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(accum.re, :, :, corr_idx),
    #             $size(accum, 1)
    #         )
    #         # im samples
    #         @cuda threads=$threads_per_block blocks=1 shmem=$shmem_size $reduce_3(
    #             $view(accum.im, :, :, corr_idx),
    #             $view(accum.im, :, :, corr_idx),
    #             $size(accum, 1)
    #         )
    #     end
    # end
end

@testset "Complex Reduction #3 per Harris" begin
    num_samples = 2500
    num_ants = 1
    num_correlators = 3
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, 2.5e6Hz, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants, num_correlators)),
            CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    threads_per_block = 256
    blocks_per_grid = cld(num_samples, threads_per_block)
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators)),
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators))
        )
    )
    shmem_size = sizeof(ComplexF32) * threads_per_block
    for corr_idx = 1:num_correlators
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_cplx_3(
            view(accum.re, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            view(input.re, :, :, corr_idx),
            view(input.im, :, :, corr_idx),
            num_samples
        )
    end
    for corr_idx = 1:num_correlators
        @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_cplx_3(
            view(accum.re, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            view(accum.re, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            size(accum, 1)
        )
    end
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    # @benchmark CUDA.@sync begin
    #     for corr_idx = 1:$num_correlators
    #         @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_cplx_3(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(accum.im, :, :, corr_idx),
    #             $view(input.re, :, :, corr_idx),
    #             $view(input.im, :, :, corr_idx),
    #             $num_samples
    #         )
    #     end
    #     for corr_idx = 1:$num_correlators
    #         @cuda threads=$threads_per_block blocks=1 shmem=$shmem_size $reduce_cplx_3(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(accum.im, :, :, corr_idx),
    #             $view(accum.re, :, :, corr_idx),
    #             $view(accum.im, :, :, corr_idx),
    #             $size(accum, 1)
    #         )
    #     end
    # end
end

# @testset "Complex Multi Reduction #3 per Harris" begin
    num_samples = 2500
    num_ants = 1
    num_correlators = 3
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, 2.5e6Hz, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants, num_correlators)),
            CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    threads_per_block = 256
    blocks_per_grid = cld(num_samples, threads_per_block)
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators)),
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators))
        )
    )
    shmem_size = sizeof(ComplexF32) * threads_per_block * num_ants * num_correlators
    @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_cplx_multi_31(
        accum.re,
        accum.im,
        input.re,
        input.im,
        num_samples,
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_cplx_multi_31(
        accum.re,
        accum.im,
        accum.re,
        accum.im,
        size(accum, 1),
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    # @benchmark CUDA.@sync begin
    #     @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_cplx_multi_3(
    #         $accum.re,
    #         $accum.im,
    #         $input.re,
    #         $input.im,
    #         $num_samples,
    #         $NumAnts(num_ants),
    #         $correlator_sample_shifts
    #     )
    #     @cuda threads=$threads_per_block blocks=1 shmem=$shmem_size $reduce_cplx_multi_3(
    #         $accum.re,
    #         $accum.im,
    #         $accum.re,
    #         $accum.im,
    #         $size(accum, 1),
    #         $NumAnts(num_ants),
    #         $correlator_sample_shifts
    #     )
    # end
end

@testset "Reduction #4 per Harris" begin
    num_samples = 2500
    num_ants = 1
    num_correlators = 3
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, 2.5e6Hz, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants, num_correlators)),
            CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    threads_per_block = 256
    # only half the grid size for reduce_4
    blocks_per_grid = cld(num_samples, threads_per_block) ÷ 2
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators)),
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators))
        )
    )
    shmem_size = sizeof(Float32) * threads_per_block
    for corr_idx = 1:num_correlators
        # re samples
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_4(
            view(accum.re, :, :, corr_idx),
            view(input.re, :, :, corr_idx),
            num_samples
        )
        # im samples
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_4(
            view(accum.im, :, :, corr_idx),
            view(input.im, :, :, corr_idx),
            num_samples
        )
    end
    for corr_idx = 1:num_correlators
        # re samples
        @cuda threads=threads_per_block÷2 blocks=1 shmem=shmem_size reduce_4(
            view(accum.re, :, :, corr_idx),
            view(accum.re, :, :, corr_idx),
            size(accum, 1)
        )
        # im samples
        @cuda threads=threads_per_block÷2 blocks=1 shmem=shmem_size reduce_4(
            view(accum.im, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            size(accum, 1)
        )
    end
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    # @benchmark CUDA.@sync begin
    #     for corr_idx = 1:$num_correlators
    #         # re samples
    #         @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_4(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(input.re, :, :, corr_idx),
    #             $num_samples
    #         )
    #         # im samples
    #         @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_4(
    #             $view(accum.im, :, :, corr_idx),
    #             $view(input.im, :, :, corr_idx),
    #             $num_samples
    #         )
    #     end
    #     for corr_idx = 1:$num_correlators
    #         # re samples
    #         @cuda threads=$threads_per_block÷2 blocks=1 shmem=$shmem_size $reduce_4(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(accum.re, :, :, corr_idx),
    #             $size(accum, 1)
    #         )
    #         # im samples
    #         @cuda threads=$threads_per_block÷2 blocks=1 shmem=$shmem_size $reduce_4(
    #             $view(accum.im, :, :, corr_idx),
    #             $view(accum.im, :, :, corr_idx),
    #             $size(accum, 1)
    #         )
    #     end
    # end
end

@testset "Complex Reduction #4 per Harris" begin
    num_samples = 2500
    num_ants = 1
    num_correlators = 3
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, 2.5e6Hz, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants, num_correlators)),
            CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    threads_per_block = 256
    # only half the grid size for reduce_4
    blocks_per_grid = cld(num_samples, threads_per_block) ÷ 2
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators)),
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators))
        )
    )
    shmem_size = sizeof(ComplexF32) * threads_per_block
    for corr_idx = 1:num_correlators
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_cplx_4(
            view(accum.re, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            view(input.re, :, :, corr_idx),
            view(input.im, :, :, corr_idx),
        )
    end
    for corr_idx = 1:num_correlators
        @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_cplx_4(
            view(accum.re, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            view(accum.re, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
        )
    end
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    # @benchmark CUDA.@sync begin
    #     for corr_idx = 1:$num_correlators
    #         @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_cplx_4(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(accum.im, :, :, corr_idx),
    #             $view(input.re, :, :, corr_idx),
    #             $view(input.im, :, :, corr_idx),
    #         )
    #     end
    #     for corr_idx = 1:$num_correlators
    #         @cuda threads=$threads_per_block blocks=$1 shmem=$shmem_size $reduce_cplx_4(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(accum.im, :, :, corr_idx),
    #             $view(accum.re, :, :, corr_idx),
    #             $view(accum.im, :, :, corr_idx),
    #         )
    #     end
    # end
end

@testset "Complex Multi Reduction #4 per Harris" begin
    num_samples = 2500
    num_ants = 1
    num_correlators = 3
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, 2.5e6Hz, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants, num_correlators)),
            CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    threads_per_block = 256
    # only half the grid size for reduce_4
    blocks_per_grid = cld(num_samples, threads_per_block) ÷ 2
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators)),
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators))
        )
    )
    shmem_size = sizeof(ComplexF32) * threads_per_block * num_ants * num_correlators
    @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_cplx_multi_4(
        accum.re,
        accum.im,
        input.re,
        input.im,
        num_samples,
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_cplx_multi_4(
        accum.re,
        accum.im,
        accum.re,
        accum.im,
        blocks_per_grid,
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    # @benchmark CUDA.@sync begin
    #     @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size reduce_cplx_multi_4(
    #         $accum.re,
    #         $accum.im,
    #         $input.re,
    #         $input.im,
    #         $num_samples,
    #         $NumAnts(num_ants),
    #         $correlator_sample_shifts
    #     )
    #     @cuda threads=$threads_per_block blocks=1 shmem=$shmem_size reduce_cplx_multi_4(
    #         $accum.re,
    #         $accum.im,
    #         $accum.re,
    #         $accum.im,
    #         $blocks_per_grid,
    #         $NumAnts(num_ants),
    #         $correlator_sample_shifts
    #     )
    # end
end

@testset "Reduction #5 per Harris" begin
    num_samples = 2048
    num_ants = 1
    num_correlators = 3
    signal_duration = 0.001
    sampling_frequency = (num_samples/signal_duration)Hz
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, sampling_frequency, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants, num_correlators)),
            CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    threads_per_block = 256
    # only half the grid size for reduce_4
    blocks_per_grid = cld(cld(num_samples, threads_per_block), 2)
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators)),
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators))
        )
    )
    shmem_size = sizeof(Float32) * threads_per_block
    for corr_idx = 1:num_correlators
        # re samples
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_5(
            view(accum.re, :, :, corr_idx),
            view(input.re, :, :, corr_idx),
            num_samples
        )
        # im samples
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_5(
            view(accum.im, :, :, corr_idx),
            view(input.im, :, :, corr_idx),
            num_samples
        )
    end
    for corr_idx = 1:num_correlators
        # re samples
        @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_5(
            view(accum.re, :, :, corr_idx),
            view(accum.re, :, :, corr_idx),
            size(accum, 1)
        )
        # im samples
        @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_5(
            view(accum.im, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            size(accum, 1)
        )
    end
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    # @benchmark CUDA.@sync begin
    #     for corr_idx = 1:$num_correlators
    #         # re samples
    #         @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_5(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(input.re, :, :, corr_idx),
    #             $num_samples
    #         )
    #         # im samples
    #         @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_5(
    #             $view(accum.im, :, :, corr_idx),
    #             $view(input.im, :, :, corr_idx),
    #             $num_samples
    #         )
    #     end
    #     for corr_idx = 1:$num_correlators
    #         # re samples
    #         @cuda threads=$threads_per_block blocks=1 shmem=$shmem_size $reduce_5(
    #             $view(accum.re, :, :, corr_idx),
    #             $view(accum.re, :, :, corr_idx),
    #             $size(accum, 1)
    #         )
    #         # im samples
    #         @cuda threads=$threads_per_block blocks=1 shmem=$shmem_size $reduce_5(
    #             $view(accum.im, :, :, corr_idx),
    #             $view(accum.im, :, :, corr_idx),
    #             $size(accum, 1)
    #         )
    #     end
    # end
end

@testset "Complex Reduction #5 per Harris" begin
    num_samples = 2048
    num_ants = 1
    num_correlators = 3
    signal_duration = 0.001
    sampling_frequency = (num_samples/signal_duration)Hz
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, sampling_frequency, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants, num_correlators)),
            CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    threads_per_block = 256
    # only half the grid size for reduce_5
    blocks_per_grid = cld(cld(num_samples, threads_per_block), 2)
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators)),
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators))
        )
    )
    shmem_size = sizeof(ComplexF32) * threads_per_block
    for corr_idx = 1:num_correlators
        @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_cplx_5(
            view(accum.re, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            view(input.re, :, :, corr_idx),
            view(input.im, :, :, corr_idx),
            num_samples
        )
    end
    Array(accum)
    for corr_idx = 1:num_correlators
        @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_cplx_5(
            view(accum.re, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            view(accum.re, :, :, corr_idx),
            view(accum.im, :, :, corr_idx),
            blocks_per_grid
        )
    end
    Array(accum)
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    @benchmark CUDA.@sync begin
        for corr_idx = 1:$num_correlators
            @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_cplx_5(
                $view($accum.re, :, :, corr_idx),
                $view($accum.im, :, :, corr_idx),
                $view($input.re, :, :, corr_idx),
                $view($input.im, :, :, corr_idx),
                $num_samples
            )
        end
        for corr_idx = 1:$num_correlators
            @cuda threads=$threads_per_block blocks=$1 shmem=$shmem_size $reduce_cplx_5(
                $view($accum.re, :, :, corr_idx),
                $view($accum.im, :, :, corr_idx),
                $view($accum.re, :, :, corr_idx),
                $view($accum.im, :, :, corr_idx),
                $blocks_per_grid
            )
        end
    end
end

@testset "Complex Multi Reduction #5 per Harris" begin
    num_samples = 2048
    num_ants = 1
    num_correlators = 3
    signal_duration = 0.001
    sampling_frequency = (num_samples/signal_duration)Hz
    correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, sampling_frequency, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants, num_correlators)),
            CUDA.zeros(Float32, (num_samples, num_ants, num_correlators))
        )
    )
    threads_per_block = 256
    # only half the grid size for reduce_5
    blocks_per_grid = cld(cld(num_samples, threads_per_block), 2)
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators)),
            CUDA.zeros(Float32, (blocks_per_grid, num_ants, num_correlators))
        )
    )
    shmem_size = sizeof(ComplexF32) * threads_per_block * num_ants * num_correlators
    @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_cplx_multi_5(
        accum.re,
        accum.im,
        input.re,
        input.im,
        num_samples,
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    Array(accum)
    @cuda threads=threads_per_block blocks=1 shmem=shmem_size reduce_cplx_multi_5(
        accum.re,
        accum.im,
        accum.re,
        accum.im,
        blocks_per_grid,
        NumAnts(num_ants),
        correlator_sample_shifts
    )
    Array(accum)
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    # @benchmark CUDA.@sync begin
    #     @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size reduce_cplx_multi_4(
    #         $accum.re,
    #         $accum.im,
    #         $input.re,
    #         $input.im,
    #         $num_samples,
    #         $NumAnts(num_ants),
    #         $correlator_sample_shifts
    #     )
    #     @cuda threads=$threads_per_block blocks=1 shmem=$shmem_size reduce_cplx_multi_4(
    #         $accum.re,
    #         $accum.im,
    #         $accum.re,
    #         $accum.im,
    #         $blocks_per_grid,
    #         $NumAnts(num_ants),
    #         $correlator_sample_shifts
    #     )
    # end
end

# @testset "Complex Multi NANT Reduction #5 per Harris" begin
    num_samples = 2 ^ 15
    num_ants = 16
    num_correlators = 1
    signal_duration = 0.001
    sampling_frequency = (num_samples/signal_duration)Hz
    # correlator = EarlyPromptLateCorrelator(NumAnts(num_ants), NumAccumulators(num_correlators))
    # correlator_sample_shifts = get_correlator_sample_shifts(GPSL1(), correlator, sampling_frequency, 0.5)
    input = StructArray{ComplexF32}(
        (
            CUDA.ones(Float32, (num_samples, num_ants)),
            CUDA.zeros(Float32, (num_samples, num_ants))
        )
    )
    threads_per_block = 1024
    # only half the grid size for reduce_5
    blocks_per_grid_x = cld(cld(num_samples, threads_per_block), 2)
    blocks_per_grid_y = num_ants
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    accum = StructArray{ComplexF32}(
        (
            CUDA.zeros(Float32, (blocks_per_grid_x, blocks_per_grid_y)),
            CUDA.zeros(Float32, (blocks_per_grid_x, blocks_per_grid_y))
        )
    )
    shmem_size = sizeof(ComplexF32) * threads_per_block
    
    @cuda threads=threads_per_block blocks=blocks_per_grid shmem=shmem_size reduce_cplx_multi_nant_5(
        accum.re,
        accum.im,
        input.re,
        input.im,
        num_samples,
        NumAnts(num_ants),
    )
    Array(accum)
    @cuda threads=threads_per_block blocks=(1, blocks_per_grid_y) shmem=shmem_size reduce_cplx_multi_nant_5(
        accum.re,
        accum.im,
        accum.re,
        accum.im,
        blocks_per_grid_x,
        NumAnts(num_ants)
    )
    Array(accum)
    CUDA.@allowscalar begin
        accum_true = ComplexF32[num_samples num_samples num_samples]
        @test Array(accum)[1, :, :,] ≈ accum_true
    end
    @benchmark CUDA.@sync begin
        @cuda threads=$threads_per_block blocks=$blocks_per_grid shmem=$shmem_size $reduce_cplx_multi_nant_5(
            $accum.re,
            $accum.im,
            $input.re,
            $input.im,
            $num_samples,
            $NumAnts(num_ants),
        )
        @cuda threads=$threads_per_block blocks=(1, $blocks_per_grid_y) shmem=$shmem_size $reduce_cplx_multi_nant_5(
            $accum.re,
            $accum.im,
            $accum.re,
            $accum.im,
            $blocks_per_grid_x,
            $NumAnts(num_ants)
        )
    end
end