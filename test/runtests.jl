import KernelAbstractionsSmooth
kas = KernelAbstractionsSmooth
using Test

@testset "KernelAbstractionsSmooth.jl" begin
    x = 1.0:1.0:10.0
    y = 10.0:(-1.0):1.0
    y_hat = zeros(length(y)) .- 999

    kas.ka_smooth!(  
        kas.default_ulw_fit_x_i_eval_x_j,
        x,
        y,
        x,
        y_hat, 
        1.0,
        0.10
    )
    @show(y_hat)
    @test y_hat[1] < 10.5
    @test y_hat[1] > 9.5
end
