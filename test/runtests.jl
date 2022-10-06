import KernelAbstractionsSmooth
kas = KernelAbstractionsSmooth
using Test

@testset "KernelAbstractionsSmooth.jl" begin
    x = 1.0:1.0:10.0
    y = 10.0:(-1.0):1.0
    y_hat = zeros(length(y)) .- 999

    kas.smooth!(  
        x, 
        y_hat, 
        x, 
        y, 
        1.0,
        kas.default_ulp_callback
    )
    @show(y_hat)
    @test y_hat[1] < 10.5
    @test y_hat[1] > 9.5
end
