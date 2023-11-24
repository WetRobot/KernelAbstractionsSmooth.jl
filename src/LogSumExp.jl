module LogSumExp
    using KernelAbstractions
    ka = KernelAbstractions
    @kernel function ka_log_sum_exp_kernel_factory(
        x::AbstractVector,
        x_max::Real,
        group_results::AbstractVector
    )
        i = @index(Global)
        g = @index(Group)
        group_results[g] += exp(x[i] - x_max)
    end

    function ka_log_sum_exp(x::AbstractVector, n_threads::Integer = 1)::Real
        ka_n_work_items_per_group = length(x) ÷ n_threads
        ka_lse! = ka_log_sum_exp_kernel_factory(
            ka.get_backend(x),
            ka_n_work_items_per_group
        )
        x_max = maximum(x)
        group_results = zeros(Float64, n_threads)
        ka_lse!(x, x_max, group_results)
        return(sum(group_results))
    end
    
    function log_sum_exp(x::AbstractVector)::Real
        x_max = maximum(x)
        return x_max + log(sum(x_i -> exp(x_i - x_max), x))
    end
    
end