module KernelAbstractionsSmooth

    using KernelAbstractions
    ka = KernelAbstractions

    function log_sum_exp(x::AbstractVector)::Real
        x_max = maximum(x)
        out = zero(eltype(x))
        for i in eachindex(x)
            out += exp(x[i] - x_max)
        end
        out = x_max + log(out)
        return(out)
    end

    function default_ulp_callback(eval_x_i::Real, fit_x_j::Real, theta::Real)
        out = (eval_x_i - fit_x_j)^2
        out /= theta
        out *= -1
        return(out)
    end

    ka.@kernel function smoothing_kernel!(    
        eval_x::AbstractVector, 
        eval_y_hat::AbstractVector, 
        fit_x::AbstractVector, 
        fit_y::AbstractVector, 
        theta::Real,
        ulp_callback::Function
    )
        i = @index(Global)
        w = ulp_callback.(eval_x[i], fit_x, theta)
        eval_y_hat_i = 0.0
        lse = log_sum_exp(w)
        for j in eachindex(fit_y)
            eval_y_hat_i += fit_y[j] * exp(w[j] - lse)
        end
        eval_y_hat[i] = eval_y_hat_i
        nothing
    end

    function smooth!(  
        eval_x::AbstractVector, 
        eval_y_hat::AbstractVector, 
        fit_x::AbstractVector, 
        fit_y::AbstractVector, 
        theta::AbstractFloat,
        ulp_callback::Function,
        ka_n_work_groups::Int = 1
    )::Nothing
        ka_device = ka.get_device(eval_y_hat)
        ka_n_work_items_per_group = length(eval_y_hat) ÷ ka_n_work_groups
        kernel! = smoothing_kernel!(ka_device, ka_n_work_items_per_group)
        event = kernel!(
            eval_x, 
            eval_y_hat, 
            fit_x, 
            fit_y, 
            theta,
            ulp_callback,
            ndrange = length(eval_y_hat)
        )
        wait(event)
        return(nothing)
    end

end
