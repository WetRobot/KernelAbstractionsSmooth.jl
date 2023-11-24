module KernelAbstractionsSmooth
    using KernelAbstractions
    ka = KernelAbstractions
    include("SmoothUtils.jl")
    su = SmoothUtils
    include("LogSumExp.jl")
    lsexp = LogSumExp

    function default_ulw_fit_x_i_eval_x_j(
        fit_x_j::Real,
        eval_x_i::Real,
        theta::Real
    )
        out = (eval_x_i - fit_x_j)^2
        out /= theta
        out *= -1
        return(out)
    end

    function compute_eval_y_hat_i(
        ulw_i_j::Function,
        i::Integer,
        J::Integer,
        get_fit_y_j::Function,
        eval_y_hat::AbstractVector, 
        theta::Real
    )::Nothing
        w_i = ulw_i_j.(i, 1:J, theta)
        eval_y_hat_i = 0.0
        lse = lsexp.log_sum_exp(w_i)
        for j in 1:J
            eval_y_hat_i += get_fit_y_j(j) * exp(w_i[j] - lse)
        end
        eval_y_hat[i] = eval_y_hat_i
        nothing
    end

    ka.@kernel function ka_smoothing_kernel(
        ulw_i_j::Function,
        J::Integer,
        get_fit_y_j::Function,
        eval_y_hat::AbstractVector, 
        theta::Real
    )::Nothing
        i = @index(Global)
        compute_eval_y_hat_i(ulw_i_j, i, J, get_fit_y_j, eval_y_hat, theta)
        nothing
    end

    function ka_smooth!(
        ulw_i_j::Function,
        J::Integer,
        get_fit_y_j::Function,
        eval_y_hat::AbstractVector, 
        theta::Real,
        n_jobs::Int
    )::Nothing
        ka_backend = ka.get_backend(eval_y_hat)
        ka_n_elems_per_job = length(eval_y_hat) ÷ n_jobs
        k! = ka_smoothing_kernel(
            ka_backend,
            ka_n_elems_per_job
        )
        k!(
            ulw_i_j, 
            J, 
            get_fit_y_j,
            eval_y_hat, 
            theta,
            ndrange = length(eval_y_hat)
        )
        synchronize(ka_backend)
        return(nothing)
    end 

    function ka_smooth!(
        ulw_fit_x_i_eval_x_j::Function,
        fit_x::AbstractVector, 
        fit_y::AbstractVector, 
        eval_x::AbstractVector, 
        eval_y_hat::AbstractVector, 
        theta::Real,
        n_jobs::Int
    )::Nothing
        function ulw_i_j(i::Integer, j::Integer, theta::Real)::Real
            return ulw_fit_x_i_eval_x_j(
                su.get_ith_slice_from_first_dim(fit_x, i),
                su.get_ith_slice_from_first_dim(eval_x, j),
                theta
            )
        end
        function get_fit_y_j(i::Integer)::Union{Real, AbstractVector}
            return su.get_ith_slice_from_first_dim(fit_y, i)
        end
        ka_smooth!(
            ulw_i_j,
            length(fit_y),
            get_fit_y_j,
            eval_y_hat, 
            theta,
            n_jobs
        )
        nothing
    end

end
