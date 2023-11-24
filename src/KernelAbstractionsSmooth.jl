module KernelAbstractionsSmooth
    using KernelAbstractions
    ka = KernelAbstractions
    include("SmoothUtils.jl")
    su = SmoothUtils
    include("LogSumExp.jl")
    lsexp = LogSumExp

    function default_ulw_i_j_kernel(
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
        get_fit_y_i::AbstractArray,
        eval_y_hat::AbstractVector, 
        theta::Real
    )::Nothing
        w_i = ulw_i_j.(i, 1:J, theta)
        eval_y_hat_i = 0.0
        lse = lsexp.log_sum_exp(w)
        for i in 1:J
            eval_y_hat_i += get_fit_y_i(i) * exp(w_i[j] - lse)
        end
        eval_y_hat[i] = eval_y_hat_i
        nothing
    end

    ka.@kernel function ka_smoothing_kernel_factory(
        ulw_i_j::Function,
        J::Integer,
        get_fit_y_i::AbstractArray,
        eval_y_hat::AbstractVector, 
        theta::Real
    )::Nothing
        i = @index(Global)
        compute_eval_y_hat_i(ulw_i_j, i, J, get_fit_y_i, eval_y_hat, theta)
        nothing
    end

    ka.@kernel function ka_smoothing_kernel_factory(    
        ulw_i_j::Function,
        fit_y::AbstractArray, 
        eval_y_hat::AbstractVector, 
        theta::Real
    )::Nothing
        compute_eval_y_hat_i(ulw_i_j, i, J, get_fit_y_i, eval_y_hat, theta)
        nothing
    end

    function ka_smooth!(
        ulw_i_j::Function,
        I::Integer,
        get_fit_y_i::Function,
        get_eval_x_i::Function,
        eval_y_hat::AbstractVector, 
        theta::Real,
        n_threads::Int = 1
    )::Nothing
        ka_backend = ka.get_backend(eval_y_hat)
        ka_n_elems_per_thread = length(eval_y_hat) ÷ n_threads
        k! = ka_smoothing_kernel_factory(
            ka_backend,
            ka_n_elems_per_thread
        )
        k!(
            ulw_i_j, 
            I, 
            get_fit_y_i,
            get_eval_x_i,
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
        n_threads::Int = 1
    )::Nothing
        function ulw_i_j(i::Integer, j::Integer)::Real
            return ulw_fit_x_i_eval_x_j(
                su.get_ith_elem_from_first_dim(fit_x, i),
                su.get_ith_elem_from_first_dim(eval_x, j)
            )
        end
        function get_fit_y_i(i::Integer)
            return su.get_ith_elem_from_first_dim(fit_y, i)
        end
        function get_eval_x_i(i::Integer)
            return su.get_ith_elem_from_first_dim(eval_x, i)
        end
        ka_smooth!(
            ulw_i_j,
            length(fit_y),
            get_fit_y_i,
            get_eval_x_i,
            eval_y_hat, 
            theta,
            n_threads
        )
        nothing
    end

end
