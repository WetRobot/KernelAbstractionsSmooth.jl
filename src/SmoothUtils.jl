module SmoothUtils        
    function get_ith_slice_from_first_dim(x::AbstractArray, i::Integer)
        n_dims = length(size(x))
        if n_dims == 1
            return(x[i])
        end
        indices = (i, ntuple(_ -> Colon(), n_dims - 1)...)
        return(x[indices...])
    end

    function get_ith_slice_from_first_dim_factory(x::AbstractArray)
        function fun(i::Integer)
            su.get_ith_slice_from_first_dim(x, i)
        end
        return fun
    end
end