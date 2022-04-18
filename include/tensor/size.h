#pragma once

#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "utils.h"

namespace eg {

class SizeInitError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

class SizeWrongDimError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <size_t N>
class Size {
public:
    using value_type = size_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using index = size_t;
    using dim_storage = std::array<value_type, N>;

    Size() = default;

    Size(const dim_storage& dims) // NOLINT
        : dims_(dims)
    {
    }

    Size(dim_storage&& dims) // NOLINT
        : dims_(std::move(dims))
    {
    }

    template <typename... Dims>
    explicit Size(Dims&&... dims)
    {
        init_with_packed_args(std::index_sequence_for<Dims...> {},
            std::forward<Dims>(dims)...);
    }

    Size(std::initializer_list<value_type> l)
    {
        init_with_list(std::make_index_sequence<N> {}, l);
    }

    auto to_tuple() const
    {
        return utils::to_tuple(dims_);
    }

    reference operator[](index i) { return dims_[i]; }

    const_reference operator[](index i) const { return dims_[i]; }

    [[nodiscard]] constexpr size_t dims() const noexcept { return N; }

    [[nodiscard]] std::string to_string() const
    {
        return utils::array_to_string(dims_);
    }

    [[nodiscard]] value_type count() const
    {
        return std::accumulate(dims_.begin(), dims_.end(), 1,
            std::multiplies<>());
    }

    template <typename Size_1 = Size<N - 1>>
    Size_1 drop_first() const
    {
        using dim_storage_1 = typename Size_1::dim_storage;
        dim_storage_1 slice;
        std::copy(dims_.begin() + 1, dims_.end(), slice.begin());

        return slice;
    }

private:
    dim_storage dims_;

    template <std::size_t... Is, typename... Dims>
    void init_with_packed_args(std::index_sequence<Is...> /*unused*/, Dims&&... dims)
    {
        static_assert(N == sizeof...(Dims),
            "Number of arguments doesn't match `N`");

        static_assert(utils::are_integral_v<Dims...>,
            "Size args must be of integral type");

        if (!utils::are_positive_v(std::forward<Dims>(dims)...)) {
            throw SizeInitError("Size args should be positive");
        }

        (static_cast<void>(dims_[Is] = dims), ...);
    }

    template <std::size_t... Is>
    void init_with_list(std::index_sequence<Is...> /*unused*/, std::initializer_list<value_type> l)
    {
        if (N != l.size()) {
            throw SizeWrongDimError("Initializer list size doesn't match N");
        }

        (static_cast<void>(dims_[Is] = *(l.begin() + Is)), ...); // NOLINT
    }
};

using Size1D = Size<1>;
using Size2D = Size<2>;
using Size3D = Size<3>;
using Size4D = Size<4>;
using Size5D = Size<5>; // NOLINT

} // namespace eg
