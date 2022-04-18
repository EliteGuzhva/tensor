#pragma once

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace eg::utils {

//! @brief are_same
template <typename Tp, typename... Ts>
inline constexpr bool are_same_v = std::conjunction_v<std::is_same<Tp, Ts>...>;

//! @brief are_integral
template <typename... Ts>
inline constexpr bool are_integral_v = std::conjunction_v<std::is_integral<Ts>...>;

//! @brief are_convertible
template <typename Tp, typename... Ts>
inline constexpr bool are_convertible_v = std::conjunction_v<std::is_convertible<Ts, Tp>...>;

//! @brief are_positive
template <typename Tp>
bool is_positive_v(Tp&& value)
{
    return value >= 0;
}

template <typename... Ts>
bool are_positive_v(Ts&&... values)
{
    return (... && is_positive_v(values));
}

//! @brief to_tuple
template <typename ArrayLike, size_t... Is>
auto _to_tuple_impl(const ArrayLike& array, std::index_sequence<Is...> /*unused*/)
{
    return std::make_tuple(array[Is]...);
}

template <typename Tp, size_t Size,
    typename Ids = std::make_index_sequence<Size>>
auto to_tuple(const std::array<Tp, Size>& array)
{
    return _to_tuple_impl(array, Ids {});
}

//! @brief array_to_string
template <typename Tp, size_t Size>
std::string array_to_string(const std::array<Tp, Size>& array)
{
    std::string array_string = "(";
    for (size_t i = 0; i < Size; i++) {
        array_string += (i == 0 ? "" : ", ") + std::to_string(array[i]);
    }
    array_string += ")";

    return array_string;
}

//! @brief emplace_back_n
template <typename Tp, typename... Ts>
void emplace_back_n(std::vector<Tp>& v, Ts&&... values)
{
    static_assert(are_convertible_v<Ts..., Tp>,
        "emplace_back_n args must be convertible to `Tp`");

    v.reserve(sizeof...(Ts));
    (void)std::initializer_list<int> { (v.emplace_back(values), 0)... };
}

//! @brief costruct_n
template <typename Tp, typename Alloc = std::allocator<Tp>,
    typename... Args>
void construct_n(Alloc& a, Tp* p, size_t n, Args&&... args)
{
    for (size_t _i = 0; _i < n; _i++) {
        std::allocator_traits<Alloc>::construct(a, p + _i, std::forward<Args>(args)...);
    }
}

} // namespace eg::utils
