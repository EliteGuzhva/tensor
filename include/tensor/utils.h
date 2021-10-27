#pragma once
#ifndef EG__TENSOR_UTIL__H
#define EG__TENSOR_UTIL__H

#include <array>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>


namespace eg {

namespace utils {

//! @brief are_same
template <typename _Tp, typename... _Ts>
inline constexpr bool are_same_v = std::conjunction_v<std::is_same<_Tp, _Ts>...>;

//! @brief are_integral
template<typename... _Ts>
inline constexpr bool are_integral_v = std::conjunction_v<std::is_integral<_Ts>...>;

//! @brief are_convertible
template<typename _Tp, typename... _Ts>
inline constexpr bool are_convertible_v = std::conjunction_v<std::is_convertible<_Ts, _Tp>...>;

//! @brief are_positive
template<typename _Tp>
bool is_positive_v(_Tp&& __value)
{
  return __value >= 0;
}

template<typename... _Ts>
bool are_positive_v(_Ts&&... __values)
{
  return (... && is_positive_v(__values));
}

//! @brief to_tuple
template<typename _Array, size_t... _Is>
auto _to_tuple_impl(const _Array& __array, std::index_sequence<_Is...>)
{
  return std::make_tuple(__array[_Is]...);
}

template <typename _Tp, size_t _Nm,
          typename _Ids = std::make_index_sequence<_Nm>>
auto to_tuple(const std::array<_Tp, _Nm>& __array)
{
  return _to_tuple_impl(__array, _Ids{});
}

//! @brief array_to_string
template<typename _Tp, size_t _Nm>
std::string array_to_string(const std::array<_Tp, _Nm>& __array)
{
  std::string _string = "(";
  for (size_t i = 0; i < _Nm; i++)
    _string += (i == 0 ? "" : ", ") + std::to_string(__array[i]);
  _string += ")";

  return _string;
}

//! @brief emplace_back_n
template<typename _Tp, typename... _Ts>
void emplace_back_n(std::vector<_Tp>& __v, _Ts&&... __values)
{
  static_assert(are_convertible_v<_Ts..., _Tp>,
      "emplace_back_n args must be convertible to `_Tp`");

  __v.reserve(sizeof...(_Ts));
  (void)std::initializer_list<int>{ (__v.emplace_back(__values), 0)... };
}

//! @brief costruct_n
template <typename _Tp, typename _Alloc = std::allocator<_Tp>,
          typename _Alloc_traits = std::allocator_traits<_Alloc>,
          typename... _Args>
void construct_n(_Alloc &__a, _Tp *__p, size_t __n, _Args &&... __args) {
  for (size_t _i = 0; _i < __n; _i++)
    _Alloc_traits::construct(__a, __p + _i, std::forward<_Args>(__args)...);
}

}

} // namespace eg

#endif // EG__TENSOR_UTIL__H
