#pragma once
#ifndef EG__TENSOR_SIZE__H
#define EG__TENSOR_SIZE__H

#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include "utils.h"


namespace eg {

class SizeInitError : public std::runtime_error
{
  using std::runtime_error::runtime_error;
};

class SizeWrongDimError : public std::runtime_error
{
  using std::runtime_error::runtime_error;
};


template<size_t _Nm>
class Size
{
public:
  using value_type = size_t;
  using reference = value_type &;
  using const_reference = const value_type &;
  using index = size_t;
  using dim_storage = std::array<value_type, _Nm>;

  Size() = default;

  Size(const dim_storage& __dims) {
    _M_dims = __dims;
  }

  Size(dim_storage&& __dims) {
    _M_dims = std::move(__dims);
  }

  template <typename... _Dims>
  explicit Size(_Dims &&... __dims) {
      _M_init_with_packed_args(std::index_sequence_for<_Dims...>{},
                               std::forward<_Dims>(__dims)...);
  }

  Size(std::initializer_list<value_type> __l) {
    _M_init_with_list(std::make_index_sequence<_Nm>{}, __l);
  }

  auto to_tuple() const {
    return utils::to_tuple(_M_dims);
  }

  reference operator[](index __i) { return _M_dims[__i]; }

  const_reference operator[](index __i) const { return _M_dims[__i]; }

  constexpr size_t dims() const noexcept { return _Nm; }

  std::string to_string() const {
    return utils::array_to_string(_M_dims);
  }

  value_type count() const {
    return std::accumulate(_M_dims.begin(), _M_dims.end(), 1,
                           std::multiplies<value_type>());
  }

  template<typename Size_1 = Size<_Nm - 1>>
  Size_1 drop_first() const {
    using dim_storage_1 = typename Size_1::dim_storage;
    dim_storage_1 _slice;
    std::copy(_M_dims.begin() + 1, _M_dims.end(), _slice.begin());

    return _slice;
  }

private:
  dim_storage _M_dims;

  template<std::size_t... _Is, typename... _Dims>
  void _M_init_with_packed_args(std::index_sequence<_Is...>, _Dims&&... __dims)
  {
    static_assert(_Nm == sizeof...(_Dims),
                  "Number of arguments doesn't match `_Nm`");

    static_assert(utils::are_integral_v<_Dims...>,
                  "Size args must be of integral type");

    if (!utils::are_positive_v(std::forward<_Dims>(__dims)...))
      throw SizeInitError("Size args should be positive");

    (static_cast<void>(_M_dims[_Is] = __dims), ...);
  }

  template<std::size_t... _Is>
  void _M_init_with_list(std::index_sequence<_Is...>, std::initializer_list<value_type> __l)
  {
    if (_Nm != __l.size())
      throw SizeWrongDimError("Initializer list size doesn't match _Nm");

    (static_cast<void>(_M_dims[_Is] = *(__l.begin() + _Is)), ...);
  }
};

using Size1D = Size<1>;
using Size2D = Size<2>;
using Size3D = Size<3>;
using Size4D = Size<4>;
using Size5D = Size<5>;

} // namespace eg

#endif // EG__TENSOR_SIZE__H
