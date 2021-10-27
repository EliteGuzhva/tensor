#pragma once
#ifndef EG__TENSOR__H
#define EG__TENSOR__H

#include <cstdint>
#include <type_traits>

#include "size.h"


namespace eg {

//! @brief _Tensor_base
template<typename _Tp, size_t _Dim, typename _Alloc = std::allocator<_Tp>>
class _Tensor_base
{
protected:
  using _Alloc_value_type = typename _Alloc::value_type;

  static_assert(std::is_same_v<typename std::remove_cv_t<_Tp>, _Tp>,
      "Tensor must have a non-const, non-volatile value_type");

  static_assert(std::is_arithmetic_v<_Tp>,
      "Tensor type should be arithmetic");

  using _Alloc_traits = std::allocator_traits<_Alloc>;

  using value_type = _Tp;
  using pointer = typename _Alloc_traits::pointer;
  using const_pointer = typename _Alloc_traits::const_pointer;
  using reference = value_type &;
  using const_reference = const value_type &;
  // TODO: iterator
  using index = size_t;
  using size_type = Size<_Dim>;
  using allocator_type = _Alloc;

  _Tensor_base() = default;

  _Tensor_base(const size_type& __size) {
    _M_set_size(__size);
  }

  _Tensor_base(size_type&& __size) {
    _M_set_size(std::move(__size));
  }

  // TODO: better pointer manipulations. maybe use iterator?
  _Tensor_base(pointer __p, const size_type& __size) {
    _M_set_size(__size);
    _M_set_data(__p);
  }

public:
  const size_type& size() const { return _M_size; }

  constexpr size_t dim() const { return _Dim; }

  constexpr size_t element_size() const { return sizeof(value_type); };

  pointer data_ptr() { return _M_data; }
  const_pointer data_ptr() const { return _M_data; }

  /* std::string to_string() const; */

protected:
  _Alloc _M_allocator;
  size_type _M_size;
  size_t _M_count = 0;
  std::array<size_t, _Dim> _M_strides;
  pointer _M_data = nullptr;

  void _M_set_size(const size_type& __size) {
    _M_size = __size;
    _M_set_stride();
  }

  void _M_set_size(size_type&& __size) {
    _M_size = std::move(__size);
    _M_set_stride();
  }

  void _M_set_stride() {
    _M_count = _M_size.count();

    // TODO: is there a better way?
    size_t _stride = _M_count;
    for (index i = 0; i < _Dim; i++) {
      _stride /= _M_size[i];
      _M_strides[i] = _stride;
    }
  }

  void _M_allocate_space() {
    _M_data = _Alloc_traits::allocate(_M_allocator, _M_count);
  }

  void _M_deallocate_space() {
    _Alloc_traits::deallocate(_M_allocator, _M_data, _M_count);
  }

  void _M_set_data(pointer __p) {
    _M_data = __p;
  }

  void _M_construct_with(const_reference __value) {
    utils::construct_n(_M_allocator, _M_data, _M_count, __value);
  }

  pointer _M_get_data_slice(index __index) {
    return _M_data + __index * _M_strides[0];
  }

  reference _M_get_value_at(index __index) {
    return *_M_get_data_slice(__index);
  }

  const_reference _M_get_value_at(index __index) const {
    return *_M_get_data_slice(__index);
  }
};


//! @brief TensorView
template<typename _Tp, size_t _Dim, typename _Alloc = std::allocator<_Tp>>
class TensorView : public _Tensor_base<_Tp, _Dim, _Alloc>
{
  using _Base = _Tensor_base<_Tp, _Dim, _Alloc>;
  using _Alloc_traits = typename _Base::_Alloc_traits;

public:
  using value_type = typename _Base::value_type;
  using pointer = typename _Base::pointer;
  using const_pointer = typename _Base::const_pointer;
  using reference = typename _Base::reference;
  using const_reference = typename _Base::const_reference;
  // TODO: iterator
  using index = typename _Base::index;
  using size_type = typename _Base::size_type;

public:
  TensorView() : _Base() {};

  explicit TensorView(pointer __p, const size_type &__size)
      : _Base(__p, __size) {}

  template <size_t _Nm = _Dim - 1, 
            typename _View = TensorView<value_type, _Nm>,
            typename = std::enable_if_t<std::greater{}(_Nm, 0), bool>>
  _View operator[](index __index) {
    pointer _slice = this->_M_get_data_slice(__index);

    return _View(_slice, this->size().drop_first());
  }

  template <size_t _Nm = _Dim - 1, 
            typename _View = TensorView<value_type, _Nm>,
            typename = std::enable_if_t<std::greater{}(_Nm, 0), bool>>
  _View operator[](index __index) const {
    pointer _slice = this->_M_get_data_slice(__index);

    return _View(_slice, this->size().drop_first());
  }

  template <size_t _Nm = _Dim - 1, 
            typename = std::enable_if_t<std::equal_to{}(_Nm, 0), bool>>
  reference operator[](index __index) {
    return this->_M_get_value_at(__index);
  }

  template <size_t _Nm = _Dim - 1, 
            typename = std::enable_if_t<std::equal_to{}(_Nm, 0), bool>>
  const_reference operator[](index __index) const {
    return this->_M_get_value_at(__index);
  }
};


//! @brief Tensor
template<typename _Tp, size_t _Dim, typename _Alloc = std::allocator<_Tp>>
class Tensor : public _Tensor_base<_Tp, _Dim, _Alloc>
{
  using _Base = _Tensor_base<_Tp, _Dim, _Alloc>;
  using _Alloc_traits = typename _Base::_Alloc_traits;

public:
  using value_type = typename _Base::value_type;
  using pointer = typename _Base::pointer;
  using const_pointer = typename _Base::const_pointer;
  using reference = typename _Base::reference;
  using const_reference = typename _Base::const_reference;
  // TODO: iterator
  using index = typename _Base::index;
  using size_type = typename _Base::size_type;

public:
  Tensor() : _Base() {};

  explicit Tensor(const size_type &__size) : _Base(__size) {
    this->_M_allocate_space();
  }

  explicit Tensor(size_type &&__size) : _Base(__size) {
    this->_M_allocate_space();
  }

  explicit Tensor(const size_type &__size, const_reference __value)
      : _Base(__size) {
    this->_M_allocate_space();
    this->_M_construct_with(__value);
  }

  explicit Tensor(size_type &&__size, const_reference __value)
      : _Base(__size) {
    this->_M_allocate_space();
    this->_M_construct_with(__value);
  }

  explicit Tensor(pointer __p, const size_type &__size)
      : _Base(__p, __size) {}

  //! @brief Returns a tensor filled with the scalar value 0, with the shape
  //!        defined by the variable argument __size.
  static Tensor zeros(const size_type &__size) { return Tensor(__size, 0); }

  //! @brief Returns a tensor filled with the scalar value 0, with the shape
  //!        defined by the variable argument __size.
  static Tensor zeros(size_type &&__size) { return Tensor(__size, 0); }

  //! @brief Returns a tensor filled with the scalar value 0, with the same size
  //!        as __other.
  template<typename _Up>
  static Tensor zeros_like(const Tensor<_Up, _Dim> &__other) {
    return Tensor(__other.size(), 0);
  }

  //! @brief ones
  static Tensor ones(const size_type &__size) { return Tensor(__size, 1); }
  static Tensor ones(size_type &&__size) { return Tensor(__size, 1); }

  //! @brief ones_like
  template<typename _Up>
  static Tensor ones_like(const Tensor<_Up, _Dim> &__other) {
    return Tensor(__other.size(), 1);
  }

  //! @brief empty
  static Tensor empty(const size_type &__size) { return Tensor(__size); }
  static Tensor empty(size_type &&__size) { return Tensor(__size); }

  //! @brief empty_like
  template<typename _Up>
  static Tensor empty_like(const Tensor<_Up, _Dim> &__other) {
    return Tensor(__other.size());
  }

  //! @brief full
  static Tensor full(const size_type &__size, const_reference __value) { return Tensor(__size, __value); }
  static Tensor full(size_type &&__size, const_reference __value) { return Tensor(__size, __value); }

  //! @brief full_like
  template<typename _Up>
  static Tensor full_like(const Tensor<_Up, _Dim> &__other, const_reference __value) {
    return Tensor(__other.size(), __value);
  }

  ~Tensor() {
    this->_M_deallocate_space();
  }

  // FIXME: maybe avoid copying this method from TensorView?
  template <size_t _Nm = _Dim - 1, 
            typename _View = TensorView<value_type, _Nm>,
            typename = std::enable_if_t<std::greater{}(_Nm, 0), bool>>
  _View operator[](index __index) {
    pointer _slice = this->_M_get_data_slice(__index);

    return _View(_slice, this->size().drop_first());
  }

  template <size_t _Nm = _Dim - 1, 
            typename _View = TensorView<value_type, _Nm>,
            typename = std::enable_if_t<std::greater{}(_Nm, 0), bool>>
  _View operator[](index __index) const {
    pointer _slice = this->_M_get_data_slice(__index);

    return _View(_slice, this->size().drop_first());
  }

  template <size_t _Nm = _Dim - 1, 
            typename = std::enable_if_t<std::equal_to{}(_Nm, 0), bool>>
  reference operator[](index __index) {
    return this->_M_get_value_at(__index);
  }

  template <size_t _Nm = _Dim - 1, 
            typename = std::enable_if_t<std::equal_to{}(_Nm, 0), bool>>
  const_reference operator[](index __index) const {
    return this->_M_get_value_at(__index);
  }

};

using Tensor1d = Tensor<double, 1>;
using Tensor2d = Tensor<double, 2>;
using Tensor3d = Tensor<double, 3>;
using Tensor4d = Tensor<double, 4>;
using Tensor5d = Tensor<double, 5>;

using Tensor1f = Tensor<float, 1>;
using Tensor2f = Tensor<float, 2>;
using Tensor3f = Tensor<float, 3>;
using Tensor4f = Tensor<float, 4>;
using Tensor5f = Tensor<float, 5>;

using Tensor1i = Tensor<int, 1>;
using Tensor2i = Tensor<int, 2>;
using Tensor3i = Tensor<int, 3>;
using Tensor4i = Tensor<int, 4>;
using Tensor5i = Tensor<int, 5>;

} // namespace eg

#endif // EG__TENSOR__H
