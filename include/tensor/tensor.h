#pragma once

#include <cstdint>
#include <type_traits>

#include "size.h"

namespace eg {

//! @brief _Tensor_base
template <typename Tp, size_t Dim, typename Alloc = std::allocator<Tp>>
class TensorBase {
protected:
    using alloc_value_type = typename Alloc::value_type;

    static_assert(std::is_same_v<typename std::remove_cv_t<Tp>, Tp>,
        "Tensor must have a non-const, non-volatile value_type");

    static_assert(std::is_arithmetic_v<Tp>,
        "Tensor type should be arithmetic");

    using allocator_type = Alloc;
    using alloc_traits = std::allocator_traits<allocator_type>;

    using value_type = Tp;
    using pointer = typename alloc_traits::pointer;
    using const_pointer = typename alloc_traits::const_pointer;
    using reference = value_type&;
    using const_reference = const value_type&;
    // TODO(elit3guzhva): iterator
    using index = size_t;
    using size_type = Size<Dim>;

    TensorBase() = default;

    explicit TensorBase(const size_type& size)
    {
        set_size(size);
    }

    explicit TensorBase(size_type&& size)
    {
        set_size(std::move(size));
    }

    // TODO(elit3guzhva): better pointer manipulations. maybe use iterator?
    TensorBase(pointer p, const size_type& size)
    {
        set_size(size);
        set_data(p);
    }

    void set_size(const size_type& size)
    {
        size_ = size;
        set_stride();
    }

    void set_size(size_type&& size)
    {
        size_ = std::move(size);
        set_stride();
    }

    void set_stride()
    {
        count_ = size_.count();

        // TODO(elit3guzhva): is there a better way?
        size_t _stride = count_;
        for (index i = 0; i < Dim; i++) {
            _stride /= size_[i];
            strides_[i] = _stride;
        }
    }

    void allocate_space()
    {
        data_ = alloc_traits::allocate(allocator_, count_);
    }

    void deallocate_space()
    {
        alloc_traits::deallocate(allocator_, data_, count_);
    }

    void set_data(pointer p)
    {
        data_ = p;
    }

    void construct_with(const_reference value)
    {
        utils::construct_n(allocator_, data_, count_, value);
    }

    pointer get_data_slice(index i)
    {
        return data_ + i * strides_[0];
    }

    reference get_value_at(index i)
    {
        return *get_data_slice(i);
    }

    const_reference get_value_at(index i) const
    {
        return *get_data_slice(i);
    }

public:
    const size_type& size() const { return size_; }

    [[nodiscard]] constexpr size_t dim() const { return Dim; }

    [[nodiscard]] constexpr size_t element_size() const { return sizeof(value_type); };

    pointer data_ptr() { return data_; }
    const_pointer data_ptr() const { return data_; }

    /* std::string to_string() const; */

private:
    allocator_type allocator_;
    size_type size_;
    size_t count_ = 0;
    std::array<size_t, Dim> strides_;
    pointer data_ = nullptr;
};

//! @brief TensorView
template <typename Tp, size_t Dim, typename Alloc = std::allocator<Tp>>
class TensorView : public TensorBase<Tp, Dim, Alloc> {
    using Base = TensorBase<Tp, Dim, Alloc>;
    using alloc_traits = typename Base::alloc_traits;

public:
    using value_type = typename Base::value_type;
    using pointer = typename Base::pointer;
    using const_pointer = typename Base::const_pointer;
    using reference = typename Base::reference;
    using const_reference = typename Base::const_reference;
    // TODO(elit3guzhva): iterator
    using index = typename Base::index;
    using size_type = typename Base::size_type;

    TensorView()
        : Base() {};

    explicit TensorView(pointer p, const size_type& size)
        : Base(p, size)
    {
    }

    template <size_t Dim_1 = Dim - 1,
        typename View = TensorView<value_type, Dim_1>,
        typename = std::enable_if_t<std::greater {}(Dim_1, 0), bool>>
    View operator[](index i)
    {
        pointer slice = this->get_data_slice(i);

        return View(slice, this->size().drop_first());
    }

    template <size_t Dim_1 = Dim - 1,
        typename View = TensorView<value_type, Dim_1>,
        typename = std::enable_if_t<std::greater {}(Dim_1, 0), bool>>
    View operator[](index i) const
    {
        pointer slice = this->get_data_slice(i);

        return View(slice, this->size().drop_first());
    }

    template <size_t Dim_1 = Dim - 1,
        typename = std::enable_if_t<std::equal_to {}(Dim_1, 0), bool>>
    reference operator[](index i)
    {
        return this->get_value_at(i);
    }

    template <size_t Dim_1 = Dim - 1,
        typename = std::enable_if_t<std::equal_to {}(Dim_1, 0), bool>>
    const_reference operator[](index i) const
    {
        return this->get_value_at(i);
    }
};

//! @brief Tensor
//! TODO: implement special functions
template <typename Tp, size_t Dim, typename Alloc = std::allocator<Tp>>
class Tensor : public TensorBase<Tp, Dim, Alloc> {
    using Base = TensorBase<Tp, Dim, Alloc>;
    using alloc_traits = typename Base::alloc_traits;

public:
    using value_type = typename Base::value_type;
    using pointer = typename Base::pointer;
    using const_pointer = typename Base::const_pointer;
    using reference = typename Base::reference;
    using const_reference = typename Base::const_reference;
    // TODO(elit3guzhva): iterator
    using index = typename Base::index;
    using size_type = typename Base::size_type;

    Tensor()
        : Base() {};

    explicit Tensor(const size_type& size)
        : Base(size)
    {
        this->allocate_space();
    }

    explicit Tensor(size_type&& size)
        : Base(size)
    {
        this->allocate_space();
    }

    Tensor(const size_type& size, const_reference value)
        : Base(size)
    {
        this->allocate_space();
        this->construct_with(value);
    }

    Tensor(size_type&& size, const_reference value)
        : Base(size)
    {
        this->allocate_space();
        this->construct_with(value);
    }

    Tensor(pointer p, const size_type& size)
        : Base(p, size)
    {
    }

    ~Tensor()
    {
        this->deallocate_space();
    }

    //! @brief Returns a tensor filled with the scalar value 0, with the shape
    //!        defined by the variable argument size.
    static Tensor zeros(const size_type& size) { return Tensor(size, 0); }

    //! @brief Returns a tensor filled with the scalar value 0, with the shape
    //!        defined by the variable argument size.
    static Tensor zeros(size_type&& size) { return Tensor(size, 0); }

    //! @brief Returns a tensor filled with the scalar value 0, with the same size
    //!        as other.
    template <typename Up>
    static Tensor zeros_like(const Tensor<Up, Dim>& other)
    {
        return Tensor(other.size(), 0);
    }

    //! @brief ones
    static Tensor ones(const size_type& size) { return Tensor(size, 1); }
    static Tensor ones(size_type&& size) { return Tensor(size, 1); }

    //! @brief ones_like
    template <typename Up>
    static Tensor ones_like(const Tensor<Up, Dim>& other)
    {
        return Tensor(other.size(), 1);
    }

    //! @brief empty
    static Tensor empty(const size_type& size) { return Tensor(size); }
    static Tensor empty(size_type&& size) { return Tensor(size); }

    //! @brief empty_like
    template <typename Up>
    static Tensor empty_like(const Tensor<Up, Dim>& other)
    {
        return Tensor(other.size());
    }

    //! @brief full
    static Tensor full(const size_type& size, const_reference value) { return Tensor(size, value); }
    static Tensor full(size_type&& size, const_reference value) { return Tensor(size, value); }

    //! @brief full_like
    template <typename Up>
    static Tensor full_like(const Tensor<Up, Dim>& other, const_reference value)
    {
        return Tensor(other.size(), value);
    }

    // FIXME: maybe avoid copying this method from TensorView?
    template <size_t Dim_1 = Dim - 1,
        typename View = TensorView<value_type, Dim_1>,
        typename = std::enable_if_t<std::greater {}(Dim_1, 0), bool>>
    View operator[](index i)
    {
        pointer slice = this->get_data_slice(i);

        return View(slice, this->size().drop_first());
    }

    template <size_t Dim_1 = Dim - 1,
        typename View = TensorView<value_type, Dim_1>,
        typename = std::enable_if_t<std::greater {}(Dim_1, 0), bool>>
    View operator[](index i) const
    {
        pointer slice = this->get_data_slice(i);

        return View(slice, this->size().drop_first());
    }

    template <size_t Dim_1 = Dim - 1,
        typename = std::enable_if_t<std::equal_to {}(Dim_1, 0), bool>>
    reference operator[](index i)
    {
        return this->get_value_at(i);
    }

    template <size_t Dim_1 = Dim - 1,
        typename = std::enable_if_t<std::equal_to {}(Dim_1, 0), bool>>
    const_reference operator[](index i) const
    {
        return this->get_value_at(i);
    }
};

using Tensor1d = Tensor<double, 1>;
using Tensor2d = Tensor<double, 2>;
using Tensor3d = Tensor<double, 3>;
using Tensor4d = Tensor<double, 4>;
using Tensor5d = Tensor<double, 5>; // NOLINT

using Tensor1f = Tensor<float, 1>;
using Tensor2f = Tensor<float, 2>;
using Tensor3f = Tensor<float, 3>;
using Tensor4f = Tensor<float, 4>;
using Tensor5f = Tensor<float, 5>; // NOLINT

using Tensor1i = Tensor<int, 1>;
using Tensor2i = Tensor<int, 2>;
using Tensor3i = Tensor<int, 3>;
using Tensor4i = Tensor<int, 4>;
using Tensor5i = Tensor<int, 5>; // NOLINT

} // namespace eg
