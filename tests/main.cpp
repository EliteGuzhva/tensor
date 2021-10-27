#include <iostream>

#include <tensor/tensor.h>

using namespace eg;


int main() {
  // Create an empty tensor
  Tensor5f tensor({15, 128, 64, 7, 7});
  std::cout << "Created a tensor of size: " << tensor.size().to_string()
            << std::endl;
  std::cout << "It has " << tensor.size().count() << " elements\n";

  // Split tensor size into individual dimensions
  auto [seq_length, batch_size, channels, height, width] = tensor.size().to_tuple();

  // Other creation options
  auto zero_tensor = Tensor1f::zeros({ 5 });
  auto one_tensor = Tensor1d::ones_like(zero_tensor);
  auto empty_tensor = Tensor2i::empty({ 2, 4 });

  auto image_size = Size3D(3, 28, 28);
  float fill_value = 3.14;
  auto full_tensor = Tensor3f::full(image_size, fill_value);

  // Custom dim
  constexpr size_t custom_dim = 7;
  auto custom_size = Size<custom_dim>(1, 2, 3, 4, 5, 6, 7);
  auto custom_tensor = Tensor<uint16_t, custom_dim>::ones(custom_size);

  // Matrix
  auto matrix = Tensor2i::empty({4, 3});

  auto [rows, cols] = matrix.size().to_tuple();

  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
      matrix[i][j] = i*cols + j + 1;

  std::cout << "\nMatrix:\n";
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++)
      std::cout << matrix[i][j] << ", ";
    std::cout << std::endl;
  }

  // Get sub-tensor (as a view)
  auto /* TensorView<int, 1> */ row0 = matrix[0];
  std::cout << "0th row size is: " << row0.size().to_string() << std::endl;

  auto& value02 = row0[2];
  value02 = 17;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++)
      std::cout << matrix[i][j] << ", ";
    std::cout << std::endl;
  }

  return 0;
}
