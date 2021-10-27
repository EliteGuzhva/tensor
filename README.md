# Tensor
Simple generic tensor implementation

## Build
```bash
cmake -B build .
```

Library is header-only, so just install it:
```bash
cmake --install build
```

Optional flags:
- ```BUILD_TESTS=ON```

## Usage
This is just a basic example:
```cpp
auto t = eg::Tensor4f::zeros({128, 3, 1080, 1920});
auto [n, c, h, w] = t.size().to_tuple();
auto /* TensorView<float, 3> */ firstImg = t[0];
```

For more examples see tests folder.
