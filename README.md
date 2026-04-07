# llm.cpp

A C++ port of [llm.c](https://github.com/karpathy/llm.c) that trains GPT-2 (124M) from scratch, featuring **tinytorch** -- a single-header tensor library with automatic differentiation.

tinytorch provides a simple computation graph API inspired by [ggml](https://github.com/ggerganov/ggml). Just copy [tinytorch.hpp](tinytorch.hpp) into your project and start training neural networks in C++.

## Prerequisites

- **Compiler**: `clang` (recommended) or `g++` with C++20 support
- **OpenMP** (optional, for ~3x speedup):
  - macOS: `brew install libomp`
  - Ubuntu/Debian: `sudo apt-get install libomp-dev`

## Quick Start

### 1. Run the tinytorch example (no downloads needed)

A simple neural network that classifies fruits by color, size, and weight:

```bash
cd example
make
./example
```

```
step 0, val_loss: 2.97879
step 1000, val_loss: 0.934616
step 2000, val_loss: 0.647546
...
step 29000, val_loss: 0.199498
training loss: 0.00478648
test case 0, x: 0.398134, 0.43387, 0.0558768, y_truth: 0, y_pred: 0 (correct)
test case 1, x: 0.121094, 0.225514, 0.210815, y_truth: 0, y_pred: 0 (correct)
...
```

### 2. Train GPT-2

```bash
# Download GPT-2 weights and training data (~500MB)
./dev/download_starter_pack.sh

# Build and run (adjust thread count to your CPU)
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2
```

```
[GPT-2]:
max_seq_len: 1024
vocab_size: 50257
padded_vocab_size: 50304
num_layers: 12
num_heads: 12
channels: 768
Number of Parameters: 124475904
...
val loss: 5.32553
step 0 train Loss: 4.67778 (took 7666.71 ms)
step 1 train Loss: 5.19158 (took 7368.44 ms)
...
step 39 train Loss: 3.90174 (took 7558.28 ms)
val loss: 4.29154
generating:
---
Being barren savour, grant
Everyone, every man and woman,
Is heir to his life in unwritten words...
---
```

## Tests

```bash
# Tensor operation tests (no data files needed)
make test_tensor && ./test_tensor

# GPT-2 forward/backward tests (requires download_starter_pack.sh)
make test_gpt2 && ./test_gpt2
```

## Using tinytorch in your own project

Just copy `tinytorch.hpp` into your project. Here is a minimal example that trains a one-hidden-layer network:

```cpp
#include "tinytorch.hpp"

namespace tt = tinytorch;

int main() {
    // create a context to hold all tensors (1MB arena)
    tt::TensorContext ctx((size_t)1024 * 1024);

    int variables = 3, neurons = 32, classes = 3;

    // input and target
    auto &x = *ctx.NewTensor({1, variables});
    auto &y = *ctx.NewTensor({1}, tt::kI32);

    // parameters
    auto &W1 = *ctx.NewTensor({neurons, variables})->RandomNorm();
    auto &b1 = *ctx.NewTensor({neurons})->RandomNorm();
    auto &W2 = *ctx.NewTensor({classes, neurons})->RandomNorm();
    auto &b2 = *ctx.NewTensor({classes})->RandomNorm();
    std::vector<tt::Tensor *> params = {&W1, &b1, &W2, &b2};

    // build computation graph
    auto &hidden = (x.MatMul(W1) + b1).Gelu();
    auto &logits = hidden.MatMul(W2) + b2;
    auto &probs = logits.Softmax();
    auto &loss = probs.CrossEntropy(y);

    // training loop
    float learning_rate = 0.001;
    for (int step = 0; step < 30000; step++) {
        // fill x and y with your training data here
        // x.Fill({color, size, weight});
        // y.Fill({label});

        loss.Forward();
        loss.ZeroGrad();
        loss.Backward();

        // SGD update
        for (auto param : params) {
            auto weights = (float *)param->data();
            auto grads = (float *)param->grad()->data();
            for (int i = 0; i < param->NumElements(); i++) {
                weights[i] -= learning_rate * grads[i];
            }
        }
    }

    // inference
    // x.Fill({0.5, 0.3, 0.8});
    // probs.Forward();
    // auto result = probs.Flatten();  // probabilities for each class

    return 0;
}
```

See [example/tinytorch_example.cpp](example/tinytorch_example.cpp) for the complete runnable example.

## Project Structure

```
tinytorch.hpp           # Single-header tensor library (~1400 lines)
train_gpt2.cpp          # GPT-2 training pipeline
test_gpt2.cpp           # GPT-2 forward/backward tests
test_tensor.cpp         # Tensor operation tests
example/                # Standalone tinytorch usage example
llmc/                   # Utility headers (dataloader, tokenizer, RNG)
dev/                    # Development scripts
```

## License

MIT
