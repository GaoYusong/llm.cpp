# llm.cpp

A C++ port of [llm.c](https://github.com/karpathy/llm.c), featuring a tiny torch library while maintaining overall simplicity. tinytorch is a single-header library that provides a simple way to train neural networks in C++. You can train your own neural network by just copying and including the [tinytorch.hpp](tinytorch.hpp) file in your project. There is a simple example in the [example](example) directory that demonstrates how to use the library. The train_gpt2 demonstrates a more complex example of training the GPT-2 model.

The tiny torch library draws inspiration from the code and concepts found in [ggml](https://github.com/ggerganov/ggml).

# quick start

run train_gpt2
```bash
# the starter pack script will download necessary files to train the GPT-2 model
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
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
TensorContext Layout
---------------------
Total memory size: 8589934592
Used  memory size: 497924928
Number of objects: 148
Checkpoint loaded successfully!
train dataset num_batches: 1192
val dataset num_batches: 128
TensorContext Layout
---------------------
Total memory size: 8589934592
Used  memory size: 3150647728
Number of objects: 727
Computation Graph created successfully!
val loss: 5.32553
step 0 train Loss: 4.67778 (took 7666.71 ms)
step 1 train Loss: 5.19158 (took 7368.44 ms)
step 2 train Loss: 4.43868 (took 7329.43 ms)
step 3 train Loss: 4.13846 (took 7341.64 ms)
step 4 train Loss: 4.14422 (took 7326.77 ms)
step 5 train Loss: 3.83472 (took 7280.71 ms)
... (omitted)
step 39 train Loss: 3.90174 (took 7558.28 ms)
val loss: 4.29154
generating:
---
Being barren savour, grant
Everyone, every man and woman,
Is heir to his life in unwritten words:
The Salmon of the family
How would gentlerish words indeed bring them
Were men so much more the meanest of the heart,
That a man feared all, but that
---
step 40 train Loss: 3.95306 (took 7513.37 ms)
```

run the tinytorch example
```bash
cd example
make
./example
```

```
step 0, val_loss: 3.29672
step 1000, val_loss: 1.21564
step 2000, val_loss: 0.916036
step 3000, val_loss: 0.722802
step 4000, val_loss: 0.64182
step 5000, val_loss: 0.571068
... (omitted)
step 29000, val_loss: 0.248639
training loss: 0.0427181
test case 0, x: 0.398134, 0.43387, 0.0558768, y_truth: 0, y_pred: 0 (correct)
test case 1, x: 0.121094, 0.225514, 0.210815, y_truth: 0, y_pred: 0 (correct)
test case 2, x: 0.164785, 0.548775, 0.266744, y_truth: 1, y_pred: 1 (correct)
test case 3, x: 0.165368, 0.333232, 0.62564, y_truth: 1, y_pred: 1 (correct)
test case 4, x: 0.138728, 0.600311, 0.430948, y_truth: 1, y_pred: 1 (correct)
test case 5, x: 0.942843, 0.356101, 0.990194, y_truth: 2, y_pred: 2 (correct)
test case 6, x: 0.195752, 0.0023702, 0.835953, y_truth: 1, y_pred: 1 (correct)
test case 7, x: 0.869364, 0.399392, 0.577386, y_truth: 1, y_pred: 1 (correct)
test case 8, x: 0.124342, 0.813798, 0.497015, y_truth: 1, y_pred: 1 (correct)
test case 9, x: 0.327459, 0.609631, 0.0607804, y_truth: 1, y_pred: 1 (correct)
test case 10, x: 0.535353, 0.673669, 0.35889, y_truth: 1, y_pred: 1 (correct)
test case 11, x: 0.867133, 0.904681, 0.974269, y_truth: 2, y_pred: 2 (correct)
test case 12, x: 0.545874, 0.506076, 0.62621, y_truth: 1, y_pred: 1 (correct)
test case 13, x: 0.716333, 0.415467, 0.755643, y_truth: 1, y_pred: 1 (correct)
test case 14, x: 0.0869354, 0.123324, 0.703014, y_truth: 0, y_pred: 1 (wrong)
test case 15, x: 0.562335, 0.16715, 0.284856, y_truth: 1, y_pred: 1 (correct)
test case 16, x: 0.582393, 0.285243, 0.0762841, y_truth: 0, y_pred: 0 (correct)
test case 17, x: 0.107266, 0.814349, 0.760979, y_truth: 1, y_pred: 1 (correct)
test case 18, x: 0.765755, 0.0510383, 0.800836, y_truth: 1, y_pred: 1 (correct)
test case 19, x: 0.655536, 0.597097, 0.412479, y_truth: 1, y_pred: 1 (correct)
test case 20, x: 0.528384, 0.55012, 0.869494, y_truth: 2, y_pred: 2 (correct)
test case 21, x: 0.58597, 0.393864, 0.672889, y_truth: 1, y_pred: 1 (correct)
test case 22, x: 0.239555, 0.204406, 0.458691, y_truth: 0, y_pred: 0 (correct)
test case 23, x: 0.216776, 0.358567, 0.440994, y_truth: 1, y_pred: 1 (correct)
test case 24, x: 0.783095, 0.470312, 0.52966, y_truth: 1, y_pred: 1 (correct)
test case 25, x: 0.98887, 0.937941, 0.978243, y_truth: 2, y_pred: 2 (correct)
test case 26, x: 0.324292, 0.36803, 0.481786, y_truth: 1, y_pred: 1 (correct)
test case 27, x: 0.380469, 0.541014, 0.823395, y_truth: 1, y_pred: 1 (correct)
test case 28, x: 0.804339, 0.531356, 0.492851, y_truth: 1, y_pred: 1 (correct)
test case 29, x: 0.350021, 0.808311, 0.285363, y_truth: 1, y_pred: 1 (correct)
```

# test

test gpt2
```bash
make test_gpt2
./test_gpt2
```

test tinytorch
```bash
make test_tensor
./test_tensor
```

# tinytorch example

a simple example of using the tinytorch library to train a neural network to classify fruits based on their color, size, and weight.

to run the tinytorch example, do the following:
```bash
cd example
make
./example
```

the runnable example code is in [example/tinytorch_example.cpp](example/tinytorch_example.cpp), here is a core part of the code:
```cpp
#include "../tinytorch.hpp"

namespace tt = tinytorch;

// let's create a neural network to approximate the following function
int classify_fruit(float color, float size, float weight) {
    // color in [0, 1] where 0 represents green and 1 represents red
    // size in [0, 1] where 0 represents small and 1 represents large
    // weight in [0, 1] where 0 represents light and 1 represents heavy
    // y in [0, 3] representing three types of fruits: apple, orange, and banana

    double y = 0.8 * color + 1.0 * size + 1.0 * weight;

    if (y < 0.9) {
        return 0;  // Apple
    } else if (y < 1.8) {
        return 1;  // Orange
    } else {
        return 2;  // Banana
    }
}

int main() {
    // create a context to hold all the tensors
    tt::TensorContext ctx((size_t)1024 * 1024);

    // generate some random data
    int train_size = 10000, val_size = 100;
    auto [train_inputs, train_targets, val_inputs, val_targets] =
        generate_dataset(train_size, val_size);

    // create a simple one hidden layer neural network
    int variables = 3;  // number of input variables
    int neurons = 32;   // number of neurons in the hidden layer
    int classes = 3;    // number of classes

    // input and target tensors
    auto &x = *ctx.NewTensor({1, variables});
    auto &y = *ctx.NewTensor({1}, tt::kI32);
    // parameters
    auto &W1 = *ctx.NewTensor({neurons, variables})->RandomNorm();
    auto &b1 = *ctx.NewTensor({neurons})->RandomNorm();
    auto &W2 = *ctx.NewTensor({classes, neurons})->RandomNorm();
    auto &b2 = *ctx.NewTensor({classes})->RandomNorm();
    vector<tt::Tensor *> params = {&W1, &b1, &W2, &b2};
    // create the computation graph
    auto &hidden = (x.MatMul(W1) + b1).Gelu();
    auto &logits = hidden.MatMul(W2) + b2;
    auto &probs = logits.Softmax();
    auto &loss = probs.CrossEntropy(y);

    float learning_rate = 0.001;
    float training_loss = 0;

    // train the model
    for (int step = 0; step < 30000; step++) {
        if (step % 1000 == 0) {
            // evaluate the model on the validation set
            float val_loss = 0;
            for (int i = 0; i < val_size; i++) {
                x.Fill(val_inputs[i]);
                y.Fill(val_targets[i]);
                loss.Forward();
                val_loss += loss.Flatten()[0];
            }
            std::cout << "step " << step << ", val_loss: " << val_loss / val_size << std::endl;
        }

        // pick a random training case
        int casei = rand() % train_size;
        x.Fill(train_inputs[casei]);
        y.Fill(train_targets[casei]);

        // forward and backward
        loss.Forward();
        loss.ZeroGrad();
        loss.Backward();

        // update the parameters
        for (auto param : params) {
            auto weights = (float *)param->data();
            auto grads = (float *)param->grad()->data();
            for (int i = 0; i < param->NumElements(); i++) {
                weights[i] -= learning_rate * grads[i];
            }
        }

        training_loss = loss.Flatten()[0];
    }

    std::cout << "training loss: " << training_loss << std::endl;

    // test the model
    for (int i = 0; i < 30; i++) {
        auto [testx, y_truth] = generate_case();
        x.Fill(testx);
        probs.Forward();

        // find the class with the highest probability
        auto probs_data = probs.Flatten();
        int y_pred = 0;
        for (int j = 0; j < classes; j++) {
            if (probs_data[j] > probs_data[y_pred]) {
                y_pred = j;
            }
        }

        std::cout << "test case " << i << ", x: " << testx[0] << ", " << testx[1] << ", "
                  << testx[2] << ", y_truth: " << y_truth << ", y_pred: " << y_pred;
        if (y_truth == y_pred) {
            std::cout << " (correct)";
        } else {
            std::cout << " (wrong)";
        }
        std::cout << endl;
    }

    return 0;
}
```

# license
MIT
