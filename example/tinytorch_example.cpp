#include <cstddef>
#include <iostream>
#include <vector>

#include "../tinytorch.hpp"

namespace tt = tinytorch;
using namespace std;  // NOLINT

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

tuple<vector<float>, int> generate_case() {
    float x0 = (float)rand() / RAND_MAX;
    float x1 = (float)rand() / RAND_MAX;
    float x2 = (float)rand() / RAND_MAX;
    int y = classify_fruit(x0, x1, x2);
    return {{x0, x1, x2}, y};
}

tuple<vector<vector<float>>, vector<int>, vector<vector<float>>, vector<int>> generate_dataset(
    int train_size, int val_size) {
    vector<vector<float>> train_inputs, val_inputs;
    vector<int> train_targets, val_targets;
    for (int i = 0; i < train_size; i++) {
        auto [x, y] = generate_case();
        train_inputs.push_back(x);
        train_targets.push_back(y);
    }
    for (int i = 0; i < val_size; i++) {
        auto [x, y] = generate_case();
        val_inputs.push_back(x);
        val_targets.push_back(y);
    }
    return {train_inputs, train_targets, val_inputs, val_targets};
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
