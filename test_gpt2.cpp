#include "boost/ut.hpp"
#define TESTING
#include <cmath>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <random>

#include "train_gpt2.cpp"  

using namespace boost; // NOLINT 

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float *wte;       // (V, C)
    float *wpe;       // (maxT, C)
    float *ln1w;      // (L, C)
    float *ln1b;      // (L, C)
    float *qkvw;      // (L, 3*C, C)
    float *qkvb;      // (L, 3*C)
    float *attprojw;  // (L, C, C)
    float *attprojb;  // (L, C)
    float *ln2w;      // (L, C)
    float *ln2b;      // (L, C)
    float *fcw;       // (L, 4*C, C)
    float *fcb;       // (L, 4*C)
    float *fcprojw;   // (L, C, 4*C)
    float *fcprojb;   // (L, C)
    float *lnfw;      // (C)
    float *lnfb;      // (C)
} ParameterTensors;   

bool similar(float a, float b) { return !std::isnan(a) && !std::isnan(b) && std::abs(a - b) < 1e-2; }

// poor man's tensor checker
int check_tensor(std::vector<float *> const &a, int m, float *b, int n, std::string const &label,
                 std::string const &indent = "") {
    const int print_upto = 5;
    int ok = 1;

    int okc = 0, notokc = 0;

    std::cout << indent << label << std::endl;
    for (int i = 0; i < n; i++) {
        const int i0 = i / m, i1 = i % m;
        if (similar(a[i0][i1], b[i])) {
            if (okc < print_upto) {
                std::cout << indent << "OK " << a[i0][i1] << " " << b[i] << "\n";
            }
            okc++;
        } else {
            if (notokc < print_upto) {
                std::cout << indent << "NOT OK " << a[i0][i1] << " " << b[i] << "\n";
            }
            notokc++;
            ok = 0;
        }
    }
    // print the final result
    if (ok) {
        std::cout << indent << "TENSOR OK" << std::endl;
    } else {
        std::cout << indent << "TENSOR NOT OK" << std::endl;
    }
    return ok;
}

int check_tensor(float *a, float *b, int n, std::string const &label, std::string const &indent = "") {
    return check_tensor({a}, n, b, n, label, indent);
}

int check_ml_tensor(std::vector<Block> const &blocks, std::function<Tensor *(const Block &)> const &select, float *b,
                    int n_layer, int n, std::string const &label) {
    const int m = n / n_layer;
    std::vector<float *> data(n_layer);
    for (int i = 0; i < n_layer; i++) {
        data[i] = (float *)select(blocks[i])->data();
    }
    const int ok = check_tensor(data, m, b, n, label);

    if (!ok) {
        // if not ok, find the first layer that is not ok
        for (int l = 0; l < n_layer; l++) {
            const int res = check_tensor((float *)select(blocks[l])->data(), b + (size_t)l * m, m,
                                         label + "-L" + std::to_string(l), "  ");
            if (!res) {
                break;
            }
        }
    }
    return ok;
}

// allocate memory for the parameters and point the individual tensors to the
// right places
float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float *params_memory = (float *)malloc(num_parameters * sizeof(float));
    // assign all the tensors
    float **ptrs[] = {&params->wte,      &params->wpe,      &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
                      &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw,  &params->fcb,
                      &params->fcprojw,  &params->fcprojb,  &params->lnfw, &params->lnfb};
    float *params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

void TestGPT2() {
    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    size_t C, V, Vp, maxT, L;
    C = model.config.channels;
    V = model.config.vocab_size;
    Vp = model.config.padded_vocab_size;
    maxT = model.config.max_seq_len;
    L = model.config.num_layers;

    // load additional information that we will use for debugging and error
    // checking
    FILE *state_file = fopen("gpt2_124M_debug_state.bin", "rb");
    if (state_file == nullptr) {
        throw std::runtime_error("Error opening state file");
    }
    int state_header[256];
    fread(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) {
        throw std::runtime_error("Bad magic state file");
    }
    if (state_header[1] != 2) {
        throw std::runtime_error("Bad version in state file");
    }
    const size_t B = state_header[2];
    const size_t T = state_header[3];
    std::cout << "[State]" << std::endl;
    std::cout << "batch_size: " << B << std::endl;
    std::cout << "seq_len: " << T << std::endl;

    size_t param_sizes[NUM_PARAMETER_TENSORS];
    // allocate space for all the parameters and read them in
    param_sizes[0] = Vp * C;
    param_sizes[1] = maxT * C;
    param_sizes[2] = L * C;
    param_sizes[3] = L * C;
    param_sizes[4] = L * (3 * C) * C;
    param_sizes[5] = L * (3 * C);
    param_sizes[6] = L * C * C;
    param_sizes[7] = L * C;
    param_sizes[8] = L * C;
    param_sizes[9] = L * C;
    param_sizes[10] = L * (4 * C) * C;
    param_sizes[11] = L * (4 * C);
    param_sizes[12] = L * C * (4 * C);
    param_sizes[13] = L * C;
    param_sizes[14] = C;
    param_sizes[15] = C;

    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }

    ParameterTensors expected_grads;
    float *expected_grads_memory = malloc_and_point_parameters(&expected_grads, param_sizes);

    // inputs and expected outputs, only used for error checking
    int *x = (int *)malloc(B * T * sizeof(int));
    int *y = (int *)malloc(B * T * sizeof(int));
    float *expected_logits = (float *)malloc(B * T * V * sizeof(float));
    float *expected_loss = (float *)malloc(1 * sizeof(float));

    fread(x, sizeof(int), B * T, state_file);
    fread(y, sizeof(int), B * T, state_file);
    fread(expected_logits, sizeof(float), B * T * V, state_file);
    fread(expected_loss, sizeof(float), 1, state_file);
    fread(expected_grads_memory, sizeof(float), num_parameters, state_file);
    fclose(state_file);

    bool allok = true;

    // expected losses are as follows, from Python
    const float expected_losses[10] = {5.270007133483887,  4.059706687927246,  3.3751230239868164, 2.8007826805114746,
                                       2.315382242202759,  1.8490285873413086, 1.3946564197540283, 0.9991465210914612,
                                       0.6240804195404053, 0.37651097774505615};

    for (int step = 0; step < 10; step++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        gpt2_forward(&model, x, y, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);

        clock_gettime(CLOCK_MONOTONIC, &end);

        const double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        if (step == 0) {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's
            // compare
            bool logits_ok = true;
            auto logits = model.logits->Flatten();
            for (int bt = 0; bt < B * T; bt++) {
                for (int v = 0; v < V; v++) {
                    const int i = bt * Vp + v;
                    const int ei = bt * V + v;
                    if (i < 10) {
                        std::cout << expected_logits[ei] << " " << logits[i] << std::endl;
                    }
                    if (!similar(expected_logits[ei], logits[i])) {
                        std::cout << "MISMATCH AT INDEX " << i << ": ";
                        std::cout << expected_logits[ei] << " " << logits[i] << std::endl;
                        logits_ok = false;
                        bt = B * T;  // break out of the loop
                        break;
                    }
                }
            }
            if (!logits_ok) {
                std::cout << "NOT ";
            }
            std::cout << "OK (LOGITS)" << std::endl;
            allok = allok && logits_ok;

            if (!similar(model.mean_loss, *expected_loss)) {
                std::cout << "LOSS MISMATCH: " << model.mean_loss << " " << *expected_loss << std::endl;
                allok = false;
            } else {
                std::cout << "OK (LOSS): " << model.mean_loss << " " << *expected_loss << std::endl;
            }

            // finally, compare the gradients
            bool gradoks[NUM_PARAMETER_TENSORS];
            gradoks[0] = check_tensor((float *)model.embedding.wte->grad()->data(), expected_grads.wte,
                                      param_sizes[0], "dwte");
            gradoks[1] = check_tensor((float *)model.embedding.wpe->grad()->data(), expected_grads.wpe,
                                      param_sizes[1], "dwpe");
            gradoks[2] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.ln1w->grad(); }, expected_grads.ln1w, L,
                param_sizes[2], "dln1w");
            gradoks[3] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.ln1b->grad(); }, expected_grads.ln1b, L,
                param_sizes[3], "dln1b");
            gradoks[4] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.qkvw->grad(); }, expected_grads.qkvw, L,
                param_sizes[4], "dqkvw");
            gradoks[5] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.qkvb->grad(); }, expected_grads.qkvb, L,
                param_sizes[5], "dqkvb");
            gradoks[6] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.attprojw->grad(); }, expected_grads.attprojw, L,
                param_sizes[6], "dattprojw");
            gradoks[7] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.attprojb->grad(); }, expected_grads.attprojb, L,
                param_sizes[7], "dattprojb");
            gradoks[8] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.ln2w->grad(); }, expected_grads.ln2w, L,
                param_sizes[8], "dln2w");
            gradoks[9] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.ln2b->grad(); }, expected_grads.ln2b, L,
                param_sizes[9], "dln2b");
            gradoks[10] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.fcw->grad(); }, expected_grads.fcw, L,
                param_sizes[10], "dfcw");
            gradoks[11] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.fcb->grad(); }, expected_grads.fcb, L,
                param_sizes[11], "dfcb");
            gradoks[12] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.fcprojw->grad(); }, expected_grads.fcprojw, L,
                param_sizes[12], "dfcprojw");
            gradoks[13] = check_ml_tensor(
                model.blocks, [](Block const &block) { return block.fcprojb->grad(); }, expected_grads.fcprojb, L,
                param_sizes[13], "dfcprojb");
            gradoks[14] = check_tensor((float *)model.lm_head.lnfw->grad()->data(), expected_grads.lnfw,
                                       param_sizes[14], "dlnfw");
            gradoks[15] = check_tensor((float *)model.lm_head.lnfb->grad()->data(), expected_grads.lnfb,
                                       param_sizes[15], "dlnfb");

            bool gradok = true;
            for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
                gradok = gradok && gradoks[i];
            }

            if (!gradok) {
                std::cout << "NOT OK (GRADIENTS)" << std::endl;
            } else {
                std::cout << "OK (GRADIENTS)" << std::endl;
            }

            allok = allok && gradok;
        }

        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.01f, step + 1);

        const float expected_loss = expected_losses[step];
        const float actual_loss = model.mean_loss;
        const bool step_loss_ok = similar(expected_loss, actual_loss);
        allok = allok && step_loss_ok;

        std::cout << "Step " << step << ": loss " << model.mean_loss << " (took " << time_elapsed_s * 1000 << " ms) ";
        if (step_loss_ok) {
            std::cout << "OK" << std::endl;
        } else {
            std::cout << "NOT OK, expected loss " << expected_loss << std::endl;
        }

        std::cout << "\n\n";
        // print profiling information
        std::cout << "Forward ";
        Tensor::ForwardProfile.Print();
        std::cout << "Backward ";
        Tensor::BackwardProfile.Print();
        std::cout << "\n\n";
    }

    if (allok) {
        std::cout << "All OK" << std::endl;
    } else {
        std::cout << "NOT OK" << std::endl;
    }

    free(x);
    free(y);
    free(expected_logits);
    free(expected_loss);
    free(expected_grads_memory);
    gpt2_free(&model);
}

int main() {
    try {
        TestGPT2();
    } catch (std::exception const &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}
