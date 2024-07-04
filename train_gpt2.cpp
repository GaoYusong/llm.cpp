#include <unistd.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#include "llmc/dataloader.h"
#include "llmc/tokenizer.h"
#include "tinytorch.hpp"

// -------------------------------------------------------------
// GPT2

using namespace tinytorch;  // NOLINT

struct GPT2Config {
    int max_seq_len;
    int vocab_size;         // vocab size, e.g. 50257
    int padded_vocab_size;  // padded to e.g. %128==0, 50304
    int num_layers;
    int num_heads;
    int channels;
};

struct Block {
    Tensor *ln1w;      // layernorm weights, (channels)
    Tensor *ln1b;      // layernorm biases, (channels)
    Tensor *qkvw;      // query, key, value weights, (3 * channels, channels)
    Tensor *qkvb;      // query, key, value biases, (3 * channels)
    Tensor *attprojw;  // attention projection weights, (channels, channels)
    Tensor *attprojb;  // attention projection biases, (channels)
    Tensor *ln2w;      // layernorm weights, (channels)
    Tensor *ln2b;      // layernorm biases, (channels)
    Tensor *fcw;       // fully connected weights, (4 * channels, channels)
    Tensor *fcb;       // fully connected biases, (4 * channels)
    Tensor *fcprojw;   // fully connected projection weights, (channels, 4 * channels)
    Tensor *fcprojb;   // fully connected projection biases, (channels)
};

struct Embedding {
    Tensor *wte;  // word token embeddings, (padded_vocab_size, channels)
    Tensor *wpe;  // word position embeddings, (max_seq_len, channels)
};

struct LMHead {
    Tensor *lnfw;  // layernorm weights, (channels)
    Tensor *lnfb;  // layernorm biases, (channels)
};

struct GPT2 {
    GPT2Config config;
    TensorContext *ctx{nullptr};  // tensor memory context for the model

    // the wegiths of the model
    struct Embedding embedding;    // the embedding layer
    std::vector<Block> blocks;     // the transformer blocks
    struct LMHead lm_head;         // the language model head
    std::vector<Tensor *> params;  // all the parameters of the model, the layout is the same as the checkpoint file
    int num_parameters;

    int batch_size;  // the batch size (B) of current forward pass
    int seq_len;     // the sequence length (T) of current forward pass

    Tensor *input{nullptr};      // the input tensor, (B, T)
    Tensor *input_pos{nullptr};  // the input position tensor, (B, T)
    Tensor *target{nullptr};     // the target tensor, (B, T)

    Tensor *logits{nullptr};  // the logits tensor, (B, T, padded_vocab_size)
    Tensor *probs{nullptr};   // the probs tensor, (B, T, padded_vocab_size)
    Tensor *losses{nullptr};  // the losses tensor, (B, T)
    float mean_loss;

    // buffers for the AdamW optimizer
    std::vector<float> m_memory;
    std::vector<float> v_memory;
};

void gpt2_build_from_checkpoint(GPT2 *model, const std::string &checkpoint_path) {
    FILE *model_file = fopen(checkpoint_path.c_str(), "rb");
    if (model_file == nullptr) {
        throw std::runtime_error("Could not open the model checkpoint file: " + checkpoint_path);
    }

    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) {
        throw std::runtime_error("Bad magic number in model checkpoint file: " + checkpoint_path);
    }
    if (model_header[1] != 3) {
        throw std::runtime_error("Bad version number in model checkpoint file: " + checkpoint_path);
    }

    // read in hyperparameters
    int maxT, V, Vp, L, NH, C;
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];

    // Print the hyperparameters
    std::cout << "[GPT-2]:" << std::endl;
    std::cout << "max_seq_len: " << maxT << std::endl;
    std::cout << "vocab_size: " << V << std::endl;
    std::cout << "padded_vocab_size: " << Vp << std::endl;
    std::cout << "num_layers: " << L << std::endl;
    std::cout << "num_heads: " << NH << std::endl;
    std::cout << "channels: " << C << std::endl;

    // initialize the parameter tensor sizes
    assert(model->ctx == nullptr);
    TensorContext *ctx;
    model->ctx = ctx = new TensorContext((size_t)8 * 1024 * 1024 * 1024);

    model->embedding.wte = ctx->NewTensor({Vp, C});
    model->embedding.wpe = ctx->NewTensor({maxT, C});
    model->params.insert(model->params.end(), {model->embedding.wte, model->embedding.wpe});

    std::vector<std::vector<Tensor *>> block_params;
    for (int l = 0; l < L; l++) {
        auto &blocks = model->blocks;
        blocks.emplace_back(Block{.ln1w = ctx->NewTensor({C}),
                                  .ln1b = ctx->NewTensor({C}),
                                  .qkvw = ctx->NewTensor({3 * C, C}),
                                  .qkvb = ctx->NewTensor({3 * C}),
                                  .attprojw = ctx->NewTensor({C, C}),
                                  .attprojb = ctx->NewTensor({C}),
                                  .ln2w = ctx->NewTensor({C}),
                                  .ln2b = ctx->NewTensor({C}),
                                  .fcw = ctx->NewTensor({4 * C, C}),
                                  .fcb = ctx->NewTensor({4 * C}),
                                  .fcprojw = ctx->NewTensor({C, 4 * C}),
                                  .fcprojb = ctx->NewTensor({C})});

        block_params.push_back({blocks[l].ln1w, blocks[l].ln1b, blocks[l].qkvw, blocks[l].qkvb, blocks[l].attprojw,
                                blocks[l].attprojb, blocks[l].ln2w, blocks[l].ln2b, blocks[l].fcw, blocks[l].fcb,
                                blocks[l].fcprojw, blocks[l].fcprojb});
    };

    // NOTICE: the order of the parameters in the checkpoint file is one parameter for all layers, then the next
    // parameter for all layers, etc.
    for (int i = 0; i < block_params[0].size(); i++) {
        for (int l = 0; l < L; l++) {
            model->params.push_back(block_params[l][i]);
        }
    }

    model->lm_head.lnfw = ctx->NewTensor({C});
    model->lm_head.lnfb = ctx->NewTensor({C});
    model->params.insert(model->params.end(), {model->lm_head.lnfw, model->lm_head.lnfb});

    // load the parameters
    model->num_parameters = 0;
    for (auto t : model->params) {
        model->num_parameters += t->NumElements();
        fread(t->data(), sizeof(float), t->NumElements(), model_file);
    }
    fclose(model_file);

    std::cout << "Number of Parameters: " << model->num_parameters << std::endl;

    ctx->PrintLayout();
    std::cout << "Checkpoint loaded successfully!" << std::endl;
}

void gpt2_forward(GPT2 *model, const int *inputs, const int *targets, int B, int T) {  // NOLINT
    if (model->num_parameters == 0) {
        throw std::runtime_error("Model has not been initialized");
    }

    int maxT, V, Vp, L, NH, C, HS;
    maxT = model->config.max_seq_len;
    V = model->config.vocab_size;
    Vp = model->config.padded_vocab_size;
    L = model->config.num_layers;
    NH = model->config.num_heads;
    C = model->config.channels;
    assert(C % NH == 0);
    HS = C / NH;

    auto ctx = model->ctx;

    // first time forward pass, create the computation graph
    if (model->input == nullptr) {
        model->batch_size = B;
        model->seq_len = T;
        model->input = ctx->NewTensor({B, T}, kI32);
        // create the position tensor
        std::vector<int> pos_p((size_t)B * T);
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < T; j++) {
                pos_p[i * T + j] = j;
            }
        }
        model->input_pos = ctx->NewTensor({B, T}, kI32)->Fill(pos_p);

        model->target = ctx->NewTensor({B, T}, kI32);

        auto &encoded = (*model->embedding.wte)[*model->input] + (*model->embedding.wpe)[*model->input_pos];
        auto *residual = &encoded;
        for (int l = 0; l < L; l++) {
            auto &block = model->blocks[l];
            auto &ln1 = residual->Norm() * *block.ln1w + *block.ln1b;  // (B, T, C)
            auto &qkv = ln1.MatMul(*block.qkvw) + *block.qkvb;         // (B, T, 3 * C)
            const auto &qkv_split = qkv.Split(C, 2);

            // multi-head attention
            auto &q = qkv_split[0]->View({B, T, NH, HS}).Transpose(1, 2);
            auto &k = qkv_split[1]->View({B, T, NH, HS}).Transpose(1, 2);
            auto &v = qkv_split[2]->View({B, T, NH, HS}).Transpose(1, 2);
            auto attn = &(q.MatMul(k) * (1.0f / sqrtf(HS)));  // (B, NH, T, T)
            // mask out the future tokens
            attn = &attn->Softmax(true);
            attn = &attn->MatMul(v.Transpose(2, 3));                            // (B, NH, T, HS)
            attn = &attn->Transpose(1, 2).View({B, T, C});                      // (B, T, C)
            auto &attn_proj = attn->MatMul(*block.attprojw) + *block.attprojb;  // (B, T, C)

            auto &residual2 = *residual + attn_proj;                   // (B, T, C)
            auto &ln2 = residual2.Norm() * *block.ln2w + *block.ln2b;  // (B, T, C)

            // feed forward
            auto &fc = ln2.MatMul(*block.fcw) + *block.fcb;                // (B, T, 4 * C)
            auto &gelu = fc.Gelu();                                        // (B, T, 4 * C)
            auto &fc_proj = gelu.MatMul(*block.fcprojw) + *block.fcprojb;  // (B, T, C)
            auto &residual3 = residual2 + fc_proj;                         // (B, T, C)

            residual = &residual3;
        }
        auto &lnf = residual->Norm() * *model->lm_head.lnfw + *model->lm_head.lnfb;  // (B, T, C)
        model->logits = &lnf.MatMul(*model->embedding.wte);                          // (B, T, Vp)
        model->probs = &model->logits->Softmax(false, V);                            // (B, T, Vp)

        model->losses = &model->probs->CrossEntropy(*model->target);  // (B, T)

        ctx->PrintLayout();
        std::cout << "Computation Graph created successfully!" << std::endl;

    } else {
        if (model->batch_size != B || model->seq_len != T) {
            if (targets != nullptr) {
                throw std::runtime_error("Dynamic batch size or sequence length not supported");
            } else {
                // model inference
                if (B > model->batch_size || T > model->seq_len) {
                    throw std::runtime_error("Too large batch size or sequence length for the current model");
                }
            }
        }
    }

    // fill the input tensor
    model->input->Fill(inputs);

    if (targets != nullptr) {
        // fill the target tensor
        model->target->Fill(targets);
        model->losses->Forward();

        float mean_loss = 0.0f;
        for (int i = 0; i < B * T; i++) {
            mean_loss += *((float *)model->losses->data() + i);
        }
        mean_loss /= B * T;
        model->mean_loss = mean_loss;
    } else {
        model->probs->Forward();
        model->mean_loss = -1.0f;
    }
}
void gpt2_backward(GPT2 *model) {
    float dloss_mean = 1.0f / (model->batch_size * model->seq_len);
    model->losses->Backward(true, dloss_mean);
}

void gpt2_zero_grad(GPT2 *model) { model->losses->ZeroGrad(); }

void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    if (model->m_memory.empty()) {
        assert(model->num_parameters > 0);
        model->m_memory = std::vector<float>(model->num_parameters, 0.0f);
        model->v_memory = std::vector<float>(model->num_parameters, 0.0f);
    }

    int idx = 0;
    for (auto param : model->params) {
        auto weights = (float *)param->data();
        auto grads = (float *)param->grad()->data();

        for (int i = 0; i < param->NumElements(); i++) {
            auto w = weights[i], g = grads[i];

            // update the first moment (momentum)
            float m = beta1 * model->m_memory[idx] + (1.0f - beta1) * g;
            // update the second moment (RMSprop)
            float v = beta2 * model->v_memory[idx] + (1.0f - beta2) * g * g;
            // bias-correct both moments
            float m_hat = m / (1.0f - powf(beta1, t));
            float v_hat = v / (1.0f - powf(beta2, t));

            // update
            model->m_memory[idx] = m;
            model->v_memory[idx] = v;
            weights[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w);

            idx++;
        }
    }
}

void gpt2_free(GPT2 *model) { delete model->ctx; }

#ifndef TESTING

// -------------------------------------------------------------
// sampler

uint32_t random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

// random float32 in [0, 1)
float random_f32(uint64_t *state) { return (random_u32(state) >> 8) / 16777216.0f; }

int sample_mult(float *probs, int n, float coin) {
    // sample index from probs (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probs[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;  // in case of rounding errors
}

// -------------------------------------------------------------

int main() {  // NOLINT

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // build the training set and validation set data loaders
    const std::string tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const std::string tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const std::string &train_token = tiny_shakespeare_train;
    const std::string &val_token = tiny_shakespeare_val;
    size_t B = 4;
    size_t T = 64;
    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_token.c_str(), B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_token.c_str(), B, T, 0, 1, 0);
    std::cout << "train dataset num_batches: " << train_loader.num_tokens / (B * T) << std::endl;
    std::cout << "val dataset num_batches: " << val_loader.num_tokens / (B * T) << std::endl;
    int val_num_batches = 5;

    // build the tokenizer from the tokenizer file
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // some memory for generating samples from the model
    uint64_t rng_state = 1337;
    const int gen_max_length = 64;
    int gen_tokens[B * T];

    // train the model
    struct timespec start, end;
    for (int step = 0; step <= 40; step++) {
        // once in a while, estimate the validation loss
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            std::cout << "val loss: " << val_loss << std::endl;
        }

        // once in a while do model inference to print generated text
        if (step > 0 && step % 20 == 0) {
            for (int i = 0; i < B * T; i++) {
                gen_tokens[i] = tokenizer.eot_token;
            }

            std::cout << "generating:\n---\n";
            for (int t = 1; t < gen_max_length; t++) {
                gpt2_forward(&model, gen_tokens, nullptr, B, T);
                float *probs = (float *)model.probs->data() + (size_t)(t - 1) * model.config.padded_vocab_size;
                float coin = random_f32(&rng_state);
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;
                if (tokenizer.init_ok) {
                    const char *token_str = tokenizer_decode(&tokenizer, next_token);
                    // TODO(ysg): resolve the mixed printf and std::cout
                    safe_printf(token_str);
                } else {
                    std::cout << next_token << " ";
                }
                std::cout << std::flush;
            }
            std::cout << "\n---\n";
        }

        // do a training step
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step + 1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        std::cout << "step " << step << " train Loss: " << model.mean_loss << " (took " << time_elapsed_s * 1000
                  << " ms)" << std::endl;
    }

    tokenizer_free(&tokenizer);
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    gpt2_free(&model);
    return 0;
}

#endif