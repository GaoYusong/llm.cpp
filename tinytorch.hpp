#pragma once

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <random>
#include <ranges>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace tinytorch {

const size_t kTensorMemAlign = 16;

template <typename T>
inline void AssertAligned(T ptr) {
    assert(((uintptr_t)(ptr)) % kTensorMemAlign == 0);
}

enum TensorType {
    kF32,
    kI32,
    kLEN  // number of tensor types
};

const size_t kTypeSize[kLEN] = {
    sizeof(float),
    sizeof(int32_t),
};

template <typename T>
inline bool IsTypeCompatible(TensorType type) {
    switch (type) {
        case kF32:
            return std::is_same<T, float>::value;
        case kI32:
            return std::is_same<T, int32_t>::value;
        default:
            throw std::runtime_error("Malformed tensor type");
    }
}

struct Object {
    size_t offset;
    size_t size;

    Object *next;

    std::byte padding[8];
};

const size_t kObjectSize = sizeof(struct Object);

template <typename T>
class TensorContextT {
   public:
    const size_t TENSOR_SIZE = sizeof(T);

    explicit TensorContextT(size_t mem_size) : mem_size_(mem_size) {
        mem_buffer_ = new std::byte[mem_size];
        n_objects_ = 0;
        objects_begin_ = nullptr;
        objects_end_ = nullptr;
    }

    ~TensorContextT() { delete[] mem_buffer_; }

    T *NewTensor(const std::vector<int> &dims, float *data) {
        return NewTensor(dims, kF32, reinterpret_cast<std::byte *>(data));
    }

    T *NewTensor(const std::vector<int> &dims, TensorType type = kF32, std::byte *data = nullptr) {
        const int n_dims = dims.size();

        size_t size_needed = 0;
        if (data == nullptr) {
            size_t data_size = kTypeSize[type];
            for (int i = 0; i < n_dims; i++) {
                data_size *= dims[i];
            }
            size_needed += ((data_size + kTensorMemAlign - 1) / kTensorMemAlign) * kTensorMemAlign;
        }
        size_needed += TENSOR_SIZE;

        // layout
        // [Struct Object][Struct Tensor][data]
        std::byte *cur = mem_buffer_;
        if (objects_end_ != nullptr) {
            cur += objects_end_->offset + objects_end_->size;
        }

        if (cur + size_needed + kObjectSize > mem_buffer_ + mem_size_) {
            throw std::runtime_error("Out of tensor memory");
        }

        Object *object = reinterpret_cast<Object *>(cur);

        *object = {.offset = (size_t)(cur - mem_buffer_) + kObjectSize,
                   .size = size_needed,
                   .next = nullptr};

        AssertAligned(object);

        if (objects_end_ != nullptr) {
            objects_end_->next = object;
        } else {
            objects_begin_ = object;
        }
        objects_end_ = object;

        T *tensor = reinterpret_cast<T *>(cur + kObjectSize);

        AssertAligned(tensor);

        *tensor = T(this, dims, type, data == nullptr ? cur + kObjectSize + TENSOR_SIZE : data);

        AssertAligned(tensor->data_);

        n_objects_++;

        return tensor;
    }

    void PrintLayout(bool verbose = false) {
        std::cout << "TensorContext Layout" << std::endl;
        std::cout << "---------------------" << std::endl;
        std::cout << "Total memory size: " << mem_size_ << std::endl;
        std::cout << "Used  memory size: "
                  << (objects_end_ == nullptr ? 0 : (objects_end_->offset + objects_end_->size))
                  << std::endl;
        std::cout << "Number of objects: " << n_objects_ << std::endl;
        if (verbose) {
            std::cout << "Objects:" << std::endl;
            Object *cur = objects_begin_;
            while (cur != nullptr) {
                std::cout << "  offset: " << cur->offset << ", size: " << cur->size << std::endl;
                cur = cur->next;
            }
        }
    }

   private:
    size_t mem_size_;
    std::byte *mem_buffer_;

    int n_objects_;

    Object *objects_begin_;
    Object *objects_end_;
};

class Tensor;
using TensorContext = TensorContextT<Tensor>;

class NormalDist {
   public:
    NormalDist() { generator_.seed(std::random_device{}()); }

    float operator()() { return normal_dist_(generator_); }

   private:
    std::normal_distribution<float> normal_dist_;

    std::default_random_engine generator_;
};

enum TensorOp {
    kOpNone,
    kOpAdd,
    kOpMul,
    kOpMatmul,
    kOpLookup,
    kOpNorm,
    kOpBroadcast,
    kOpView,
    kOpTranspose,
    kOpGelu,
    kOpSoftmax,
    kOpCrossEntropy,
    kOpLEN  // number of tensor operations
};

const std::string TENSOR_OP_NAMES[kOpLEN] = { // NOLINT
    "NONE",      "ADD",  "MUL",       "MATMUL", "LOOKUP",  "NORM",
    "BROADCAST", "VIEW", "TRANSPOSE", "GELU",   "SOFTMAX", "CROSS_ENTROPY",
};

const int kMaxTensorDims = 4;
const int kMaxTensorOpParams = 2;

// Tensor operations time profile
class Profile {
   public:
    void Reset() {
        times_.clear();
        counts_.clear();
    }

    void AddTime(TensorOp op, double time) {
        times_[op] += time;
        counts_[op]++;
    }

    void Print() {
        std::cout << "Profile" << std::endl;
        std::cout << "-------" << std::endl;
        for (int i = 0; i < kOpLEN; i++) {
            if (counts_[i] > 0) {
                std::cout << TENSOR_OP_NAMES[i] << ": " << times_[i] * 1000 << "ms ("
                          << (times_[i] / counts_[i] * 1000) << "ms per op, " << counts_[i]
                          << " times)" << std::endl;
            }
        }
    }

   private:
    std::unordered_map<int, double> times_;
    std::unordered_map<int, size_t> counts_;
};

class Tensor {
   public:
    // Add
    Tensor &operator+(Tensor &other) { return operator2(other, kOpAdd); }

    // Mul
    Tensor &operator*(Tensor &other) { return operator2(other, kOpMul); }

    // Mul by scalar
    Tensor &operator*(float val) {
        assert(type_ == kF32);
        return *this * *(ctx_->NewTensor({1}, type_)->Fill(val));
    }

   private:
    Tensor &operator2(Tensor &other_ref, TensorOp op) {
        auto other = &other_ref;
        assert(other != this);
        if (!SameShape(*other)) {
            assert(can_broadcast_to(*other, *this));
            other = &broadcast_to(ctx_, *other, *this);
        }
        Tensor *dst = ctx_->NewTensor(Dims());
        dst->op_ = op;
        dst->src0_ = this;
        dst->src1_ = other;
        return *dst;
    }

   public:
    // Lookup
    Tensor &operator[](Tensor &index) {
        assert(index.type_ == kI32);
        assert(n_dims_ + index.n_dims_ - 1 <= kMaxTensorDims);
        std::vector<int> ds;
        for (int i = kMaxTensorDims - 1; i > kMaxTensorDims - n_dims_; i--) {
            ds.push_back(dims_[i]);
        }
        for (int i = kMaxTensorDims - 1; i >= kMaxTensorDims - index.n_dims_; i--) {
            ds.push_back(index.dims_[i]);
        }
        reverse(ds.begin(), ds.end());
        Tensor *dst = ctx_->NewTensor(ds, type_);
        dst->op_ = kOpLookup;
        dst->src0_ = this;
        dst->src1_ = &index;
        return *dst;
    }

    // Norm
    Tensor &Norm() {
        assert(type_ == kF32);
        Tensor *dst = ctx_->NewTensor(Dims(), type_);
        dst->op_ = kOpNorm;
        dst->src0_ = this;
        return *dst;
    }

    // Gelu
    Tensor &Gelu() {
        assert(type_ == kF32);
        Tensor *dst = ctx_->NewTensor(Dims(), type_);
        dst->op_ = kOpGelu;
        dst->src0_ = this;
        return *dst;
    }

    // Softmax
    Tensor &Softmax(bool is_casual = false, int vocab_size = 0) {
        assert(type_ == kF32);
        if (vocab_size > 0) {
            assert(vocab_size <= dims_[kMaxTensorDims - 1]);
        }
        Tensor *dst = ctx_->NewTensor(Dims(), type_);
        dst->op_ = kOpSoftmax;
        dst->src0_ = this;
        dst->op_params_[0] = is_casual;
        dst->op_params_[1] = vocab_size;
        return *dst;
    }

    // CrossEntropy
    Tensor &CrossEntropy(Tensor &target) {
        assert(type_ == kF32 && target.type_ == kI32);
        auto shape = Dims();
        shape.pop_back();
        assert(shape == target.Dims());
        Tensor *dst = ctx_->NewTensor(shape, type_);
        dst->op_ = kOpCrossEntropy;
        dst->src0_ = this;
        dst->src1_ = &target;
        return *dst;
    }

    // Split
    std::vector<Tensor *> Split(int size, int axis) {
        assert(axis < n_dims_);
        auto dimi = kMaxTensorDims - n_dims_ + axis;
        assert(dims_[dimi] % size == 0);
        std::vector<Tensor *> tensors;
        if (dims_[dimi] == size) {
            tensors.push_back(this);
            return tensors;
        }

        std::vector<int> shape = Dims();
        shape[axis] = size;
        for (int i = 0; i < dims_[dimi] / size; i++) {
            tensors.push_back(&View(shape, i, axis));
        }
        return tensors;
    }

    // View
    // TODO(ysg): the view is actually a copy, we need to implement a real view
    Tensor &View(const std::vector<int> &shape, int split_no = 0, int split_axis = 0) {
        assert(NumElements() % num_of_elements(shape) == 0);
        int dimi = kMaxTensorDims - n_dims_ + split_axis;
        assert(dims_[dimi] % (NumElements() / num_of_elements(shape)) == 0);
        Tensor *dst = ctx_->NewTensor(shape, type_);
        dst->op_ = kOpView;
        dst->src0_ = this;
        dst->op_params_[0] = split_no;
        dst->op_params_[1] = split_axis;
        return *dst;
    }

    // Tranpose
    Tensor &Transpose(int axis0, int axis1) {
        assert(axis0 < n_dims_ && axis1 < n_dims_);
        auto dimi0 = kMaxTensorDims - n_dims_ + axis0;
        auto dimi1 = kMaxTensorDims - n_dims_ + axis1;
        std::vector<int> shape = Dims();
        std::swap(shape[axis0], shape[axis1]);
        Tensor *dst = ctx_->NewTensor(shape, type_);
        dst->op_ = kOpTranspose;
        dst->src0_ = this;
        dst->op_params_[0] = dimi0;
        dst->op_params_[1] = dimi1;
        return *dst;
    }

    // Matmul
    // (B, M, N) x (B, P, N) -> (B, M, P)
    // we assume that the input tensors are in the format (B, M, N) and (B, P, N)
    Tensor &MatMul(Tensor &other_ref) {
        auto other = &other_ref;
        assert(other != this);
        if (!can_matmul(*this, *other)) {
            assert(can_broadcast_to(*other, *this, 2));
            other = &broadcast_to(ctx_, *other, *this, 2);
            assert(can_matmul(*this, *other));
        }
        std::vector<int> dst_dims = {dims_[0], dims_[1], dims_[2], other->dims_[2]};
        dst_dims.erase(dst_dims.begin(), dst_dims.begin() + dst_dims.size() - n_dims_);

        Tensor *dst = ctx_->NewTensor(dst_dims);
        dst->op_ = kOpMatmul;
        dst->src0_ = this;
        dst->src1_ = other;
        return *dst;
    }

    void Forward() {
        std::vector<Tensor *> sorted = topo_sort(this);

        struct timespec start, end;
        ForwardProfile.Reset();
        for (auto *t : sorted) {
            clock_gettime(CLOCK_MONOTONIC, &start);
            switch (t->op_) {
                case kOpAdd:
                    add_forward(t, t->src0_, t->src1_);
                    break;
                case kOpMul:
                    mul_forward(t, t->src0_, t->src1_);
                    break;
                case kOpMatmul:
                    matmul_forward(t, t->src0_, t->src1_);
                    break;
                case kOpLookup:
                    lookup_forward(t, t->src0_, t->src1_);
                    break;
                case kOpNorm:
                    norm_forward(t, t->src0_);
                    break;
                case kOpTranspose:
                    transpose_forward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
                    break;
                case kOpView:
                    view_forward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
                    break;
                case kOpBroadcast:
                    broadcast_forward(t, t->src0_);
                    break;
                case kOpGelu:
                    gelu_forward(t, t->src0_);
                    break;
                case kOpSoftmax:
                    softmax_forward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
                    break;
                case kOpCrossEntropy:
                    cross_entropy_forward(t, t->src0_, t->src1_);
                    break;
                case kOpNone:
                    // no-op
                    break;
                default:
                    throw std::runtime_error("Forward(): Not implemented, " +
                                             TENSOR_OP_NAMES[t->op_]);
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            ForwardProfile.AddTime(t->op_,
                                   end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9);
        }
    }

    void Backward(bool init_grad = true, float init_val = 1.0f) {
        std::vector<Tensor *> sorted = topo_sort(this);

        if (init_grad) {
            AllocGrad(false)->grad()->Fill(init_val);
        }

        struct timespec start, end;
        BackwardProfile.Reset();
        for (auto *t : sorted | std::ranges::views::reverse) {
            clock_gettime(CLOCK_MONOTONIC, &start);
            switch (t->op_) {
                case kOpAdd:
                    add_backward(t, t->src0_, t->src1_);
                    break;
                case kOpMul:
                    mul_backward(t, t->src0_, t->src1_);
                    break;
                case kOpMatmul:
                    matmul_backward(t, t->src0_, t->src1_);
                    break;
                case kOpLookup:
                    lookup_backward(t, t->src0_, t->src1_);
                    break;
                case kOpNorm:
                    norm_backward(t, t->src0_);
                    break;
                case kOpTranspose:
                    transpose_backward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
                    break;
                case kOpView:
                    view_backward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
                    break;
                case kOpBroadcast:
                    broadcast_backward(t, t->src0_);
                    break;
                case kOpGelu:
                    gelu_backward(t, t->src0_);
                    break;
                case kOpSoftmax:
                    softmax_backward(t, t->src0_, t->op_params_[0], t->op_params_[1]);
                    break;
                case kOpCrossEntropy:
                    cross_entropy_backward(t, t->src0_, t->src1_);
                    break;
                case kOpNone:
                    // no-op
                    break;
                default:
                    throw std::runtime_error("Backward(): Not implemented, " +
                                             TENSOR_OP_NAMES[t->op_]);
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            BackwardProfile.AddTime(
                t->op_, end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9);
        }
    }

    void ZeroGrad() {
        std::vector<Tensor *> sorted = topo_sort(this);

        for (auto t : sorted) {
            if (t->grad_ != nullptr) {
                t->grad_->Fill(0.0f);
            }
        }
    }

    void PrintTensor(bool include_data = true, size_t sample_size = 10) {
        std::cout << "Tensor" << std::endl;
        std::cout << "------" << std::endl;
        std::cout << "n_dims: " << n_dims_ << std::endl;
        std::cout << "dims: ";
        for (int i = kMaxTensorDims - n_dims_; i < kMaxTensorDims; i++) {
            std::cout << dims_[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "stride: ";
        for (int i = kMaxTensorDims - n_dims_; i < kMaxTensorDims; i++) {
            std::cout << strides_[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "op: " << TENSOR_OP_NAMES[op_] << "(" << this << ")" << std::endl;
        if (src0_ != nullptr) {
            std::cout << "src0: " << TENSOR_OP_NAMES[src0_->op_] << "(" << src0_ << ")"
                      << std::endl;
        }
        if (src1_ != nullptr) {
            std::cout << "src1: " << TENSOR_OP_NAMES[src1_->op_] << "(" << src1_ << ")"
                      << std::endl;
        }

        if (include_data) {
            std::cout << "data: \n";
            size_t upto = std::min(n_vec(), sample_size);

            for (size_t i = 0; i < upto; i++) {
                vec_print(vsize(), type_, data_ + i * vstride() * kTypeSize[type_]);
                std::cout << std::endl;
            }

            if (grad_ != nullptr) {
                std::cout << "grad: \n";
                for (size_t i = 0; i < upto; i++) {
                    vec_print(grad_->vsize(), type_,
                              grad_->data_ + i * grad_->vstride() * kTypeSize[type_]);
                    std::cout << std::endl;
                }
            }
        }
    }

   public:
    TensorType type() { return type_; }

    std::byte *data() { return data_; }

    inline Tensor *grad() { return grad_; }

    // just for testing
    inline Tensor *RandomGrad() {
        grad_ = ctx_->NewTensor(Dims())->RandomNorm();
        return this;
    }

    // just for testing
    inline Tensor *FillGrad(float *data) {
        assert(grad_ == nullptr);
        grad_ = ctx_->NewTensor(Dims())->Fill(data);
        return this;
    }

    inline Tensor *AllocGrad(bool init = true) {
        if (grad_ == nullptr) {
            grad_ = ctx_->NewTensor(Dims());
            if (init) {
                grad_->Fill(0.0f);
            }
        }
        return this;
    }

    inline Tensor *CopyDataFrom(const Tensor &other) {
        assert(SameShape(other));
        memcpy(data_, other.data_, NumElements() * kTypeSize[type_]);
        return this;
    }

    std::vector<Tensor *> Tensors() { return topo_sort(this); }

    inline std::vector<float> Flatten() const { return Flatten<float>(); }

    template <typename T>
    inline std::vector<T> Flatten() const {
        assert(IsTypeCompatible<T>(type_));
        assert(IsContiguous());
        if (data_ == nullptr) {
            return {};
        }
        T *ptr = (T *)data_;
        std::vector<T> vec(ptr, ptr + NumElements());
        return vec;
    }

    inline size_t NumElements() const {
        static_assert(kMaxTensorDims == 4, "MAX_TENSOR_DIMS is not 4 - update this function");
        return (size_t)dims_[0] * dims_[1] * dims_[2] * dims_[3];
    }

    template <typename T>
    Tensor *Fill(const std::vector<T> &in_data) {
        assert(in_data.size() == NumElements());
        return Fill(in_data.data());
    }

    template <typename T>
    typename std::enable_if<!std::is_pointer<T>::value, Tensor *>::type Fill(T val) {
        assert(IsTypeCompatible<T>(type_));
        for (size_t i = 0; i < n_vec(); i++) {
            vec_fill(vsize(), (T *)data_ + i * vstride(), val);
        }
        return this;
    }

    template <typename T>
    typename std::enable_if<std::is_scalar<T>::value, Tensor *>::type Fill(const T *in_data) {
        assert(IsTypeCompatible<T>(type_));
        assert(IsContiguous());
        for (size_t i = 0; i < n_vec(); i++) {
            vec_fill(vsize(), (T *)data_ + i * vstride(), in_data + i * vstride());
        }
        return this;
    }

    Tensor *RandomNorm() {
        assert(type_ == kF32);
        assert(IsContiguous());
        for (size_t i = 0; i < n_vec(); i++) {
            vec_random_norm(vsize(), (float *)data_ + i * vstride());
        }
        return this;
    }

    inline std::vector<int> Dims() const {
        return std::vector<int>(dims_ + kMaxTensorDims - n_dims_, dims_ + kMaxTensorDims);
    }

    inline std::vector<size_t> Strides() const {
        return std::vector<size_t>(strides_ + kMaxTensorDims - n_dims_, strides_ + kMaxTensorDims);
    }

    inline bool IsContiguous() const {
        static_assert(kMaxTensorDims == 4, "MAX_TENSOR_DIMS is not 4 - update this function");
        return strides_[3] == 1 && strides_[2] == strides_[3] * dims_[3] &&
               strides_[1] == strides_[2] * dims_[2] && strides_[0] == strides_[1] * dims_[1];
    }

    bool SameShape(const Tensor &other, bool check_type = true, bool check_stride = false) const {
        return Dims() == other.Dims() && (!check_stride || Strides() == other.Strides()) &&
               (!check_type || type_ == other.type_);
    }

   private:
    Tensor() = delete;
    Tensor(TensorContextT<Tensor> *ctx, const std::vector<int> &shape, TensorType type,
           std::byte *data)
        : ctx_(ctx),
          n_dims_(shape.size()),
          data_(data),
          type_(type),
          op_(kOpNone),
          grad_(nullptr),
          src0_(nullptr),
          src1_(nullptr) {
        assert(n_dims_ <= kMaxTensorDims);

        for (int i = 0; i < n_dims_; i++) {
            dims_[i + kMaxTensorDims - n_dims_] = shape[i];
        }
        for (int i = 0; i < kMaxTensorDims - n_dims_; i++) {
            dims_[i] = 1;
        }
        strides_[kMaxTensorDims - 1] = 1;
        for (int i = kMaxTensorDims - 2; i >= 0; i--) {
            strides_[i] = strides_[i + 1] * dims_[i + 1];
        }
    }

    static bool can_matmul(const Tensor &src0, const Tensor &src1) {
        static_assert(kMaxTensorDims == 4, "MAX_TENSOR_DIMS is not 4 - update this function");
        return src0.n_dims_ >= 2 && src0.n_dims_ == src1.n_dims_ &&
               src0.dims_[3] == src1.dims_[3] && src0.dims_[0] == src1.dims_[0] &&
               src0.dims_[1] == src1.dims_[1];
    }

    // start_dim_r is the starting dimension from the right
    static bool can_broadcast_to(const Tensor &from, const Tensor &to, int start_dim_r = 0) {
        const auto &shape = to.Dims();
        bool ok = shape.size() >= from.n_dims_ && shape.size() <= kMaxTensorDims;
        assert(from.n_dims_ >= start_dim_r);
        for (int i = start_dim_r; i < from.n_dims_; i++) {
            ok = ok && (from.dims_[kMaxTensorDims - i - 1] == shape[shape.size() - i - 1] ||
                        from.dims_[kMaxTensorDims - i - 1] == 1);
        }
        return ok;
    }

    // start_dim_r is the starting dimension from the right
    static Tensor &broadcast_to(TensorContext *ctx, Tensor &from, const Tensor &to,
                                int start_dim_r = 0) {
        // check that the shape is compatible with the current tensor
        assert(can_broadcast_to(from, to, start_dim_r));
        auto dshape = to.Dims();
        for (int i = 0; i < start_dim_r; i++) {
            dshape[dshape.size() - i - 1] = from.dims_[kMaxTensorDims - i - 1];
        }
        Tensor *dst = ctx->NewTensor(dshape, from.type_);
        dst->op_ = kOpBroadcast;
        dst->src0_ = &from;
        return *dst;
    }

    size_t n_vec() const {
        static_assert(kMaxTensorDims == 4, "MAX_TENSOR_DIMS is not 4 - update this function");
        return (size_t)dims_[0] * dims_[1] * dims_[2];
    }

    size_t vstride() const {
        static_assert(kMaxTensorDims == 4, "MAX_TENSOR_DIMS is not 4 - update this function");
        return strides_[2];
    }

    size_t vsize() const {
        static_assert(kMaxTensorDims == 4, "MAX_TENSOR_DIMS is not 4 - update this function");
        return (size_t)dims_[3];
    }

    size_t n_mat() const {
        static_assert(kMaxTensorDims == 4, "MAX_TENSOR_DIMS is not 4 - update this function");
        return (size_t)dims_[0] * dims_[1];
    }

    std::tuple<int, int> mat() const {
        static_assert(kMaxTensorDims == 4, "MAX_TENSOR_DIMS is not 4 - update this function");
        return {dims_[2], dims_[3]};
    }

    size_t mstride() const {
        static_assert(kMaxTensorDims == 4, "MAX_TENSOR_DIMS is not 4 - update this function");
        return strides_[1];
    }

    static size_t num_of_elements(const std::vector<int> &shape) {
        size_t e = 1;
        for (auto s : shape) {
            e *= s;
        }
        return e;
    }

    // Add
    // TODO(ysg): support strided add
    static void add_forward(Tensor *dst, Tensor *src0, Tensor *src1) {
        assert(dst->type_ == kF32);
        assert(src0->SameShape(*src1) && src1->SameShape(*dst));
        assert(src0->IsContiguous() && src1->IsContiguous() && dst->IsContiguous());

        size_t n = dst->n_vec();
        for (size_t i = 0; i < n; i++) {
            vec_add(dst->vsize(), (float *)dst->data_ + i * dst->vstride(),
                    (float *)src0->data_ + i * src0->vstride(),
                    (float *)src1->data_ + i * src1->vstride());
        }
    }

    static void add_backward(Tensor *dst, Tensor *src0, Tensor *src1) {
        if (src0->grad_ == nullptr) {
            src0->AllocGrad(false)->grad()->CopyDataFrom(*dst->grad_);
        } else {
            add_forward(src0->grad_, src0->grad_, dst->grad_);
        }
        if (src1->grad_ == nullptr) {
            src1->AllocGrad(false)->grad()->CopyDataFrom(*dst->grad_);
        } else {
            add_forward(src1->grad_, src1->grad_, dst->grad_);
        }
    }

    // Mul
    // TODO(ysg): support strided mul
    static void mul_forward(Tensor *dst, Tensor *src0, Tensor *src1, bool is_acc = false) {
        assert(dst->type_ == kF32);
        assert(src0->SameShape(*src1) && src1->SameShape(*dst));
        assert(src0->IsContiguous() && src1->IsContiguous() && dst->IsContiguous());

        size_t n = dst->n_vec(), m = dst->vsize();
        if (!is_acc) {
            for (size_t i = 0; i < n; i++) {
                float *out = (float *)dst->data_ + i * dst->vstride();
                float *in0 = (float *)src0->data_ + i * src0->vstride();
                float *in1 = (float *)src1->data_ + i * src1->vstride();
                for (size_t j = 0; j < m; j++) {
                    out[j] = in0[j] * in1[j];
                }
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                float *out = (float *)dst->data_ + i * dst->vstride();
                float *in0 = (float *)src0->data_ + i * src0->vstride();
                float *in1 = (float *)src1->data_ + i * src1->vstride();
                for (size_t j = 0; j < m; j++) {
                    out[j] += in0[j] * in1[j];
                }
            }
        }
    }

    static void mul_backward(Tensor *dst, Tensor *src0, Tensor *src1) {
        src0->AllocGrad();
        mul_forward(src0->grad_, dst->grad_, src1, true);

        src1->AllocGrad();
        mul_forward(src1->grad_, dst->grad_, src0, true);
    }

    // Matmul
    // TODO(ysg): support strided matmul
    static void matmul_forward(Tensor *dst, Tensor *src0, Tensor *src1) {
        assert(src0->n_mat() == dst->n_mat() && src1->n_mat() == dst->n_mat());
        assert(dst->type_ == kF32 && src0->type_ == dst->type_ && src1->type_ == dst->type_);
        assert(src0->IsContiguous() && src1->IsContiguous() && dst->IsContiguous());

        size_t n = dst->dims_[2], m = dst->dims_[3], p = src0->dims_[3];
        #pragma omp parallel for collapse(2)
        for (size_t mati = 0; mati < dst->n_mat(); mati++) {
            for (size_t i = 0; i < n; i++) {
                float *out = (float *)dst->data_ + mati * dst->mstride() + i * dst->strides_[2];
                float *in0 = (float *)src0->data_ + mati * src0->mstride() + i * src0->strides_[2];
                for (size_t j = 0; j < m; j++) {
                    float *in1 =
                        (float *)src1->data_ + mati * src1->mstride() + j * src1->strides_[2];
                    out[j] = vec_dot_f32(p, in0, in1);
                }
            }
        }
    }

    static void matmul_backward(Tensor *dst, Tensor *src0, Tensor *src1) {
        src0->AllocGrad();
        src1->AllocGrad();

        size_t matn = dst->n_mat();
        float *dout = (float *)dst->grad_->data_;
        float *din0 = (float *)src0->grad_->data_, *in0 = (float *)src0->data_;
        float *din1 = (float *)src1->grad_->data_, *in1 = (float *)src1->data_;

        // src0->grad += dst->grad matmul src1^T
        size_t n = src0->dims_[2], m = src0->dims_[3], p = dst->dims_[3];
        #pragma omp parallel for collapse(2)
        for (size_t mati = 0; mati < matn; mati++) {
            float *in1_ma = in1 + mati * src1->mstride();
            for (size_t i = 0; i < n; i++) {
                float *din0_mai = din0 + mati * src0->mstride() + i * src0->strides_[2];
                float *dout_mai = dout + mati * dst->mstride() + i * dst->strides_[2];
                for (size_t k = 0; k < p; k++) {
                    for (size_t j = 0; j < m; j++) {
                        din0_mai[j] += dout_mai[k] * in1_ma[k * src1->strides_[2] + j];
                    }
                }
            }
        }

        // src1->grad += dst->grad^T matmul src0^T
        n = src1->dims_[2], m = src1->dims_[3], p = dst->dims_[2];
        #pragma omp parallel for
        for (size_t mati = 0; mati < matn; mati++) {
            float *dout_ma = dout + mati * dst->mstride();
            float *in0_ma = in0 + mati * src0->mstride();
            for (size_t k = 0; k < p; k++) {
                for (size_t i = 0; i < n; i++) {
                    float *din1_mai = din1 + mati * src1->mstride() + i * src1->strides_[2];
                    for (size_t j = 0; j < m; j++) {
                        din1_mai[j] +=
                            dout_ma[k * dst->strides_[2] + i] * in0_ma[k * src0->strides_[2] + j];
                    }
                }
            }
        }
    }

    inline static float vec_dot_f32(const size_t n, const float *va, const float *vb) {
        float sum = 0.0f;

// TODO(ysg): resolve this
#ifdef F32_NEON_IS_SLOWER  // __ARM_NEON
        const size_t n4 = n / 4 * 4;
        float32x4_t sum4 = vdupq_n_f32(0.0f);
        for (size_t i = 0; i < n4; i += 4) {
            float32x4_t va4 = vld1q_f32(va + i);
            float32x4_t vb4 = vld1q_f32(vb + i);
            sum4 = vmlaq_f32(sum4, va4, vb4);
        }
        sum = sum4[0] + sum4[1] + sum4[2] + sum4[3];
#else
        const size_t n4 = 0;
#endif

        for (size_t i = n4; i < n; i++) {
            sum += va[i] * vb[i];
        }
        return sum;
    }

    // Lookup
    static void lookup_forward(Tensor *dst, Tensor *src0, Tensor *src1) {
        assert(dst->type_ == src0->type_ && src1->type_ == kI32);
        assert(src0->IsContiguous() && src1->IsContiguous() && dst->IsContiguous());

        size_t i0_size = src0->dims_[kMaxTensorDims - src0->n_dims_];
        size_t i0_stride = src0->strides_[kMaxTensorDims - src0->n_dims_];
        size_t type_size = kTypeSize[src0->type_];

        for (size_t i = 0; i < src1->NumElements(); i++) {
            int32_t idx = ((int32_t *)src1->data_)[i];
            assert(idx >= 0 && idx < i0_size);
            memcpy(dst->data_ + i * i0_stride * type_size,
                   src0->data_ + idx * i0_stride * type_size, i0_stride * type_size);
        }
    }

    static void lookup_backward(Tensor *dst, Tensor *src0, Tensor *src1) {
        src0->AllocGrad();

        size_t i0_stride = src0->strides_[kMaxTensorDims - src0->n_dims_];
        size_t type_size = kTypeSize[src0->type_];

        for (size_t i = 0; i < src1->NumElements(); i++) {
            int32_t idx = ((int32_t *)src1->data_)[i];
            vec_add(i0_stride, (float *)src0->grad_->data_ + idx * i0_stride,
                    (float *)src0->grad_->data_ + idx * i0_stride,
                    (float *)dst->grad_->data_ + i * i0_stride);
        }
    }

    // Norm
    static void norm_forward(Tensor *dst, Tensor *src) {
        assert(src->type_ == kF32 && dst->type_ == src->type_);
        assert(src->IsContiguous() && dst->IsContiguous());

        for (size_t idx = 0; idx < src->n_vec(); idx++) {
            const float *vec = (float *)src->data_ + idx * src->vstride();
            size_t vec_size = src->vsize();

            // calculate the mean and the rstd (without bias correction)
            float mean = vec_mean(vec_size, vec);
            float rstd = vec_rstd(vec_size, vec, mean);

            float *out = (float *)dst->data_ + idx * dst->vstride();
            for (size_t i = 0; i < vec_size; i++) {
                out[i] = (vec[i] - mean) * rstd;
            }
        }
    }

    static void norm_backward(Tensor *dst, Tensor *src) {
        src->AllocGrad();

        for (size_t idx = 0; idx < src->n_vec(); idx++) {
            const float *a = (float *)src->data_ + idx * src->vstride();
            const float *b = (float *)dst->data_ + idx * dst->vstride();
            size_t vec_size = src->vsize();
            assert(vec_size > 0);

            float mean = vec_mean(vec_size, a);
            float rstd = vec_rstd(vec_size, a, mean);

            float *sgrad = (float *)src->grad_->data_ + idx * src->vstride();
            float *dgrad = (float *)dst->grad_->data_ + idx * dst->vstride();

            float dgrad_mean = 0.0f, dgrad2_mean = 0.0f;
            for (size_t i = 0; i < vec_size; i++) {
                dgrad_mean += dgrad[i];
                dgrad2_mean += dgrad[i] * b[i];
            }
            dgrad_mean /= vec_size;
            dgrad2_mean /= vec_size;

            for (size_t i = 0; i < vec_size; i++) {
                sgrad[i] += ((dgrad[i] - dgrad_mean) - dgrad2_mean * b[i]) * rstd;
            }
        }
    }

    static float vec_mean(size_t vec_size, const float *src) {
        float sum = 0.0f;
        for (size_t i = 0; i < vec_size; i++) {
            sum += src[i];
        }
        return sum / vec_size;
    }

    static float vec_rstd(size_t vec_size, const float *src, float mean) {
        float eps = 1e-5f;
        float sum = 0.0f;
        for (size_t i = 0; i < vec_size; i++) {
            float diff = src[i] - mean;
            sum += diff * diff;
        }
        float var = sum / vec_size;
        return 1.0f / sqrtf(var + eps);
    }

    // View
    static void view_forward(Tensor *dst, Tensor *src, int split_no, int split_axis) {
        assert(dst->type_ == kF32);
        int dimi, offset;
        int sdims[kMaxTensorDims];
        calculate_split(sdims, dimi, offset, dst, src, split_no, split_axis);

        size_t d0 = dst->dims_[0], d1 = dst->dims_[1], d2 = dst->dims_[2], d3 = dst->dims_[3];
        for (size_t i0 = 0; i0 < d0; i0++) {
            for (size_t i1 = 0; i1 < d1; i1++) {
                for (size_t i2 = 0; i2 < d2; i2++) {
                    for (size_t i3 = 0; i3 < d3; i3++) {
                        size_t idx = i0 * dst->strides_[0] + i1 * dst->strides_[1] +
                                     i2 * dst->strides_[2] + i3 * dst->strides_[3];
                        float *dd = (float *)dst->data_ + idx;
                        size_t sidx[4] = {idx / sdims[1] / sdims[2] / sdims[3],
                                          idx / sdims[2] / sdims[3] % sdims[1],
                                          idx / sdims[3] % sdims[2], idx % sdims[3]};
                        sidx[dimi] += offset;
                        float *sd = (float *)src->data_ + sidx[0] * src->strides_[0] +
                                    sidx[1] * src->strides_[1] + sidx[2] * src->strides_[2] +
                                    sidx[3] * src->strides_[3];
                        *dd = *sd;
                    }
                }
            }
        }
    }

    static void view_backward(Tensor *dst, Tensor *src, int split_no, int split_axis) {
        src->AllocGrad();

        int dimi, offset;
        int sdims[kMaxTensorDims];
        calculate_split(sdims, dimi, offset, dst, src, split_no, split_axis);

        size_t d0 = dst->dims_[0], d1 = dst->dims_[1], d2 = dst->dims_[2], d3 = dst->dims_[3];
        for (size_t i0 = 0; i0 < d0; i0++) {
            for (size_t i1 = 0; i1 < d1; i1++) {
                for (size_t i2 = 0; i2 < d2; i2++) {
                    for (size_t i3 = 0; i3 < d3; i3++) {
                        size_t idx = i0 * dst->strides_[0] + i1 * dst->strides_[1] +
                                     i2 * dst->strides_[2] + i3 * dst->strides_[3];
                        float *dd = (float *)dst->grad_->data_ + idx;
                        size_t sidx[4] = {idx / sdims[1] / sdims[2] / sdims[3],
                                          idx / sdims[2] / sdims[3] % sdims[1],
                                          idx / sdims[3] % sdims[2], idx % sdims[3]};
                        sidx[dimi] += offset;
                        float *sd = (float *)src->grad_->data_ + sidx[0] * src->strides_[0] +
                                    sidx[1] * src->strides_[1] + sidx[2] * src->strides_[2] +
                                    sidx[3] * src->strides_[3];
                        *sd += *dd;
                    }
                }
            }
        }
    }

    static void calculate_split(int *dims, int &dimi, int &offset, Tensor *dst, Tensor *src,
                                int split_no, int split_axis) {
        dimi = kMaxTensorDims - src->n_dims_ + split_axis;
        int split_size = src->dims_[dimi] / (src->NumElements() / dst->NumElements());
        offset = split_no * split_size;

        for (int i = 0; i < kMaxTensorDims; i++) {
            dims[i] = src->dims_[i];
        }
        dims[dimi] = split_size;
    }

    // Tranpose
    static void transpose_forward(Tensor *dst, Tensor *src, int dimi0, int dimi1) {
        assert(dst->type_ == kF32);

        transpose_impl((float *)dst->data_, dst->strides_, (float *)src->data_, src->strides_,
                       src->dims_, dimi0, dimi1, false);
    }

    static void transpose_backward(Tensor *dst, Tensor *src, int dimi0, int dimi1) {
        src->AllocGrad();

        transpose_impl((float *)src->grad_->data_, src->strides_, (float *)dst->grad_->data_,
                       dst->strides_, dst->dims_, dimi0, dimi1, true);
    }

    static void transpose_impl(float *out, size_t *out_strides, float *in, size_t *in_strides,
                               int *dims, int dimi0, int dimi1, bool is_acc) {
        size_t d0 = dims[0], d1 = dims[1], d2 = dims[2], d3 = dims[3];
        for (size_t i0 = 0; i0 < d0; i0++) {
            for (size_t i1 = 0; i1 < d1; i1++) {
                for (size_t i2 = 0; i2 < d2; i2++) {
                    for (size_t i3 = 0; i3 < d3; i3++) {
                        float *sd = in + (i0 * in_strides[0] + i1 * in_strides[1] +
                                          i2 * in_strides[2] + i3 * in_strides[3]);
                        size_t di[4] = {i0, i1, i2, i3};
                        std::swap(di[dimi0], di[dimi1]);
                        float *dd = out + (di[0] * out_strides[0] + di[1] * out_strides[1] +
                                           di[2] * out_strides[2] + di[3] * out_strides[3]);

                        if (is_acc) {
                            *dd += *sd;
                        } else {
                            *dd = *sd;
                        }
                    }
                }
            }
        }
    }

    // Broadcast
    static void broadcast_forward(Tensor *dst, Tensor *src0) {
        assert(dst->type_ == kF32);
        size_t d0 = dst->dims_[0], d1 = dst->dims_[1], d2 = dst->dims_[2], d3 = dst->dims_[3];

        for (size_t i0 = 0; i0 < d0; i0++) {
            for (size_t i1 = 0; i1 < d1; i1++) {
                for (size_t i2 = 0; i2 < d2; i2++) {
                    for (size_t i3 = 0; i3 < d3; i3++) {
                        float *dd =
                            (float *)dst->data_ + (i0 * dst->strides_[0] + i1 * dst->strides_[1] +
                                                   i2 * dst->strides_[2] + i3 * dst->strides_[3]);
                        size_t si0 = i0 % src0->dims_[0], si1 = i1 % src0->dims_[1],
                               si2 = i2 % src0->dims_[2], si3 = i3 % src0->dims_[3];
                        float *sd = (float *)src0->data_ +
                                    (si0 * src0->strides_[0] + si1 * src0->strides_[1] +
                                     si2 * src0->strides_[2] + si3 * src0->strides_[3]);
                        *dd = *sd;
                    }
                }
            }
        }
    }

    static void broadcast_backward(Tensor *dst, Tensor *src0) {
        src0->AllocGrad();

        size_t d0 = dst->grad_->dims_[0], d1 = dst->grad_->dims_[1], d2 = dst->grad_->dims_[2],
               d3 = dst->grad_->dims_[3];

        for (size_t i0 = 0; i0 < d0; i0++) {
            for (size_t i1 = 0; i1 < d1; i1++) {
                for (size_t i2 = 0; i2 < d2; i2++) {
                    for (size_t i3 = 0; i3 < d3; i3++) {
                        auto dd = (float *)dst->grad_->data_ +
                                  (i0 * dst->grad_->strides_[0] + i1 * dst->grad_->strides_[1] +
                                   i2 * dst->grad_->strides_[2] + i3 * dst->grad_->strides_[3]);
                        size_t si0 = i0 % src0->grad_->dims_[0], si1 = i1 % src0->grad_->dims_[1],
                               si2 = i2 % src0->grad_->dims_[2], si3 = i3 % src0->grad_->dims_[3];
                        auto sd = (float *)src0->grad_->data_ +
                                  (si0 * src0->grad_->strides_[0] + si1 * src0->grad_->strides_[1] +
                                   si2 * src0->grad_->strides_[2] + si3 * src0->grad_->strides_[3]);
                        *sd += *dd;
                    }
                }
            }
        }
    }

    // Gelu
    // compute_gelu and backward_gelu are copid from karpathy/llm.c
    static void gelu_forward(Tensor *dst, Tensor *src) {
        assert(dst->SameShape(*src, true, true));

        auto out = (float *)dst->data_;
        auto inp = (float *)src->data_;
        auto N = dst->NumElements();
        float s = sqrtf(2.0f / M_PI);
        for (int i = 0; i < N; i++) {
            float x = inp[i];
            float cube = 0.044715f * x * x * x;
            out[i] = 0.5f * x * (1.0f + tanhf(s * (x + cube)));
        }
    }

    static void gelu_backward(Tensor *dst, Tensor *src) {
        src->AllocGrad();

        auto N = dst->NumElements();
        auto dinp = (float *)src->grad_->data_;
        auto inp = (float *)src->data_;
        auto dout = (float *)dst->grad_->data_;

        float s = sqrtf(2.0f / M_PI);
        for (int i = 0; i < N; i++) {
            float x = inp[i];
            float cube = 0.044715f * x * x * x;
            float tanh_arg = s * (x + cube);
            float tanh_out = tanhf(tanh_arg);
            float coshf_out = coshf(tanh_arg);
            float sech_out = 1.0f / (coshf_out * coshf_out);
            float local_grad = 0.5f * (1.0f + tanh_out) +
                               x * 0.5f * sech_out * s * (1.0f + 3.0f * 0.044715f * x * x);
            dinp[i] += local_grad * dout[i];
        }
    }

    // Softmax
    static void softmax_forward(Tensor *dst, Tensor *src, bool is_casual, int vocab_size) {
        assert(dst->SameShape(*src));
        assert(dst->IsContiguous() && src->IsContiguous());

        auto [n, m] = dst->mat();
        assert(m > 0);

        for (size_t mati = 0; mati < dst->n_mat(); mati++) {
            for (size_t i = 0; i < n; i++) {
                auto logits = (float *)src->data_ + mati * src->mstride() + i * m;
                auto probs = (float *)dst->data_ + mati * dst->mstride() + i * m;
                int V = vocab_size > 0 ? vocab_size : m;
                size_t end = is_casual ? i + 1 : V;

                float maxv = -10000.0f;
                for (size_t j = 0; j < end; j++) {
                    maxv = fmaxf(maxv, logits[j]);
                }

                float sum = 0.0f;
                for (size_t j = 0; j < end; j++) {
                    probs[j] = expf(logits[j] - maxv);
                    sum += probs[j];
                }

                for (size_t j = 0; j < end; j++) {
                    probs[j] = probs[j] * (1.0f / sum);
                }

                // [end, V) is padded with 0.0f due to the causal mask
                // [V, m) is padded with 0.0f due to the padded vocab
                for (size_t j = end; j < m; j++) {
                    probs[j] = 0.0f;
                }
            }
        }
    }

    static void softmax_backward(Tensor *dst, Tensor *src, bool is_casual, int vocab_size) {
        src->AllocGrad();

        assert(dst->SameShape(*src));
        auto [n, m] = dst->mat();
        assert(m > 0);

        for (size_t mati = 0; mati < dst->n_mat(); mati++) {
            for (size_t i = 0; i < n; i++) {
                auto dout = (float *)dst->grad_->data_ + mati * dst->mstride() + i * m;
                auto out = (float *)dst->data_ + mati * dst->mstride() + i * m;
                auto din = (float *)src->grad_->data_ + mati * src->mstride() + i * m;
                int V = vocab_size > 0 ? vocab_size : m;
                auto end = is_casual ? i + 1 : V;

                float dsum = 0.0f;
                for (int j = 0; j < end; j++) {
                    dsum += dout[j] * out[j];
                }

                for (int j = 0; j < end; j++) {
                    din[j] += out[j] * (dout[j] - dsum);
                }
            }
        }
    }

    // CrossEntropy
    static void cross_entropy_forward(Tensor *dst, Tensor *src, Tensor *src1) {
        assert(dst->IsContiguous() && src->IsContiguous());
        auto losses = (float *)dst->data_;
        auto targets = (int32_t *)src1->data_;
        auto vs = src->vsize();

        for (size_t vi = 0; vi < src->n_vec(); vi++) {
            auto probs = (float *)src->data_ + vi * vs;
            auto ix = targets[vi];
            losses[vi] = -logf(probs[ix]);
        }
    }

    static void cross_entropy_backward(Tensor *dst, Tensor *src, Tensor *src1) {
        src->AllocGrad();

        auto targets = (int32_t *)src1->data_;
        auto vs = src->vsize();

        for (size_t vi = 0; vi < src->n_vec(); vi++) {
            auto probs = (float *)src->data_ + vi * vs;
            auto loss = ((float *)dst->grad_->data_)[vi];
            auto din = (float *)src->grad_->data_ + vi * vs;
            auto ix = targets[vi];
            din[ix] += -1.0f / probs[ix] * loss;
        }
    }

    static void vec_add(size_t vec_size, float *out, float *src0, float *src1) {
        for (size_t i = 0; i < vec_size; i++) {
            out[i] = src0[i] + src1[i];
        }
    }

    template <typename T>
    static void vec_fill(size_t vec_size, T *out, const T *data) {
        for (size_t i = 0; i < vec_size; i++) {
            out[i] = data[i];
        }
    }
    template <typename T>
    static void vec_fill(size_t vec_size, T *out, T val) {
        for (size_t i = 0; i < vec_size; i++) {
            out[i] = val;
        }
    }

    static void vec_random_norm(size_t vec_size, float *out) {
        static NormalDist NORMAL_DIST;
        for (size_t i = 0; i < vec_size; i++) {
            out[i] = NORMAL_DIST();
        }
    }

    static void vec_print(size_t vec_size, TensorType type, std::byte *vec) {
        std::cout << "[";
        if (type == kF32) {
            float *vec_float = reinterpret_cast<float *>(vec);
            for (size_t i = 0; i < vec_size; i++) {
                std::cout << vec_float[i] << ",]"[i == vec_size - 1];
            }
        } else if (type == kI32) {
            int32_t *vec_int32 = reinterpret_cast<int32_t *>(vec);
            for (size_t i = 0; i < vec_size; i++) {
                std::cout << vec_int32[i] << ",]"[i == vec_size - 1];
            }
        } else {
            throw std::runtime_error("vec_print(): Not implemented");
        }
    }

    static std::vector<Tensor *> topo_sort(Tensor *tensor) {
        std::vector<Tensor *> sorted;

        std::unordered_map<Tensor *, int> visited;

        std::function<void(Tensor *)> dfs = [&](Tensor *tensor) {
            auto it = visited.find(tensor);
            if (it != visited.end()) {
                if (it->second == 1) {
                    throw std::runtime_error("topo_sort(): Cycle detected");
                }
                return;
            }
            visited[tensor] = 1;
            if (tensor->src0_ != nullptr) {
                dfs(tensor->src0_);
            }
            if (tensor->src1_ != nullptr) {
                dfs(tensor->src1_);
            }
            sorted.push_back(tensor);
            visited[tensor] = 2;
        };

        dfs(tensor);

        return sorted;
    }

   public:
    static Profile ForwardProfile;   // NOLINT
    static Profile BackwardProfile;  // NOLINT

   private:
    int n_dims_;
    int dims_[kMaxTensorDims];
    size_t strides_[kMaxTensorDims];

    TensorType type_;
    TensorOp op_;
    int op_params_[kMaxTensorOpParams];

    Tensor *grad_;
    Tensor *src0_;
    Tensor *src1_;

    std::byte *data_;

    TensorContext *ctx_;

    friend class TensorContextT<Tensor>;
};

static_assert(sizeof(Object) % kTensorMemAlign == 0,
              "Object size must be a multiple of TENSOR_MEM_ALIGN");
static_assert(sizeof(Tensor) % kTensorMemAlign == 0,
              "Tensor size must be a multiple of TENSOR_MEM_ALIGN");

#ifndef TENSOR_PROFILE_DEFINE
#define TENSOR_PROFILE_DEFINE
Profile Tensor::ForwardProfile;   // NOLINT
Profile Tensor::BackwardProfile;  // NOLINT
#endif

}  // namespace tinytorch
