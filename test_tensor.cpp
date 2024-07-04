#include <iostream>

#include "boost/ut.hpp"
#include "tinytorch.hpp"

using namespace tinytorch;  // NOLINT
using namespace boost;      // NOLINT

namespace std {

template <typename T>
ostream &operator<<(ostream &os, const vector<T> &vec) {
    os << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        os << vec[i];
        if (i < vec.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

};  // namespace std

#define TO_VEC(a) std::vector<float>(a, a + sizeof(a) / sizeof(float))

const float eps = 1e-5; // NOLINT

bool CheckVectorEqual(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); i++) {
        if (std::isnan(a[i]) || std::isnan(b[i]) || fabs(a[i] - b[i]) > eps) {
            return false;
        }
    }
    return true;
}

void TestTensor(TensorContext &ctx) {
    ut::test("TestTensorAdd") = [&ctx] {
        std::vector<int> a_dims = {2, 3, 2};
        float a_data[] = {-0.156085f, 0.119778f,  0.761198f,  0.347593f, -0.474833f, -0.314762f,
                          -0.510900f, -0.273722f, -0.890527f, 0.362633f, -0.958626f, 0.358971f};
        std::vector<int> b_dims = {2, 3, 2};
        float b_data[] = {0.642503f,  0.716670f,  -0.840944f, -0.951772f, -0.858585f, 0.895501f,
                          -0.205840f, -0.693812f, -0.889058f, 0.434394f,  0.890442f,  0.936250f};
        std::vector<int> result_dims = {2, 3, 2};
        float result_data[] = {0.486419f,  0.836448f,  -0.079746f, -0.604179f, -1.333419f, 0.580739f,
                               -0.716740f, -0.967534f, -1.779585f, 0.797027f,  -0.068184f, 1.295222f};

        auto &a = *ctx.NewTensor(a_dims)->Fill(a_data);
        auto &b = *ctx.NewTensor(b_dims)->Fill(b_data);
        auto &c = a + b;
        c.Forward();
        ut::expect(CheckVectorEqual(c.Flatten(), TO_VEC(result_data)))
            << c.Flatten() << " != " << TO_VEC(result_data) << "\n";
    };

    ut::test("TestTensorMul") = [&ctx] {
        std::vector<int> a_dims = {2, 3, 2};
        float a_data[] = {0.321711f,  0.549407f, -0.011015f, -0.610029f, -0.721674f, -0.857888f,
                          -0.992038f, 0.380269f, 0.134258f,  -0.639243f, -0.338716f, 0.289208f};
        float a_grad_data[] = {0.110795f, -0.960914f, 0.314134f,  -0.145329f, -0.071354f, -0.052769f,
                               0.418165f, -0.533604f, -0.073091f, 0.607927f,  0.679904f,  0.377298f};
        std::vector<int> b_dims = {2, 3, 2};
        float b_data[] = {0.110795f, -0.960914f, 0.314134f,  -0.145329f, -0.071354f, -0.052769f,
                          0.418165f, -0.533604f, -0.073091f, 0.607927f,  0.679904f,  0.377298f};
        float b_grad_data[] = {0.321711f,  0.549407f, -0.011015f, -0.610029f, -0.721674f, -0.857888f,
                               -0.992038f, 0.380269f, 0.134258f,  -0.639243f, -0.338716f, 0.289208f};
        std::vector<int> result_dims = {2, 3, 2};
        float result_data[] = {0.035644f,  -0.527933f, -0.003460f, 0.088655f,  0.051494f,  0.045270f,
                               -0.414835f, -0.202913f, -0.009813f, -0.388613f, -0.230294f, 0.109118f};
        auto &a = *ctx.NewTensor(a_dims)->Fill(a_data);
        auto &b = *ctx.NewTensor(b_dims)->Fill(b_data);
        auto b_grad = b.RandomGrad()->grad();
        auto &b_grad_truth = *ctx.NewTensor(b_dims)->Fill(b_grad_data) + *b_grad;
        b_grad_truth.Forward();
        auto &c = a * b;
        c.Forward();
        c.Backward();

        ut::expect(CheckVectorEqual(c.Flatten(), TO_VEC(result_data)))
            << c.Flatten() << " != " << TO_VEC(result_data) << "\n";
        ut::expect(CheckVectorEqual(a.grad()->Flatten(), TO_VEC(a_grad_data)))
            << "a grad: " << a.grad()->Flatten() << " != " << TO_VEC(a_grad_data) << "\n";
        ut::expect(CheckVectorEqual(b.grad()->Flatten(), b_grad_truth.Flatten()))
            << "b grad: " << b.grad()->Flatten() << " != " << b_grad_truth.Flatten() << "\n";
    };

    ut::test("TestTensorMulMat") = [&ctx] {
        std::vector<int> a_dims = {2, 3, 2};
        float a_data[] = {0.038856f, -0.205084f, -0.576147f, -0.253080f, -0.973295f, 0.236236f,
                          0.515430f, 0.641762f,  -0.492475f, -0.325701f, -0.156254f, 0.507203f};
        std::vector<int> b_dims = {2, 4, 2};
        float b_data[] = {-0.926039f, 0.038612f,  0.818189f,  0.535480f, 0.217580f,  0.744733f,  0.888028f, -0.513070f,
                          -0.341174f, -0.588775f, -0.814142f, 0.241264f, -0.224577f, -0.357855f, 0.081625f, -0.727418f};
        std::vector<int> result_dims = {2, 3, 4};
        float result_data[] = {-0.043901f, -0.078027f, -0.144279f, 0.139727f,  0.523762f,  -0.606916f,
                               -0.313835f, -0.381786f, 0.910431f,  -0.669840f, -0.035837f, -0.985519f,
                               -0.553705f, -0.264799f, -0.345412f, -0.424757f, 0.359784f,  0.322365f,
                               0.227153f,  0.196722f,  -0.245319f, 0.249583f,  -0.146414f, -0.381703f};
        auto &a = *ctx.NewTensor(a_dims)->Fill(a_data);
        auto &b = *ctx.NewTensor(b_dims)->Fill(b_data);
        auto &c = a.MatMul(b);
        c.Forward();
        ut::expect(CheckVectorEqual(c.Flatten(), TO_VEC(result_data)))
            << c.Flatten() << " != " << TO_VEC(result_data) << "\n";
    };

    ut::test("TestTensorLinear") = [&ctx] {
        std::vector<int> a_dims = {2, 3, 2};
        float a_data[] = {-0.124705f, 0.461944f, -0.035526f, 0.603655f,  -0.565689f, 0.580007f,
                          0.739768f,  0.555825f, 0.353833f,  -0.393867f, -0.692182f, 0.200951f};
        float a_grad_data[] = {-0.334757f, -0.358625f, -0.334757f, -0.358625f, -0.334757f, -0.358625f,
                               0.461304f,  -0.141092f, 0.461304f,  -0.141092f, 0.461304f,  -0.141092f};
        std::vector<int> b_dims = {2, 4, 2};
        float b_data[] = {-0.337031f, -0.849364f, 0.880950f,  0.673869f, -0.806584f, 0.693106f,
                          -0.072092f, -0.876236f, -0.607369f, 0.573827f, -0.175670f, -0.252041f,
                          0.433276f,  -0.186126f, 0.811068f,  -0.276751f};
        float b_grad_data[] = {-0.725921f, 1.645606f, -0.725921f, 1.645606f, -0.725921f, 1.645606f,
                               -0.725921f, 1.645606f, 0.401419f,  0.362910f, 0.401419f,  0.362910f,
                               0.401419f,  0.362910f, 0.401419f,  0.362910f};
        std::vector<int> c_dims = {2, 3, 4};
        float c_data[] = {0.405725f,  -0.488369f, 0.916681f,  -0.544182f, 0.213521f,  -0.491035f, 0.491596f, -0.358687f,
                          0.530066f,  -0.208435f, -0.675905f, 0.453980f,  -0.926804f, -0.315975f, 0.534358f, -0.848598f,
                          -0.431841f, -0.352661f, 0.723182f,  0.930128f,  -0.483508f, -0.560935f, 0.256884f, 0.824113f};
        float c_grad_data[] = {1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
                               1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
                               1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f};
        std::vector<int> result_dims = {2, 3, 4};
        float result_data[] = {0.055396f,  -0.286938f, 1.337443f, -0.939964f, -0.287228f, -0.115548f,
                               0.938648f,  -0.885070f, 0.228083f, -0.315930f, 0.182377f,  -0.013461f,
                               -1.057168f, -0.586021f, 0.751428f, -0.402421f, -0.872759f, -0.315548f,
                               0.949798f,  1.326114f,  0.052213f, -0.489987f, -0.080424f, 0.207093f};
        float result_grad_data[] = {1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
                                    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
                                    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
                                    1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f};

        auto &a = *ctx.NewTensor(a_dims)->Fill(a_data);
        auto &b = *ctx.NewTensor(b_dims)->Fill(b_data);
        auto &c = *ctx.NewTensor(c_dims)->Fill(c_data);
        auto &d = a.MatMul(b) + c;
        d.Forward();
        d.ZeroGrad();
        d.Backward();

        // std::cout << "Tensor a: " << std::endl;
        // a.PrintTensor();
        // std::cout << "Tensor b: " << std::endl;
        // b.PrintTensor();
        // std::cout << "Tensor c: " << std::endl;
        // c.PrintTensor();
        // std::cout << "Tensor d: " << std::endl;
        // d.PrintTensor();

        ut::expect(CheckVectorEqual(d.Flatten(), TO_VEC(result_data)))
            << d.Flatten() << " != " << TO_VEC(result_data) << "\n";
        ut::expect(CheckVectorEqual(d.grad()->Flatten(), TO_VEC(result_grad_data)))
            << "result grad: " << d.grad()->Flatten() << " != " << TO_VEC(result_grad_data) << "\n";
        ut::expect(CheckVectorEqual(a.grad()->Flatten(), TO_VEC(a_grad_data)))
            << "a grad: " << a.grad()->Flatten() << " != " << TO_VEC(a_grad_data) << "\n";
        ut::expect(CheckVectorEqual(b.grad()->Flatten(), TO_VEC(b_grad_data)))
            << "b grad: " << b.grad()->Flatten() << " != " << TO_VEC(b_grad_data) << "\n";
        ut::expect(CheckVectorEqual(c.grad()->Flatten(), TO_VEC(c_grad_data)))
            << "c grad: " << c.grad()->Flatten() << " != " << TO_VEC(c_grad_data) << "\n";
    };

    ut::test("TestTensorLookup") = [&ctx] {
        int32_t a_data[] = {2, 1, 2, 0, 3, 4};
        auto a = ctx.NewTensor({2, 3}, kI32)->Fill(a_data);
        float b_data[] = {0.642503f,  0.716670f,  -0.840944f, -0.951772f, -0.858585f, 0.895501f,
                          -0.205840f, -0.693812f, -0.889058f, 0.434394f,  0.890442f,  0.936250f};
        float b_grad_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f};
        auto b = ctx.NewTensor({6, 2})->Fill(b_data);

        float result_data[] = {-0.858585f, 0.895501f, -0.840944f, -0.951772f, -0.858585f, 0.895501f,
                               0.642503f,  0.716670f, -0.205840f, -0.693812f, -0.889058f, 0.434394f};

        auto &c = (*b)[*a];

        c.Forward();
        c.ZeroGrad();
        c.Backward();

        ut::expect(CheckVectorEqual(c.Flatten(), TO_VEC(result_data)))
            << c.Flatten() << " != " << TO_VEC(result_data) << "\n";
        ut::expect(CheckVectorEqual(b->grad()->Flatten(), TO_VEC(b_grad_data)))
            << "gradients: " << b->grad()->Flatten() << " != " << TO_VEC(b_grad_data) << "\n";
    };

    ut::test("TestTensorNorm") = [&ctx] {
        std::vector<int> a_dims = {2, 2, 3};
        float a_data[] = {0.7323f, -0.2345f, 0.4123f, 0.1234f, 0.2234f, -0.3371f,
                          0.7323f, -0.2345f, 0.4123f, 0.1234f, 0.2234f, -0.3371f};
        float a_grad_data[] = {-0.268548f, -0.132720f, 0.401268f, -2.745161f, 2.255450f, 0.489711f,
                               -0.268548f, -0.132720f, 0.401268f, -2.745161f, 2.255450f, 0.489711f};

        float norm_data[] = {1.066593f, -1.337468f, 0.270875f, 0.492263f, 0.901913f, -1.394176f,
                             1.066593f, -1.337468f, 0.270875f, 0.492263f, 0.901913f, -1.394176f};

        std::vector<int> c_dims = {2, 3, 3};
        float c_data[] = {0.134287f,  0.829944f,  -0.215419f, -0.760618f, 0.586073f,  0.464571f,
                          0.665903f,  -0.128526f, 0.454756f,  0.134287f,  0.829944f,  -0.215419f,
                          -0.760618f, 0.586073f,  0.464571f,  0.665903f,  -0.128526f, 0.454756f};

        auto a = ctx.NewTensor(a_dims)->Fill(a_data);
        auto &b = a->Norm();
        auto c = ctx.NewTensor(c_dims)->Fill(c_data);
        auto &d = b.MatMul(*c);

        d.Forward();
        d.ZeroGrad();
        d.Backward();

        ut::expect(CheckVectorEqual(b.Flatten(), TO_VEC(norm_data)))
            << b.Flatten() << " != " << TO_VEC(norm_data) << "\n";
        ut::expect(CheckVectorEqual(a->grad()->Flatten(), TO_VEC(a_grad_data)))
            << "gradients: " << a->grad()->Flatten() << " != " << TO_VEC(a_grad_data) << "\n";
    };

    ut::test("TestTensorBroadcastAdd") = [&ctx] {
        std::vector<int> a_dims = {2, 3, 2};
        float a_data[] = {-0.725446f, 0.189742f,  -0.337311f, -0.109581f, -0.344868f, 0.790175f,
                          -0.882165f, -0.972059f, -0.197083f, 0.185942f,  -0.532996f, 0.035993f};
        float a_grad_data[] = {1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f,
                               1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f, 1.000000f};
        std::vector<int> b_dims = {2, 1, 2};
        float b_data[] = {-0.845297f, -0.231948f, -0.285176f, -0.540625f};
        float b_grad_data[] = {3.000000f, 3.000000f, 3.000000f, 3.000000f};
        std::vector<int> result_dims = {2, 3, 2};
        float result_data[] = {-1.570742f, -0.042206f, -1.182608f, -0.341529f, -1.190165f, 0.558227f,
                               -1.167341f, -1.512683f, -0.482260f, -0.354683f, -0.818172f, -0.504632f};

        auto &a = *ctx.NewTensor(a_dims)->Fill(a_data);
        auto &b = *ctx.NewTensor(b_dims)->Fill(b_data);
        auto &c = a + b;
        c.Forward();
        c.ZeroGrad();
        c.Backward();

        ut::expect(CheckVectorEqual(c.Flatten(), TO_VEC(result_data)))
            << c.Flatten() << " != " << TO_VEC(result_data) << "\n";
        ut::expect(CheckVectorEqual(a.grad()->Flatten(), TO_VEC(a_grad_data)))
            << "a grad: " << a.grad()->Flatten() << " != " << TO_VEC(a_grad_data) << "\n";
        ut::expect(CheckVectorEqual(b.grad()->Flatten(), TO_VEC(b_grad_data)))
            << "b grad: " << b.grad()->Flatten() << " != " << TO_VEC(b_grad_data) << "\n";
    };

    ut::test("TestTensorBroadcastMatMul") = [&ctx] {
        std::vector<int> a_dims = {2, 3, 2};
        float a_data[] = {-0.076818f, -0.239651f, -0.489204f, -0.388082f, -0.071079f, -0.728905f,
                          -0.950086f, 0.922597f,  -0.275149f, -0.701350f, 0.798951f,  -0.661272f};
        float a_grad_data[] = {-1.944579f, 0.638576f, -1.944579f, 0.638576f, -1.944579f, 0.638576f,
                               -1.944579f, 0.638576f, -1.944579f, 0.638576f, -1.944579f, 0.638576f};
        std::vector<int> b_dims = {1, 3, 2};
        float b_data[] = {-0.060925f, 0.826868f, -0.963677f, -0.242371f, -0.919977f, 0.054079f};
        float b_grad_data[] = {-1.063383f, -1.796663f, -1.063383f, -1.796663f, -1.063383f, -1.796663f};
        std::vector<int> result_dims = {2, 3, 3};
        float result_data[] = {-0.193480f, 0.132112f, 0.057710f, -0.291088f, 0.565494f,  0.429069f,
                               -0.598378f, 0.245162f, 0.025973f, 0.820750f,  0.691965f,  0.923950f,
                               -0.563161f, 0.435141f, 0.215202f, -0.595460f, -0.609658f, -0.770778f};

        auto &a = *ctx.NewTensor(a_dims)->Fill(a_data);
        auto &b = *ctx.NewTensor(b_dims)->Fill(b_data);
        auto &c = a.MatMul(b);
        c.Forward();
        c.ZeroGrad();
        c.Backward();
        ut::expect(CheckVectorEqual(c.Flatten(), TO_VEC(result_data)))
            << c.Flatten() << " != " << TO_VEC(result_data) << "\n";
        ut::expect(CheckVectorEqual(a.grad()->Flatten(), TO_VEC(a_grad_data)))
            << "a grad: " << a.grad()->Flatten() << " != " << TO_VEC(a_grad_data) << "\n";
        ut::expect(CheckVectorEqual(b.grad()->Flatten(), TO_VEC(b_grad_data)))
            << "b grad: " << b.grad()->Flatten() << " != " << TO_VEC(b_grad_data) << "\n";
    };

    ut::test("TestTensorTranspose") = [&ctx] {
        auto x = ctx.NewTensor({2, 3, 4})->Fill<float>(
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
        auto &x_t1 = x->Transpose(0, 1);
        x_t1.Forward();
        auto &x_t2 = x->Transpose(1, 2);
        x_t2.Forward();

        std::vector<float> x_t1_data = {0,  1,  2,  3,  12, 13, 14, 15, 4,  5,  6,  7,
                                        16, 17, 18, 19, 8,  9,  10, 11, 20, 21, 22, 23};
        ut::expect(CheckVectorEqual(x_t1.Flatten(), x_t1_data)) << x_t1.Flatten() << " != " << x_t1_data << "\n";

        std::vector<float> x_t2_data = {0,  4,  8,  1,  5,  9,  2,  6,  10, 3,  7,  11,
                                        12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23};

        ut::expect(CheckVectorEqual(x_t2.Flatten(), x_t2_data)) << x_t2.Flatten() << " != " << x_t2_data << "\n";

        auto x_t1_grad = x_t1.RandomGrad()->grad();
        x_t1.Backward(false);
        auto x_t1_grad_t1 = x_t1_grad->Transpose(0, 1);
        x_t1_grad_t1.Forward();

        ut::expect(CheckVectorEqual(x->grad()->Flatten(), x_t1_grad_t1.Flatten()))
            << "x gradients: " << x->grad()->Flatten() << " != " << x_t1_grad_t1.Flatten() << "\n";
    };

    ut::test("TestTensorView1") = [&ctx] {
        float x_data[] = {-0.141649f, -0.125871f, -1.265625f, -0.557144f, -0.593670f, -0.271154f,
                          2.106452f,  -0.763804f, 0.276285f,  -1.824517f, -0.027871f, 0.173058f,
                          -1.062135f, -1.543186f, -0.903289f, -1.008523f, -1.003833f, -0.068816f,
                          0.494010f,  -0.588222f, 0.829179f,  -0.457088f, 1.069247f,  0.996707f};
        std::vector<int> x_dims = {2, 3, 4};
        std::vector<int> y_dims = {2, 3, 2, 2};
        std::vector<int> z_dims = {2, 1, 3, 4};
        auto x = ctx.NewTensor(x_dims)->Fill(x_data);
        auto &y = x->View(y_dims);
        auto y_grad = y.RandomGrad()->grad();
        auto &z = x->View(z_dims);
        auto z_grad = z.RandomGrad()->grad();
        auto x_grad = y_grad->View(x_dims) + z_grad->View(x_dims);
        x_grad.Forward();
        y.Forward();
        y.Backward(false);
        z.Forward();
        z.Backward(false);
        ut::expect(y.Dims() == y_dims) << "y shape: " << y.Dims() << " != " << y_dims << "\n";
        ut::expect(CheckVectorEqual(y.Flatten(), TO_VEC(x_data))) << y.Flatten() << " != " << TO_VEC(x_data) << "\n";
        ut::expect(z.Dims() == z_dims) << "z shape: " << z.Dims() << " != " << z_dims << "\n";
        ut::expect(CheckVectorEqual(z.Flatten(), TO_VEC(x_data))) << z.Flatten() << " != " << TO_VEC(x_data) << "\n";
        ut::expect(CheckVectorEqual(x->grad()->Flatten(), x_grad.Flatten()))
            << "x grad: " << x->grad()->Flatten() << " != " << x_grad.Flatten() << "\n";
    };

    ut::test("TestTensorView2") = [&ctx] {
        float x_data[] = {-0.141649f, -0.125871f, -1.265625f, -0.557144f, -0.593670f, -0.271154f,
                          2.106452f,  -0.763804f, 0.276285f,  -1.824517f, -0.027871f, 0.173058f,
                          -1.062135f, -1.543186f, -0.903289f, -1.008523f, -1.003833f, -0.068816f,
                          0.494010f,  -0.588222f, 0.829179f,  -0.457088f, 1.069247f,  0.996707f};
        std::vector<int> x_dims = {4, 6};
        std::vector<int> y_dims = {2, 2, 2, 3};
        std::vector<int> z_dims = {1, 4, 1, 6};
        auto x = ctx.NewTensor(x_dims)->Fill(x_data);
        auto &y = x->View(y_dims);
        auto y_grad = y.RandomGrad()->grad();
        auto &z = x->View(z_dims);
        auto z_grad = z.RandomGrad()->grad();
        auto x_grad = y_grad->View(x_dims) + z_grad->View(x_dims);
        x_grad.Forward();
        y.Forward();
        y.Backward(false);
        z.Forward();
        z.Backward(false);
        ut::expect(y.Dims() == y_dims) << "y shape: " << y.Dims() << " != " << y_dims << "\n";
        ut::expect(CheckVectorEqual(y.Flatten(), TO_VEC(x_data))) << y.Flatten() << " != " << TO_VEC(x_data) << "\n";
        ut::expect(z.Dims() == z_dims) << "z shape: " << z.Dims() << " != " << z_dims << "\n";
        ut::expect(CheckVectorEqual(z.Flatten(), TO_VEC(x_data))) << z.Flatten() << " != " << TO_VEC(x_data) << "\n";
        ut::expect(CheckVectorEqual(x->grad()->Flatten(), x_grad.Flatten()))
            << "x grad: " << x->grad()->Flatten() << " != " << x_grad.Flatten() << "\n";
    };

    ut::test("TestTensorView3") = [&ctx] {
        float x_data[] = {-0.141649f, -0.125871f, -1.265625f, -0.557144f, -0.593670f, -0.271154f,
                          2.106452f,  -0.763804f, 0.276285f,  -1.824517f, -0.027871f, 0.173058f,
                          -1.062135f, -1.543186f, -0.903289f, -1.008523f, -1.003833f, -0.068816f,
                          0.494010f,  -0.588222f, 0.829179f,  -0.457088f, 1.069247f,  0.996707f};
        std::vector<int> x_dims = {2, 2, 2, 3};
        std::vector<int> y_dims = {4, 6};
        std::vector<int> z_dims = {2, 4, 3};
        auto x = ctx.NewTensor(x_dims)->Fill(x_data);
        auto &y = x->View(y_dims);
        auto y_grad = y.RandomGrad()->grad();
        auto &z = x->View(z_dims);
        auto z_grad = z.RandomGrad()->grad();
        auto x_grad = y_grad->View(x_dims) + z_grad->View(x_dims);
        x_grad.Forward();
        y.Forward();
        y.Backward(false);
        z.Forward();
        z.Backward(false);
        ut::expect(y.Dims() == y_dims) << "y shape: " << y.Dims() << " != " << y_dims << "\n";
        ut::expect(CheckVectorEqual(y.Flatten(), TO_VEC(x_data))) << y.Flatten() << " != " << TO_VEC(x_data) << "\n";
        ut::expect(z.Dims() == z_dims) << "z shape: " << z.Dims() << " != " << z_dims << "\n";
        ut::expect(CheckVectorEqual(z.Flatten(), TO_VEC(x_data))) << z.Flatten() << " != " << TO_VEC(x_data) << "\n";
        ut::expect(CheckVectorEqual(x->grad()->Flatten(), x_grad.Flatten()))
            << "x grad: " << x->grad()->Flatten() << " != " << x_grad.Flatten() << "\n";
    };

    ut::test("TestTensorView4") = [&ctx] {
        float x_data[] = {-0.141649f, -0.125871f, -1.265625f, -0.557144f, -0.593670f, -0.271154f,
                          2.106452f,  -0.763804f, 0.276285f,  -1.824517f, -0.027871f, 0.173058f,

                          -1.062135f, -1.543186f, -0.903289f, -1.008523f, -1.003833f, -0.068816f,
                          0.494010f,  -0.588222f, 0.829179f,  -0.457088f, 1.069247f,  0.996707f};
        std::vector<int> x_dims = {2, 2, 6};
        std::vector<int> y_dims = {4, 2};
        float y_data[] = {-1.265625f, -0.557144f, 0.276285f, -1.824517f, -0.903289f, -1.008523f, 0.829179f, -0.457088f};
        std::vector<int> z_dims = {4, 1, 3};
        float z_data[] = {-0.557144f, -0.593670f, -0.271154f, -1.824517f, -0.027871f, 0.173058f,
                          -1.008523f, -1.003833f, -0.068816f, -0.457088f, 1.069247f,  0.996707f};
        auto x = ctx.NewTensor(x_dims)->Fill(x_data);
        auto &y = x->View(y_dims, 1, 2);
        auto &z = x->View(z_dims, 1, 2);
        y.Forward();
        z.Forward();
        ut::expect(y.Dims() == y_dims) << "y shape: " << y.Dims() << " != " << y_dims << "\n";
        ut::expect(CheckVectorEqual(y.Flatten(), TO_VEC(y_data))) << y.Flatten() << " != " << TO_VEC(y_data) << "\n";
        ut::expect(z.Dims() == z_dims) << "z shape: " << z.Dims() << " != " << z_dims << "\n";
        ut::expect(CheckVectorEqual(z.Flatten(), TO_VEC(z_data))) << z.Flatten() << " != " << TO_VEC(z_data) << "\n";

        y.ZeroGrad();
        auto y_grad = y.RandomGrad()->grad();
        y.Backward(false);
        auto &y_grad_truth = x->grad()->View(y_dims, 1, 2);
        y_grad_truth.Forward();
        ut::expect(CheckVectorEqual(y_grad->Flatten(), y_grad_truth.Flatten()))
            << "y grad: " << y_grad->Flatten() << " != " << y_grad_truth.Flatten() << "\n";

        z.ZeroGrad();
        auto z_grad = z.RandomGrad()->grad();
        z.Backward(false);
        auto &z_grad_truth = x->grad()->View(z_dims, 1, 2);
        z_grad_truth.Forward();
        ut::expect(CheckVectorEqual(z_grad->Flatten(), z_grad_truth.Flatten()))
            << "z grad: " << z_grad->Flatten() << " != " << z_grad_truth.Flatten() << "\n";
    };

    ut::test("TestTensorSplit") = [&ctx] {
        float x_data[] = {-0.141649f, -0.125871f, -1.265625f, -0.557144f, -0.593670f, -0.271154f, 2.106452f,
                          -0.763804f, 0.276285f,  -1.824517f, -0.027871f, 0.173058f,  -1.062135f, -1.543186f,
                          -0.903289f, -1.008523f, -1.003833f, -0.068816f, 0.494010f,  -0.588222f, 0.829179f,
                          -0.457088f, 1.069247f,  0.996707f,  0.069449f,  -0.040386f, -0.372081f, 1.546610f,
                          0.060131f,  -0.008539f, 1.380819f,  -1.187713f, 0.851277f,  0.755909f,  0.320159f,
                          0.047182f,  -1.942095f, -0.469703f, 0.206541f,  -0.570447f, -0.161775f, -0.976735f,
                          0.450884f,  0.903816f,  -0.139475f, -0.567350f, -0.927794f, -0.488965f};

        auto x = ctx.NewTensor({2, 6, 4})->Fill(x_data);
        auto splits = x->Split(2, 1);

        std::vector<float> a_data = {-0.141649f, -0.125871f, -1.265625f, -0.557144f, -0.593670f, -0.271154f,
                                     2.106452f,  -0.763804f, 0.069449f,  -0.040386f, -0.372081f, 1.546610f,
                                     0.060131f,  -0.008539f, 1.380819f,  -1.187713f};
        std::vector<float> b_data = {0.276285f,  -1.824517f, -0.027871f, 0.173058f, -1.062135f, -1.543186f,
                                     -0.903289f, -1.008523f, 0.851277f,  0.755909f, 0.320159f,  0.047182f,
                                     -1.942095f, -0.469703f, 0.206541f,  -0.570447f};
        std::vector<float> c_data = {-1.003833f, -0.068816f, 0.494010f,  -0.588222f, 0.829179f, -0.457088f,
                                     1.069247f,  0.996707f,  -0.161775f, -0.976735f, 0.450884f, 0.903816f,
                                     -0.139475f, -0.567350f, -0.927794f, -0.488965f};
        auto a = splits[0], b = splits[1], c = splits[2];
        c->Forward();
        ut::expect(CheckVectorEqual(c->Flatten(), c_data)) << c->Flatten() << " != " << c_data << "\n";

        auto splits2 = x->Split(2, 2);

        std::vector<float> e_data = {-0.141649f, -0.125871f, -0.593670f, -0.271154f, 0.276285f,  -1.824517f,
                                     -1.062135f, -1.543186f, -1.003833f, -0.068816f, 0.829179f,  -0.457088f,
                                     0.069449f,  -0.040386f, 0.060131f,  -0.008539f, 0.851277f,  0.755909f,
                                     -1.942095f, -0.469703f, -0.161775f, -0.976735f, -0.139475f, -0.567350f};
        std::vector<float> f_data = {-1.265625f, -0.557144f, 2.106452f, -0.763804f, -0.027871f, 0.173058f,
                                     -0.903289f, -1.008523f, 0.494010f, -0.588222f, 1.069247f,  0.996707f,
                                     -0.372081f, 1.546610f,  1.380819f, -1.187713f, 0.320159f,  0.047182f,
                                     0.206541f,  -0.570447f, 0.450884f, 0.903816f,  -0.927794f, -0.488965f};

        auto &d = a->MatMul(*b);
        d.Forward();
        d.Backward();
        ut::expect(CheckVectorEqual(a->Flatten(), a_data)) << a->Flatten() << " != " << a_data << "\n";
        ut::expect(CheckVectorEqual(b->Flatten(), b_data)) << b->Flatten() << " != " << b_data << "\n";
        std::vector<float> d_data = {0.129375f,  2.049810f,  0.139811f, -0.083424f,
                                     -0.017561f, -1.075018f, 0.430777f, 0.849955f};
        ut::expect(CheckVectorEqual(d.Flatten(), d_data)) << d.Flatten() << " != " << d_data << "\n";

        std::vector<float> x_grad_data = {
            -0.785850f, -3.367703f, -0.931161f, -0.835465f, -0.785850f, -3.367703f, -0.931161f, -0.835465f,
            -0.735318f, -0.397025f, 0.840827f,  -1.320947f, -0.735318f, -0.397025f, 0.840827f,  -1.320947f,
            0.000000f,  0.000000f,  0.000000f,  0.000000f,  0.000000f,  0.000000f,  0.000000f,  0.000000f,
            -1.090818f, 0.286207f,  0.526700f,  -0.523265f, -1.090818f, 0.286207f,  0.526700f,  -0.523265f,
            0.129580f,  -0.048925f, 1.008738f,  0.358897f,  0.129580f,  -0.048925f, 1.008738f,  0.358897f,
            0.000000f,  0.000000f,  0.000000f,  0.000000f,  0.000000f,  0.000000f,  0.000000f,  0.000000f};

        ut::expect(CheckVectorEqual(x->grad()->Flatten(), x_grad_data))
            << "x gradients: " << x->grad()->Flatten() << " != " << x_grad_data << "\n";

        auto e = splits2[0], f = splits2[1];
        e->Forward();
        f->Forward();
        ut::expect(CheckVectorEqual(e->Flatten(), e_data)) << e->Flatten() << " != " << e_data << "\n";
        ut::expect(CheckVectorEqual(f->Flatten(), f_data)) << f->Flatten() << " != " << f_data << "\n";

        auto &g = e->MatMul(*f);
        g.Forward();
        // forward is idempotent
        g.Forward();
        std::vector<float> g_data = {
            0.249402f,  -0.202235f, -0.017835f, 0.254894f,  0.004064f,  -0.276914f, 0.902435f,  -1.043429f, -0.030379f,
            0.809721f,  -0.133780f, -0.905041f, 0.666845f,  1.975553f,  -0.323448f, 1.590503f,  1.209709f,  -1.523092f,
            2.204041f,  -1.058645f, -0.237458f, 2.515754f,  0.383031f,  -2.673788f, 1.308817f,  -2.061964f, 0.016069f,
            0.976155f,  -0.455424f, -1.141936f, -0.794766f, 2.095752f,  -0.102213f, -0.288005f, 0.678492f,  0.431015f,
            -0.088302f, 0.143864f,  0.020329f,  0.037382f,  -0.005188f, -0.044687f, -0.035580f, 0.093172f,  0.018849f,
            0.017291f,  0.019394f,  -0.051614f, 0.852353f,  0.277656f,  0.308209f,  -0.255383f, 1.067030f,  -1.159422f,
            -0.003830f, -2.123810f, -0.643941f, -0.133182f, -1.300185f, 2.031532f,  -1.450435f, 0.936698f,  -0.097878f,
            0.523763f,  -0.955731f, 0.627684f,  -0.825572f, 0.481258f,  -0.071423f, 0.294836f,  -0.575667f, 0.406818f};

        ut::expect(CheckVectorEqual(g.Flatten(), g_data)) << g.Flatten() << " != " << g_data << "\n";
        g.Backward();
        std::vector<float> x_grad_data2 = {
            0.687073f,  -5.115632f, -2.626983f, -5.126099f, 0.687073f,  -5.115632f, -2.626983f, -5.126099f,
            0.737605f,  -2.144953f, -0.854996f, -5.611580f, 0.737605f,  -2.144953f, -0.854996f, -5.611580f,
            1.472923f,  -1.747928f, -1.695822f, -4.290633f, 1.472923f,  -1.747928f, -1.695822f, -4.290633f,
            -0.032290f, 0.536689f,  -0.735788f, -1.830068f, -0.032290f, 0.536689f,  -0.735788f, -1.830068f,
            1.188109f,  0.201558f,  -0.253751f, -0.947906f, 1.188109f,  0.201558f,  -0.253751f, -0.947906f,
            1.058529f,  0.250483f,  -1.262489f, -1.306803f, 1.058529f,  0.250483f,  -1.262489f, -1.306803f};
        ut::expect(CheckVectorEqual(x->grad()->Flatten(), x_grad_data2))
            << "x gradients: " << x->grad()->Flatten() << " != " << x_grad_data2 << "\n";
    };

    ut::test("TestTensorGelu") = [&ctx] {
        std::vector<int> dims = {2, 3};
        float x_data[] = {-1.119413f, 0.348449f, 0.720475f, -0.974813f, 1.244371f, 1.265503f};
        float x_grad_data[] = {0.031505f, 0.340091f, 1.085720f, 0.024649f, 2.494036f, 2.551048f};
        float y_data[] = {-0.147375f, 0.221696f, 0.550659f, -0.160819f, 1.111398f, 1.135126f};
        float y_grad_data[] = {-0.294750f, 0.443392f, 1.101318f, -0.321638f, 2.222795f, 2.270252f};
        auto x = ctx.NewTensor(dims)->Fill(x_data);
        auto y = x->Gelu().FillGrad(y_grad_data);
        y->Forward();
        y->Backward(false);
        ut::expect(CheckVectorEqual(y->Flatten(), TO_VEC(y_data))) << y->Flatten() << " != " << TO_VEC(y_data) << "\n";
        ut::expect(CheckVectorEqual(x->grad()->Flatten(), TO_VEC(x_grad_data)))
            << "x gradients: " << x->grad()->Flatten() << " != " << TO_VEC(x_grad_data) << "\n";
    };

    ut::test("TestTensorSoftmax") = [&ctx] {
        std::vector<int> dims = {2, 2, 5};
        float x_data[] = {0.111361f,  -0.000386f, -1.804357f, 0.369323f,  1.306343f,  -0.004050f, 1.898653f,
                          1.455873f,  0.272952f,  1.880325f,  0.164516f,  -0.822872f, 1.178338f,  -0.938773f,
                          -1.107920f, 0.836621f,  0.351749f,  -0.460701f, 0.153248f,  -0.459156f};
        float x_grad_data[] = {-0.053102f, -0.051777f, -0.013515f, -0.051447f, 0.169841f,  -0.022711f, 0.040656f,
                               -0.025984f, -0.027836f, 0.035875f,  -0.078526f, -0.050058f, 0.214313f,  -0.045784f,
                               -0.039946f, 0.093127f,  -0.009149f, -0.030365f, -0.023234f, -0.030379f};

        float y_data[] = {0.150618f, 0.134694f, 0.022176f, 0.194944f, 0.497568f, 0.050223f, 0.336697f,
                          0.216243f, 0.066253f, 0.330583f, 0.210944f, 0.078587f, 0.581386f, 0.069987f,
                          0.059096f, 0.374864f, 0.230832f, 0.102436f, 0.189273f, 0.102595f};

        float y_grad_data[] = {0.301237f, 0.269387f, 0.044353f, 0.389888f, 0.995135f, 0.100447f, 0.673395f,
                               0.432487f, 0.132506f, 0.661165f, 0.421889f, 0.157174f, 1.162773f, 0.139973f,
                               0.118191f, 0.749728f, 0.461664f, 0.204873f, 0.378546f, 0.205189f};
        auto x = ctx.NewTensor(dims)->Fill(x_data);
        auto y = x->Softmax().FillGrad(y_grad_data);
        y->Forward();
        y->Backward(false);
        ut::expect(CheckVectorEqual(y->Flatten(), TO_VEC(y_data))) << y->Flatten() << " != " << TO_VEC(y_data) << "\n";
        ut::expect(CheckVectorEqual(x->grad()->Flatten(), TO_VEC(x_grad_data)))
            << "x gradients: " << x->grad()->Flatten() << " != " << TO_VEC(x_grad_data) << "\n";
    };

    ut::test("TestTensorSoftmaxCasual") = [&ctx] {
        std::vector<int> dims = {2, 2, 5};
        float x_data[] = {0.111361f,  -0.000386f, -1.804357f, 0.369323f,  1.306343f,  -0.004050f, 1.898653f,
                          1.455873f,  0.272952f,  1.880325f,  0.164516f,  -0.822872f, 1.178338f,  -0.938773f,
                          -1.107920f, 0.836621f,  0.351749f,  -0.460701f, 0.153248f,  -0.459156f};
        float y_grad_data[] = {0.301237f, 0.269387f, 0.044353f, 0.389888f, 0.995135f, 0.100447f, 0.673395f,
                               0.432487f, 0.132506f, 0.661165f, 0.421889f, 0.157174f, 1.162773f, 0.139973f,
                               0.118191f, 0.749728f, 0.461664f, 0.204873f, 0.378546f, 0.205189f};
        auto x = ctx.NewTensor(dims)->Fill(x_data);
        auto y = x->Softmax(true).FillGrad(y_grad_data);
        y->Forward();
        y->Backward(false);

        float y_data[] = {1, 0, 0, 0, 0, 0.129803, 0.870197, 0, 0, 0, 1, 0, 0, 0, 0, 0.618898, 0.381102, 0, 0, 0};
        float x_grad_data[] = {0, 0, 0, 0, 0, -0.0647168, 0.0647168,  0, 0, 0,
                               0, 0, 0, 0, 0, 0.0679438,  -0.0679437, 0, 0, 0};

        ut::expect(CheckVectorEqual(y->Flatten(), TO_VEC(y_data))) << y->Flatten() << " != " << TO_VEC(y_data) << "\n";
        ut::expect(CheckVectorEqual(x->grad()->Flatten(), TO_VEC(x_grad_data)))
            << "x gradients: " << x->grad()->Flatten() << " != " << TO_VEC(x_grad_data) << "\n";
    };

    ut::test("TestTensorCrossEntropy") = [&ctx] {
        std::vector<int> x_dims = {2, 2, 5};
        float x_data[] = {0.651805f,  -0.589272f, 1.826531f,  2.116205f,  -0.652173f, 1.867640f,  -0.457756f,
                          -0.216584f, -1.390391f, 0.527376f,  -1.137916f, -1.974276f, -0.499488f, -1.299625f,
                          -1.138694f, -1.578463f, -1.322330f, -2.240308f, 0.190388f,  -0.643646f};
        float x_grad_data[] = {-0.222596f, 0.007922f, 0.088714f, 0.118521f,  0.007439f,  0.164216f, -0.233949f,
                               0.020429f,  0.006316f, 0.042988f, 0.048293f,  -0.229075f, 0.091444f, 0.041083f,
                               0.048256f,  0.022284f, 0.028790f, -0.238504f, 0.130677f,  0.056752f};
        std::vector<int> y_dims = {2, 2};
        int targets_data[] = {0, 1, 1, 2};
        float y_data[] = {2.210769f, 2.745677f, 2.480527f, 3.079425f};
        float y_grad_data[] = {0.250000f, 0.250000f, 0.250000f, 0.250000f};

        auto x = ctx.NewTensor(x_dims)->Fill(x_data);
        auto targets = ctx.NewTensor(y_dims, kI32)->Fill(targets_data);
        auto y = x->Softmax().CrossEntropy(*targets).FillGrad(y_grad_data);

        y->Forward();
        y->Backward(false);
        ut::expect(CheckVectorEqual(y->Flatten(), TO_VEC(y_data))) << y->Flatten() << " != " << TO_VEC(y_data) << "\n";
        ut::expect(CheckVectorEqual(x->grad()->Flatten(), TO_VEC(x_grad_data)))
            << "x gradients: " << x->grad()->Flatten() << " != " << TO_VEC(x_grad_data) << "\n";
    };
}

int main() {
    TensorContext ctx((size_t)1024 * 1024);

    TestTensor(ctx);
    ctx.PrintLayout();

    return 0;
}
