#include <torch/script.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be CUDA tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

torch::Tensor knn_cuda(torch::Tensor x, torch::Tensor y, int64_t k, torch::Tensor batch_x, torch::Tensor batch_y);

torch::Tensor knn(torch::Tensor x, torch::Tensor y, int64_t k, torch::Tensor batch_x, torch::Tensor batch_y)
{
    CHECK_CUDA(x);
    IS_CONTIGUOUS(x);

    CHECK_CUDA(y);
    IS_CONTIGUOUS(y);

    CHECK_CUDA(batch_x);
    CHECK_CUDA(batch_y);

    return knn_cuda(x, y, k, batch_x, batch_y);
}

static auto registry = torch::RegisterOperators("torch_cluster_ops::knn", &knn);
