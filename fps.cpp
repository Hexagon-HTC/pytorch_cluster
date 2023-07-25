#include <torch/script.h>

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be CUDA tensor")
#define IS_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " is not contiguous");

torch::Tensor fps_cuda(torch::Tensor x, torch::Tensor batch, double ratio);

torch::Tensor fps(torch::Tensor x, torch::Tensor batch, double ratio)
{
    CHECK_CUDA(x);
    IS_CONTIGUOUS(x);
    CHECK_CUDA(batch);

    return fps_cuda(x, batch, ratio);
}

static auto registry = torch::RegisterOperators("torch_cluster_ops::fps", &fps);
