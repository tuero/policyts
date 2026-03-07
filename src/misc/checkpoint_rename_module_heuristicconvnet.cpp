#include <libpolicyts/libpolicyts.h>

#include <nlohmann/json.hpp>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>

ABSL_FLAG(std::string, model_path, "", "Path for the twoheaded convnet model wrapper.");
ABSL_FLAG(std::string, output_dir, "/opt/pts/", "Base path to store all checkpoints and metrics");

using json = nlohmann::json;
using ModelT = libpts::model::HeuristicConvNetWrapper;

// -------------

struct HeuristicConvNetOutput {
    torch::Tensor heuristic;
};

class HeuristicConvNetImpl : public torch::nn::Module {
public:
    /**
     * ResNet style heuristic convnet
     * @param observation_shape Input observation shape to the network
     * @param resnet_channels Number of channels for each resenet block
     * @param resnet_blocks Number of resnet blocks
     * @param reduce_channels Number of channels in the heuristic reduce head
     * @param mlp_layers Hidden layer sizes for the heuristic head MLP
     * @param use_batchnorm Flag to use batchnorm in the resnet layers
     */
    HeuristicConvNetImpl(
        const libpts::ObservationShape &observation_shape,
        int resnet_channels,
        int resnet_blocks,
        int reduce_channels,
        const std::vector<int> &mlp_layers,
        bool use_batchnorm
    )
        : input_channels_(observation_shape.c),
          input_height_(observation_shape.h),
          input_width_(observation_shape.w),
          resnet_channels_(resnet_channels),
          reduce_channels_(reduce_channels),
          mlp_input_size_(reduce_channels_ * input_height_ * input_width_),
          resnet_head_(
              libpts::model::ResidualHead(input_channels_, resnet_channels_, use_batchnorm, "representation_")
          ),
          conv1x1_(libpts::model::conv1x1(resnet_channels_, reduce_channels_)),
          mlp_(mlp_input_size_, mlp_layers, 1, "heuristic_head_")
    {
        // ResNet body
        for (int i = 0; i < resnet_blocks; ++i) {
            resnet_layers_->push_back(libpts::model::ResidualBlock(resnet_channels_, i, use_batchnorm));
        }
        register_module("representation_head", resnet_head_);
        register_module("representation_layers", resnet_layers_);
        register_module("heuristic_1x1", conv1x1_);
        register_module("mlp", mlp_);
    }

    [[nodiscard]] auto forward(torch::Tensor x) -> HeuristicConvNetOutput
    {
        torch::Tensor output = resnet_head_->forward(x);
        // ResNet body
        for (int i = 0; i < static_cast<int>(resnet_layers_->size()); ++i) {
            output = resnet_layers_[i]->as<libpts::model::ResidualBlock>()->forward(output);
        }

        torch::Tensor heuristic = conv1x1_->forward(output);
        heuristic = heuristic.view({-1, mlp_input_size_});
        heuristic = mlp_->forward(heuristic);
        return {heuristic};
    }

private:
    int input_channels_;
    int input_height_;
    int input_width_;
    int resnet_channels_;
    int reduce_channels_;
    int mlp_input_size_;
    libpts::model::ResidualHead resnet_head_;
    torch::nn::Conv2d conv1x1_;    // Conv pass before passing to heuristic mlp
    libpts::model::MLP mlp_;
    torch::nn::ModuleList resnet_layers_;
};
TORCH_MODULE(HeuristicConvNet);

// -------------

static std::string remap_new_to_old(std::string k, const std::string &new_prefix, const std::string &old_prefix)
{
    if (k == new_prefix) return old_prefix;
    if (k.rfind(new_prefix + ".", 0) == 0) {
        return old_prefix + k.substr(new_prefix.size());
    }
    return k;
}

template <typename OldM, typename NewM>
static void
    copy_with_prefix_rename(const OldM &oldm, NewM &newm, const std::string &new_prefix, const std::string &old_prefix)
{
    torch::NoGradGuard guard;
    auto old_params = oldm->named_parameters();
    auto new_params = newm->get_named_parameters()["heuristic_convnet"];

    for (auto &item : new_params) {
        const std::string &new_key = item.key();
        std::string old_key = remap_new_to_old(new_key, new_prefix, old_prefix);
        if (const auto *src = old_params.find(old_key)) {
            item.value().copy_(*src);
        }
    }

    auto old_bufs = oldm->named_buffers();
    auto new_bufs = newm->get_named_buffers()["heuristic_convnet"];

    for (auto &item : new_bufs) {
        const std::string &new_key = item.key();
        std::string old_key = remap_new_to_old(new_key, new_prefix, old_prefix);
        if (const auto *src = old_bufs.find(old_key)) {
            item.value().copy_(*src);
        }
    }
}

int main(int argc, char **argv)
{
    absl::ParseCommandLine(argc, argv);

    libpts::ObservationShape obs_shape{34, 14, 14};

    std::ifstream f(absl::GetFlag(FLAGS_model_path));
    json model_config_json = json::parse(f);

    int resnet_channels = model_config_json["resnet_channels"].template get<int>();
    int resnet_blocks = model_config_json["resnet_blocks"].template get<int>();
    int reduce_channels = model_config_json["heuristic_channels"].template get<int>();
    std::vector<int> mlp_layers = model_config_json["heuristic_mlp_layers"].template get<std::vector<int>>();
    bool use_batchnorm = model_config_json["use_batchnorm"].template get<bool>();

    // Old model
    HeuristicConvNet model_old(obs_shape, resnet_channels, resnet_blocks, reduce_channels, mlp_layers, use_batchnorm);
    std::string full_path = absl::StrCat(absl::GetFlag(FLAGS_output_dir), "/checkpoints/checkpoint-", -1, ".pt");
    torch::load(model_old, full_path);

    // Init new output model
    auto output_model_wrapper =
        std::make_shared<ModelT>(model_config_json, obs_shape, "cpu", absl::GetFlag(FLAGS_output_dir));
    output_model_wrapper->print();

    copy_with_prefix_rename(model_old, output_model_wrapper, "mlp.head_mlp", "mlp.heuristic_head_mlp");

    output_model_wrapper->save_checkpoint_without_optimizer();
}
