#include <libpolicyts/libpolicyts.h>

#include <nlohmann/json.hpp>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_cat.h>
#include <spdlog/spdlog.h>

#include <ranges>
#include <string>
#include <vector>

constexpr std::size_t INF_SIZE_T = std::numeric_limits<std::size_t>::max();
constexpr double INF_D = std::numeric_limits<double>::max();
constexpr int INF_I = std::numeric_limits<int>::max();

// NOLINTBEGIN
ABSL_FLAG(std::string, environment, "", "String name of the environment");
ABSL_FLAG(std::string, problems_path, "", "Path to problems file");
ABSL_FLAG(std::string, output_dir, "/opt/pts/", "Base path to store all checkpoints and metrics");
ABSL_FLAG(std::string, model_path, "", "Path for the twoheaded convnet model wrapper.");
ABSL_FLAG(int, search_budget, 4000, "Maximum number of expanded nodes before termination");
ABSL_FLAG(int, inference_batch_size, 32, "Number of search expansions to batch per inference query");
ABSL_FLAG(double, mix_epsilon, 0, "Percentage to mix with uniform policy");
ABSL_FLAG(int, seed, 0, "Seed for all sources of RNG");
ABSL_FLAG(std::size_t, num_train, INF_SIZE_T, "Number of instances of the max to use for training");
ABSL_FLAG(std::size_t, num_validate, INF_SIZE_T, "Number of instances of the max to use for validation");
ABSL_FLAG(int, max_iterations, INF_I, "Maximum number of iterations of running the bootstrap process");
ABSL_FLAG(int, max_budget, INF_I, "Maximum search budget before terminating the bootstrap process");
ABSL_FLAG(double, time_budget, INF_D, "Budget in seconds before terminating");
ABSL_FLAG(double, validation_solved_ratio, 0.99, "Percentage of validation set to solve before checkpointing");
ABSL_FLAG(
    libpts::train::BootstrapPolicy,
    bootstrap_policy,
    libpts::train::BootstrapPolicy::LTS_CM,
    "Bootstrap policy, double or lts_cm for the context models version"
);
ABSL_FLAG(double, bootstrap_factor, 0.1, "Bootstrap increase factor if using LTS_CM policy");
ABSL_FLAG(int, learning_batch_size, 256, "Batch size used for model updates");
ABSL_FLAG(int, num_threads, 1, "Number of threads to run in the search thread pool");
ABSL_FLAG(int, num_problems_per_batch, 32, "Number of problems per bootstrap iteration");
ABSL_FLAG(int, grad_steps, 10, "Number of gradient updates per batch iteration");
ABSL_FLAG(int, device_num, 0, "Torch cuda device number to use (defaults to 1)");
// NOLINTEND

namespace {

namespace phs = libpts::algorithm::phs;
using ModelT = libpts::model::TwoHeadedConvNetWrapper;

template <typename EnvT>
auto create_search_inputs(
    const std::vector<EnvT> &problems,
    std::shared_ptr<libpts::StopToken> stop_token,
    std::shared_ptr<ModelT> model_wrapper
) {
    using SearchInputT = phs::SearchInput<EnvT, ModelT>;
    std::vector<SearchInputT> search_inputs;
    for (auto i : std::views::iota(static_cast<std::size_t>(0)) | std::views::take(problems.size())) {
        search_inputs.emplace_back(
            std::format("puzzle_{:d}", i),
            problems[i],
            absl::GetFlag(FLAGS_search_budget),
            absl::GetFlag(FLAGS_inference_batch_size),
            absl::GetFlag(FLAGS_mix_epsilon),
            stop_token,
            model_wrapper
        );
    }
    return libpts::train::split_train_validate(
        search_inputs,
        absl::GetFlag(FLAGS_num_train),
        absl::GetFlag(FLAGS_num_validate),
        absl::GetFlag(FLAGS_seed)
    );
}

using json = nlohmann::json;

template <typename EnvT>
class PHSLearner {
    using SearchInputT = phs::SearchInput<EnvT, ModelT>;
    using SearchOutputT = phs::SearchOutput<EnvT>;
    using LearningInputT = typename ModelT::LearningInput;

public:
    PHSLearner(std::shared_ptr<ModelT> model_wrapper, int learning_batch_size, int grad_steps)
        : model_wrapper_(std::move(model_wrapper)),
          learning_batch_size_(learning_batch_size),
          grad_steps_(grad_steps) {}

    void checkpoint() {
        model_wrapper_->save_checkpoint_without_optimizer(-1);
    }

    void preprocess(
        [[maybe_unused]] std::vector<phs::SearchInput<EnvT, ModelT>> &batch,
        [[maybe_unused]] bool is_train
    ) {}

    void process_data(std::vector<SearchOutputT> &&search_outputs) {
        learning_inputs.clear();
        for (auto &&result : std::move(search_outputs)) {
            if (result.solution_found) {
                for (auto &&[s, a, c] : std::views::zip(
                         result.solution_path_states,
                         result.solution_path_actions,
                         result.solution_path_costs
                     ))
                {
                    learning_inputs.push_back(
                        LearningInputT{
                        .observation = s.get_observation(),
                        .target_action = a,
                        .target_cost_to_goal = c,
                        .solution_expanded = result.num_expanded
                        }
                    );
                }
            }
        }
    }

    void learning_step(std::mt19937 &rng) {
        if (!learning_inputs.empty()) {
            for (int _ : std::views::iota(0) | std::views::take(grad_steps_)) {
                std::ranges::shuffle(learning_inputs, rng);
                double loss = 0;
                for (auto &&batch_item : learning_inputs | std::views::chunk(learning_batch_size_)) {
                    std::vector<LearningInputT> batch = std::ranges::to<std::vector>(batch_item);
                    loss += model_wrapper_->learn(batch);
                }
                SPDLOG_INFO(
                    "Loss: {:.4f}",
                    loss / libpts::ceil_div(static_cast<int>(learning_inputs.size()), learning_batch_size_)
                );
            }
        }
    }

private:
    std::shared_ptr<ModelT> model_wrapper_;
    int learning_batch_size_;
    int grad_steps_;
    std::vector<LearningInputT> learning_inputs;
};

template <typename EnvT>
void templated_main() {
    using SearchInputT = phs::SearchInput<EnvT, ModelT>;
    using SearchOutputT = phs::SearchOutput<EnvT>;

    // Load problems
    auto [problems, _] = libpts::env::load_problems<EnvT>(absl::GetFlag(FLAGS_problems_path));

    // Init model
    std::ifstream f(absl::GetFlag(FLAGS_model_path));
    json model_config_json = json::parse(f);
    auto model_wrapper = std::make_shared<ModelT>(
        model_config_json,
        problems[0].observation_shape(),
        EnvT::num_actions,
        std::format("cuda:{:d}", absl::GetFlag(FLAGS_device_num)),
        absl::GetFlag(FLAGS_output_dir)
    );

    model_wrapper->print();

    // Install signaller
    std::shared_ptr<libpts::StopToken> stop_token = libpts::signal_installer();

    // Create search inputs
    auto [problems_train, problems_validate] = create_search_inputs(problems, stop_token, model_wrapper);

    // Learner
    PHSLearner<EnvT> phs_learner(
        model_wrapper,
        absl::GetFlag(FLAGS_learning_batch_size),
        absl::GetFlag(FLAGS_grad_steps)
    );

    std::mt19937 rng(static_cast<std::mt19937::result_type>(absl::GetFlag(FLAGS_seed)));
    libpts::train::train_bootstrap<SearchInputT, SearchOutputT>(
        problems_train,
        problems_validate,
        phs::search<EnvT, ModelT>,
        phs_learner,
        absl::GetFlag(FLAGS_output_dir),
        rng,
        stop_token,
        absl::GetFlag(FLAGS_search_budget),
        absl::GetFlag(FLAGS_validation_solved_ratio),
        absl::GetFlag(FLAGS_num_threads),
        absl::GetFlag(FLAGS_num_problems_per_batch),
        absl::GetFlag(FLAGS_max_iterations),
        absl::GetFlag(FLAGS_max_budget),
        absl::GetFlag(FLAGS_time_budget),
        absl::GetFlag(FLAGS_bootstrap_policy),
        absl::GetFlag(FLAGS_bootstrap_factor)
    );
}
}    // namespace

int main(int argc, char **argv) {
    absl::ParseCommandLine(argc, argv);

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(absl::GetFlag(FLAGS_output_dir));

    // Initialize torch and loggers (console + file)
    libpts::model::init_torch(static_cast<uint64_t>(absl::GetFlag(FLAGS_seed)));
    libpts::init_loggers(false, absl::GetFlag(FLAGS_output_dir), "_train");

    // Dump invocation of program
    libpts::log_flags(argc, argv);

    if (absl::GetFlag(FLAGS_environment) == libpts::env::BoulderDashState::name) {
        templated_main<libpts::env::BoulderDashState>();
    } else if (absl::GetFlag(FLAGS_environment) == libpts::env::CraftWorldState::name) {
        templated_main<libpts::env::CraftWorldState>();
    } else if (absl::GetFlag(FLAGS_environment) == libpts::env::SokobanState::name) {
        templated_main<libpts::env::SokobanState>();
    } else if (absl::GetFlag(FLAGS_environment) == libpts::env::TSPDeadlockState::name) {
        templated_main<libpts::env::TSPDeadlockState>();
    } else {
        SPDLOG_ERROR("Unknown environment type: {:s}.", absl::GetFlag(FLAGS_environment));
        std::exit(1);
    }

    libpts::close_loggers();
}
