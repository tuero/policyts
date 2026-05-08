#include <libpolicyts/libpolicyts.h>

#include "siirlts_rerooter.h"

#include <nlohmann/json.hpp>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_cat.h>
#include <spdlog/spdlog.h>

#include <ranges>
#include <string>
#include <vector>

constexpr double INF_D = std::numeric_limits<double>::max();
constexpr int INF_I = std::numeric_limits<int>::max();

namespace rlts = libpts::algorithm::rlts;

// NOLINTBEGIN
ABSL_FLAG(std::string, environment, "", "String name of the environment");
ABSL_FLAG(std::string, problems_path, "", "Path to problems file");
ABSL_FLAG(std::string, output_dir, "/opt/pts/", "Base path to store all checkpoints and metrics");
ABSL_FLAG(std::string, model_path, "", "Path for the twoheaded convnet model wrapper.");
ABSL_FLAG(int, search_budget, INF_I, "Maximum number of expanded nodes before termination");
ABSL_FLAG(int, inference_batch_size, 32, "Number of search expansions to batch per inference query");
ABSL_FLAG(double, mix_epsilon, 0.01, "Percentage to mix with uniform policy");
ABSL_FLAG(ClusterLevel, cluster_level, ClusterLevel::Half, "Level in the Louvain clustering to sample from");
ABSL_FLAG(double, graph_update_factor, 1.2, "Update frequency factor for the graph update mode");
ABSL_FLAG(double, ua, 1, "Coefficient for clustering rerooter weight before adding");
ABSL_FLAG(double, ub, 1, "Coefficient for heuristic rerooter weight before adding");
ABSL_FLAG(double, alpha, 1, "Coefficient for heuristic term inside exp(-alpha * h)");
ABSL_FLAG(RobustMode, robust_mode, RobustMode::None, "The robust correction mode to use");
ABSL_FLAG(rlts::CostMode, cost_mode, rlts::CostMode::Slenderness, "The cost mode to use (slenderness, dpi)");
ABSL_FLAG(
    libpts::algorithm::rlts::PruningPolicy,
    prune_policy,
    libpts::algorithm::rlts::PruningPolicy::Eager,
    "Pruning mode (none, passive, eager)"
);
ABSL_FLAG(WeightMode, weight_mode, WeightMode::AllCountParentIncrement, "Weight mode");
ABSL_FLAG(int, max_iterations, INF_I, "Budget in number of iterations before terminating training/testing procedure");
ABSL_FLAG(double, time_budget, INF_D, "Budget in seconds before terminating");
ABSL_FLAG(int, num_threads, 1, "Number of threads to run in the search thread pool");
ABSL_FLAG(std::string, export_suffix, "", "Export suffix to place on output logs/files");
ABSL_FLAG(int, device_num, 0, "Torch cuda device number to use (defaults to 1)");
// NOLINTEND

namespace {

template <typename EnvT, typename ModelT>
auto create_search_inputs(
    const std::vector<EnvT> &problems,
    std::shared_ptr<libpts::StopToken> stop_token,
    std::shared_ptr<ModelT> model_wrapper
)
{
    using SearchInputT = rlts::SearchInput<EnvT, ModelT, SIIRLTSRerooter<EnvT>>;
    std::vector<SearchInputT> search_inputs;
    for (auto i : std::views::iota(static_cast<std::size_t>(0)) | std::views::take(problems.size())) {
        search_inputs.push_back(
            SearchInputT{
            .puzzle_name = std::format("puzzle_{:d}", i),
            .state = problems[i],
            .search_budget = absl::GetFlag(FLAGS_search_budget),
            .inference_batch_size = absl::GetFlag(FLAGS_inference_batch_size),
            .mix_epsilon = absl::GetFlag(FLAGS_mix_epsilon),
            .cost_mode = absl::GetFlag(FLAGS_cost_mode),
            .prune_policy = absl::GetFlag(FLAGS_prune_policy),
            .stop_token = stop_token,
            .model = model_wrapper,
            .rerooter = SIIRLTSRerooter<EnvT>{
            absl::GetFlag(FLAGS_robust_mode),
            absl::GetFlag(FLAGS_cluster_level),
            absl::GetFlag(FLAGS_weight_mode),
            absl::GetFlag(FLAGS_graph_update_factor),
            absl::GetFlag(FLAGS_ua),
            absl::GetFlag(FLAGS_ub),
            absl::GetFlag(FLAGS_alpha),
            0,
            }
            }
        );
    }
    return search_inputs;
}

using json = nlohmann::json;

template <typename EnvT, typename ModelT>
void runner(json &model_config_json)
{
    using SearchInputT = rlts::SearchInput<EnvT, ModelT, SIIRLTSRerooter<EnvT>>;
    using SearchOutputT = rlts::SearchOutput<EnvT, SIIRLTSRerooterSearchOutput>;

    // Load problems
    auto [problems, _] = libpts::env::load_problems<EnvT>(absl::GetFlag(FLAGS_problems_path));
    if (problems.empty()) {
        SPDLOG_ERROR("No problems were loaded.");
        std::exit(1);
    }

    // Init model
    auto model_wrapper = std::make_shared<ModelT>(
        model_config_json,
        problems[0].observation_shape(),
        EnvT::num_actions,
        std::format("cuda:{:d}", absl::GetFlag(FLAGS_device_num)),
        absl::GetFlag(FLAGS_output_dir)
    );

    model_wrapper->load_checkpoint_without_optimizer(-1);
    model_wrapper->print();

    // Install signaller
    std::shared_ptr<libpts::StopToken> stop_token = libpts::signal_installer();

    // Create search inputs
    auto search_inputs = create_search_inputs(problems, stop_token, model_wrapper);

    using MetricsTrackerT = rlts::RLTSMetricsTracker<EnvT, SIIRLTSRerooterSearchOutput>;
    libpts::test::test_runner<SearchInputT, SearchOutputT, MetricsTrackerT>(
        search_inputs,
        rlts::search<EnvT, ModelT, SIIRLTSRerooter<EnvT>>,
        absl::GetFlag(FLAGS_output_dir),
        stop_token,
        absl::GetFlag(FLAGS_search_budget),
        absl::GetFlag(FLAGS_num_threads),
        absl::GetFlag(FLAGS_max_iterations),
        absl::GetFlag(FLAGS_time_budget),
        absl::GetFlag(FLAGS_export_suffix)
    );
}

template <typename EnvT>
void templated_main()
{
    // Load model json
    std::ifstream f(absl::GetFlag(FLAGS_model_path));
    json model_config_json = json::parse(f);

    // Check model type
    if (!model_config_json.contains("model_type")) {
        spdlog::error("Model config json missing 'model_type' key.");
        std::exit(1);
    }
    if (model_config_json["model_type"] == "policy_convnet") {
        runner<EnvT, libpts::model::PolicyConvNetWrapper>(model_config_json);
    } else if (model_config_json["model_type"] == "twoheaded_convnet") {
        runner<EnvT, libpts::model::TwoHeadedConvNetWrapper>(model_config_json);
    } else {
        spdlog::error("Unsupported model_type.");
        std::exit(1);
    }
}
}    // namespace

int main(int argc, char **argv)
{
    absl::ParseCommandLine(argc, argv);

    // Create output directory if it doesn't exist
    std::filesystem::create_directories(absl::GetFlag(FLAGS_output_dir));

    // Initialize torch and loggers (console + file)
    libpts::model::init_torch(0);
    std::string export_suffix = absl::GetFlag(FLAGS_export_suffix);
    if (export_suffix != "") {
        export_suffix = absl::StrCat("_", export_suffix);
    }
    libpts::init_loggers(false, absl::GetFlag(FLAGS_output_dir), absl::StrCat("_test", export_suffix));

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
