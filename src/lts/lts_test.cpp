#include <libpolicyts/libpolicyts.h>

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

// NOLINTBEGIN
ABSL_FLAG(std::string, environment, "", "String name of the environment");
ABSL_FLAG(std::string, problems_path, "", "Path to problems file");
ABSL_FLAG(std::string, output_dir, "/opt/pts/", "Base path to store all checkpoints and metrics");
ABSL_FLAG(std::string, model_path, "", "Path for the twoheaded convnet model wrapper.");
ABSL_FLAG(int, search_budget, INF_I, "Maximum number of expanded nodes before termination");
ABSL_FLAG(int, inference_batch_size, 32, "Number of search expansions to batch per inference query");
ABSL_FLAG(double, mix_epsilon, 0, "Percentage to mix with uniform policy");
ABSL_FLAG(int, max_iterations, INF_I, "Budget in number of iterations before terminating training/testing procedure");
ABSL_FLAG(double, time_budget, INF_D, "Budget in seconds before terminating");
ABSL_FLAG(int, num_threads, 1, "Number of threads to run in the search thread pool");
ABSL_FLAG(std::string, export_suffix, "", "Export suffix to place on output logs/files");
ABSL_FLAG(int, device_num, 0, "Torch cuda device number to use (defaults to 1)");
// NOLINTEND

namespace {

namespace lts = libpts::algorithm::lts;
using ModelT = libpts::model::PolicyConvNetWrapper;

template <typename EnvT>
auto create_search_inputs(
    const std::vector<EnvT> &problems,
    std::shared_ptr<libpts::StopToken> stop_token,
    std::shared_ptr<ModelT> model_wrapper
) {
    using SearchInputT = lts::SearchInput<EnvT, ModelT>;
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
    return search_inputs;
}

using json = nlohmann::json;

template <typename EnvT>
void templated_main() {
    using SearchInputT = lts::SearchInput<EnvT, ModelT>;
    using SearchOutputT = lts::SearchOutput<EnvT>;

    // Load problems
    auto [problems, _] = libpts::env::load_problems<EnvT>(absl::GetFlag(FLAGS_problems_path));
    if (problems.empty()) {
        SPDLOG_ERROR("No problems were loaded.");
        std::exit(1);
    }

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

    model_wrapper->load_checkpoint_without_optimizer(-1);
    model_wrapper->print();

    // Install signaller
    std::shared_ptr<libpts::StopToken> stop_token = libpts::signal_installer();

    // Create search inputs
    auto search_inputs = create_search_inputs(problems, stop_token, model_wrapper);

    libpts::test::test_runner<SearchInputT, SearchOutputT>(
        search_inputs,
        lts::search<EnvT, ModelT>,
        absl::GetFlag(FLAGS_output_dir),
        stop_token,
        absl::GetFlag(FLAGS_search_budget),
        absl::GetFlag(FLAGS_num_threads),
        absl::GetFlag(FLAGS_max_iterations),
        absl::GetFlag(FLAGS_time_budget),
        absl::GetFlag(FLAGS_export_suffix)
    );
}
}    // namespace

int main(int argc, char **argv) {
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
