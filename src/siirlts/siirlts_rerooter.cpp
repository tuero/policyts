
// siirlts_rerooter.h
// Rerooter for Structure Induced Information for Rerooting LTS

#include "siirlts_rerooter.h"

#include <absl/strings/str_split.h>
#include <spdlog/spdlog.h>

#include <format>
#include <ostream>
#include <ranges>
#include <vector>

auto AbslParseFlag(absl::string_view text, ClusterLevel *cluster_level, std::string *error) -> bool
{
    if (text == "min") {
        *cluster_level = ClusterLevel::Min;
        return true;
    }
    if (text == "half") {
        *cluster_level = ClusterLevel::Half;
        return true;
    }
    if (text == "max") {
        *cluster_level = ClusterLevel::Max;
        return true;
    }
    *error = "unknown value for enumeration";
    return false;
}
auto AbslUnparseFlag(ClusterLevel cluster_level) -> std::string
{
    switch (cluster_level) {
    case ClusterLevel::Min:
        return "min";
    case ClusterLevel::Half:
        return "half";
    case ClusterLevel::Max:
        return "max";
    }
    return absl::StrCat(cluster_level);
}

auto AbslParseFlag(absl::string_view text, WeightMode *weight_mode, std::string *error) -> bool
{
    if (text == "zeros") {
        *weight_mode = WeightMode::Zeros;
        return true;
    }
    if (text == "closed_count") {
        *weight_mode = WeightMode::ClosedCount;
        return true;
    }
    if (text == "closed_count_increment") {
        *weight_mode = WeightMode::ClosedCountIncrement;
        return true;
    }
    if (text == "open_count") {
        *weight_mode = WeightMode::OpenCount;
        return true;
    }
    if (text == "open_count_increment") {
        *weight_mode = WeightMode::OpenCountIncrement;
        return true;
    }
    if (text == "all_count") {
        *weight_mode = WeightMode::AllCount;
        return true;
    }
    if (text == "all_count_parent") {
        *weight_mode = WeightMode::AllCountParent;
        return true;
    }
    if (text == "all_count_increment") {
        *weight_mode = WeightMode::AllCountIncrement;
        return true;
    }
    if (text == "all_count_parent_increment") {
        *weight_mode = WeightMode::AllCountParentIncrement;
        return true;
    }
    if (text == "open_count_parent_increment") {
        *weight_mode = WeightMode::OpenCountParentIncrement;
        return true;
    }
    if (text == "closed_count_parent_increment") {
        *weight_mode = WeightMode::ClosedCountParentIncrement;
        return true;
    }
    *error = "unknown value for enumeration";
    return false;
}

auto AbslUnparseFlag(WeightMode color_match_method) -> std::string
{
    switch (color_match_method) {
    case WeightMode::Zeros:
        return "zeros";
    case WeightMode::ClosedCount:
        return "closed_count";
    case WeightMode::ClosedCountIncrement:
        return "closed_count_increment";
    case WeightMode::OpenCount:
        return "open_count";
    case WeightMode::OpenCountIncrement:
        return "open_count_increment";
    case WeightMode::AllCount:
        return "all_count";
    case WeightMode::AllCountParent:
        return "all_count_parent";
    case WeightMode::AllCountIncrement:
        return "all_count_increment";
    case WeightMode::AllCountParentIncrement:
        return "all_count_parent_increment";
    case WeightMode::OpenCountParentIncrement:
        return "open_count_parent_increment";
    case WeightMode::ClosedCountParentIncrement:
        return "closed_count_parent_increment";
    }
    return absl::StrCat(color_match_method);
}

auto AbslParseFlag(absl::string_view text, RobustMode *mode, std::string *error) -> bool
{
    if (text == "none") {
        *mode = RobustMode::None;
        return true;
    }
    if (text == "cluster") {
        *mode = RobustMode::Cluster;
        return true;
    }
    if (text == "heuristic") {
        *mode = RobustMode::Heuristic;
        return true;
    }
    if (text == "both") {
        *mode = RobustMode::Both;
        return true;
    }
    *error = "unknown value for enumeration";
    return false;
}

auto AbslUnparseFlag(RobustMode mode) -> std::string
{
    switch (mode) {
    case RobustMode::None:
        return "none";
    case RobustMode::Cluster:
        return "cluster";
    case RobustMode::Heuristic:
        return "heuristic";
    case RobustMode::Both:
        return "both";
    }
    return absl::StrCat(mode);
}

auto SIIRLTSRerooterWeightMetrics::make_from_str(const std::string &str) -> SIIRLTSRerooterWeightMetrics
{
    std::vector<std::string> strs = absl::StrSplit(str, ',');
    // NOLINTBEGIN (*-magic-numbers)
    if (strs.size() != 9) {
        const auto error_msg = std::format("Error reading line {:s}, {:d}", str, strs.size());
        spdlog::error(error_msg);
        throw std::runtime_error(error_msg);
    }
    return {
        .iter = std::stoi(strs[0]),
        .puzzle_name = strs[1],
        .weight_idx = std::stoi(strs[2]),
        .w = std::stod(strs[3]),
        .w_cluster = std::stod(strs[4]),
        .w_heuristic = std::stod(strs[5]),
        .cw = std::stod(strs[6]),
        .cw_cluster = std::stod(strs[7]),
        .cw_heuristic = std::stod(strs[8]),
    };
    // NOLINTEND
}
void SIIRLTSRerooterWeightMetrics::dump_header(std::ostream &os)
{
    os << "iter,";
    os << "puzzle_name,";
    os << "weight_idx,";
    os << "w,";
    os << "wa,";
    os << "wb,";
    os << "w_cumulative,";
    os << "wa_cumulative,";
    os << "wb_cumulative\n";
}

auto operator<<(std::ostream &os, const SIIRLTSRerooterWeightMetrics &metrics_item) -> std::ostream &
{
    os << metrics_item.iter << ",";
    os << metrics_item.puzzle_name << ",";
    os << metrics_item.weight_idx << ",";
    os << metrics_item.w << ",";
    os << metrics_item.w_cluster << ",";
    os << metrics_item.w_heuristic << ",";
    os << metrics_item.cw << ",";
    os << metrics_item.cw_cluster << ",";
    os << metrics_item.cw_heuristic << "\n";
    return os;
}

auto SIIRLTSRerooterSearchOutput::to_metric_items(int bootstrap_iter, const std::string &puzzle_name) const
    -> std::vector<SIIRLTSRerooterWeightMetrics>
{
    std::vector<SIIRLTSRerooterWeightMetrics> metrics;
    for (auto &&[i, w, wa, wb, cw, cwa, cwb] : std::views::zip(
             std::views::iota(0),
             solution_path_w,
             solution_path_w_cluster,
             solution_path_w_heuristic,
             solution_path_cw,
             solution_path_cw_cluster,
             solution_path_cw_heuristic
         ))
    {
        metrics.push_back({
            .iter = bootstrap_iter,
            .puzzle_name = puzzle_name,
            .weight_idx = i,
            .w = w,
            .w_cluster = wa,
            .w_heuristic = wb,
            .cw = cw,
            .cw_cluster = cwa,
            .cw_heuristic = cwb,
        });
    }
    return metrics;
}
