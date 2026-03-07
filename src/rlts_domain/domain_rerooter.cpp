
// siirlts_rerooter.h
// Rerooter for Structure Induced Information for Rerooting LTS

#include "domain_rerooter.h"

#include <absl/strings/str_split.h>
#include <spdlog/spdlog.h>

#include <format>
#include <ostream>
#include <ranges>
#include <vector>

auto RLTSDomainRerooterWeightMetrics::make_from_str(const std::string &str) -> RLTSDomainRerooterWeightMetrics
{
    std::vector<std::string> strs = absl::StrSplit(str, ',');
    // NOLINTBEGIN (*-magic-numbers)
    if (strs.size() != 5) {
        const auto error_msg = std::format("Error reading line {:s}, {:d}", str, strs.size());
        spdlog::error(error_msg);
        throw std::runtime_error(error_msg);
    }
    return {
        .iter = std::stoi(strs[0]),
        .puzzle_name = strs[1],
        .weight_idx = std::stoi(strs[2]),
        .w = std::stod(strs[3]),
        .cw = std::stod(strs[4]),
    };
    // NOLINTEND
}
void RLTSDomainRerooterWeightMetrics::dump_header(std::ostream &os)
{
    os << "iter,";
    os << "puzzle_name,";
    os << "weight_idx,";
    os << "w,";
    os << "w_cumulative\n";
}

auto operator<<(std::ostream &os, const RLTSDomainRerooterWeightMetrics &metrics_item) -> std::ostream &
{
    os << metrics_item.iter << ",";
    os << metrics_item.puzzle_name << ",";
    os << metrics_item.weight_idx << ",";
    os << metrics_item.w << ",";
    os << metrics_item.cw << "\n";
    return os;
}

auto RLTSDomainRerooterSearchOutput::to_metric_items(int bootstrap_iter, const std::string &puzzle_name) const
    -> std::vector<RLTSDomainRerooterWeightMetrics>
{
    std::vector<RLTSDomainRerooterWeightMetrics> metrics;
    for (auto &&[i, w, cw] : std::views::zip(std::views::iota(0), solution_path_w, solution_path_cw)) {
        metrics.push_back({
            .iter = bootstrap_iter,
            .puzzle_name = puzzle_name,
            .weight_idx = i,
            .w = w,
            .cw = cw,
        });
    }
    return metrics;
}
