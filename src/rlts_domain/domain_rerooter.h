// siirlts_rerooter.h
// Rerooter for Structure Induced Information for Rerooting LTS

#ifndef RLTSDomain_REROOTER_H_
#define RLTSDomain_REROOTER_H_

#include <libpolicyts/libpolicyts.h>

#include <absl/flags/flag.h>

#include <ostream>
#include <string>
#include <vector>

struct RLTSDomainRerooterWeightMetrics {
    int iter;
    std::string puzzle_name;
    int weight_idx;
    double w;
    double cw;

    static auto make_from_str(const std::string &str) -> RLTSDomainRerooterWeightMetrics;
    static void dump_header(std::ostream &os);
    friend auto operator<<(std::ostream &os, const RLTSDomainRerooterWeightMetrics &metrics_item) -> std::ostream &;
};

struct RLTSDomainRerooterSearchOutput {
    [[nodiscard]] auto to_metric_items(int bootstrap_iter, const std::string &puzzle_name) const
        -> std::vector<RLTSDomainRerooterWeightMetrics>;

    double cumulative_weight;
    std::vector<double> solution_path_w{};
    std::vector<double> solution_path_cw{};
};

template <libpts::algorithm::rlts::IsEnv EnvT>
class RLTSDomainRerooter {
public:
    RLTSDomainRerooter(bool _use_robust)
        : use_robust(_use_robust) {};

    void reset()
    {
        node_ids.clear();
        open_node_ids.clear();
        closed_node_ids.clear();
        node_id_to_w.clear();
        node_id_to_cw.clear();
        solution_path_w.clear();
        solution_path_cw.clear();
    }

    void init(const libpts::algorithm::rlts::Node<EnvT> &root_node)
    {
        node_ids.insert(root_node.id);
        open_node_ids.insert(root_node.id);
    }

    void expanded(
        const libpts::algorithm::rlts::Node<EnvT> &node,
        [[maybe_unused]] const libpts::algorithm::rlts::TreeProxyView<EnvT> &tree_view
    )
    {
        assert(open_node_ids.contains(node.id));
        open_node_ids.erase(node.id);
        closed_node_ids.insert(node.id);
    }

    void generated(
        [[maybe_unused]] const libpts::algorithm::rlts::Node<EnvT> &current_node,
        const libpts::algorithm::rlts::Node<EnvT> &child_node,
        [[maybe_unused]] const libpts::algorithm::rlts::TreeProxyView<EnvT> &tree_view
    )
    {
        node_ids.insert(child_node.id);
        open_node_ids.insert(child_node.id);
    }

    void visited(
        [[maybe_unused]] const libpts::algorithm::rlts::Node<EnvT> &current_node,
        [[maybe_unused]] const libpts::algorithm::rlts::Node<EnvT> &child_node,
        [[maybe_unused]] const libpts::algorithm::rlts::TreeProxyView<EnvT> &tree_view
    )
    {}

    auto operator()(
        const libpts::algorithm::rlts::Node<EnvT> &node,
        [[maybe_unused]] const libpts::algorithm::rlts::TreeProxyView<EnvT> &tree_view
    ) -> double
    {
        double w = 0;
        if constexpr (std::is_same_v<EnvT, libpts::env::BoulderDashState>) {
            constexpr auto events =
                libpts::env::BoulderDashEvent::kRewardCollectDiamond | libpts::env::BoulderDashEvent::kRewardCollectKey;
            w = node.state.query_events(events) ? 1 : 0;
        } else if (std::is_same_v<EnvT, libpts::env::CraftWorldState>) {
            constexpr auto events = std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftBronzeBar)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftPlank)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftRope)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftNails)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftBronzeHammer)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftBronzePick)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftIronPick)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftBridge)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftGoldBar)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCraftGemRing)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeUseAxe)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeUseBridge)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCollectTin)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCollectCopper)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCollectWood)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCollectIron)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCollectGold)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeCollectGem)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeUseAtWorkstation1)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeUseAtWorkstation2)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeUseAtWorkstation3)
                                    | std::to_underlying(libpts::env::CraftWorldEvent::kRewardCodeUseAtFurnace);
            w = node.state.query_events(events) ? 1 : 0;
        } else if (std::is_same_v<EnvT, libpts::env::SokobanState>) {
            w = node.state.query_events(libpts::env::SokobanEvent::kRewardBoxInGoal) ? 1 : 0;
        } else if (std::is_same_v<EnvT, libpts::env::TSPDeadlockState>) {
            w = node.state.query_events(libpts::env::TSPEvent::kCityVisited) ? 1 : 0;
        }

        // Weights (ensure root gets weight of 1)
        bool is_root = !node.parent;
        w = is_root ? 1.0 : w;
        cw += w;

        // Store
        node_id_to_w[node.id] = w;
        node_id_to_cw[node.id] = cw;

        // Robustness
        w = use_robust ? w / cw : w;
        return is_root ? 1.0 : w;
    }

    void batch_inferenced([[maybe_unused]] const libpts::algorithm::rlts::TreeProxyView<EnvT> &tree_view) {}

    auto get_search_output() const -> RLTSDomainRerooterSearchOutput
    {
        return {
            .cumulative_weight = cw,
            .solution_path_w = solution_path_w,
            .solution_path_cw = solution_path_cw,
        };
    }

    void solution_found(
        const libpts::algorithm::rlts::Node<EnvT> &node,
        [[maybe_unused]] const libpts::algorithm::rlts::TreeProxyView<EnvT> &tree_view
    )
    {
        auto current = &node;
        while (current->parent) {
            int node_id = current->parent->id;
            solution_path_w.push_back(node_id_to_w.at(node_id));
            solution_path_cw.push_back(node_id_to_cw.at(node_id));
            current = current->parent;
        }
        std::ranges::reverse(solution_path_w);
        std::ranges::reverse(solution_path_cw);
    }

private:
    bool use_robust;
    double cw{0};
    std::unordered_set<int> node_ids;
    std::unordered_set<int> open_node_ids;
    std::unordered_set<int> closed_node_ids;
    std::unordered_map<int, double> node_id_to_w;
    std::unordered_map<int, double> node_id_to_cw;
    std::vector<double> solution_path_w{};
    std::vector<double> solution_path_cw{};
};

#endif    // RLTSDomain_REROOTER_H_
