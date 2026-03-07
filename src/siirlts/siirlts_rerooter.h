// siirlts_rerooter.h
// Rerooter for Structure Induced Information for Rerooting LTS

#ifndef SIIRLTS_REROOTER_H_
#define SIIRLTS_REROOTER_H_

#include <libpolicyts/libpolicyts.h>

#include <absl/flags/flag.h>

#include <ostream>
#include <string>
#include <vector>

enum class WeightMode {
    Zeros,
    ClosedCount,
    ClosedCountIncrement,
    OpenCount,
    OpenCountIncrement,
    AllCount,
    AllCountParent,
    AllCountIncrement,
    OpenCountParentIncrement,
    ClosedCountParentIncrement,
    AllCountParentIncrement,
};
auto AbslParseFlag(absl::string_view text, WeightMode *weight_mode, std::string *error) -> bool;
auto AbslUnparseFlag(WeightMode color_match_method) -> std::string;

enum class ClusterLevel {
    Min,
    Half,
    Max,
};
auto AbslParseFlag(absl::string_view text, ClusterLevel *cluster_level, std::string *error) -> bool;
auto AbslUnparseFlag(ClusterLevel cluster_level) -> std::string;

enum class RobustMode {
    None,
    Cluster,
    Heuristic,
    Both,
};
auto AbslParseFlag(absl::string_view text, RobustMode *mode, std::string *error) -> bool;
auto AbslUnparseFlag(RobustMode mode) -> std::string;

struct SIIRLTSRerooterWeightMetrics {
    int iter;
    std::string puzzle_name;
    int weight_idx;
    double w;
    double w_cluster;
    double w_heuristic;
    double cw;
    double cw_cluster;
    double cw_heuristic;

    static auto make_from_str(const std::string &str) -> SIIRLTSRerooterWeightMetrics;
    static void dump_header(std::ostream &os);
    friend auto operator<<(std::ostream &os, const SIIRLTSRerooterWeightMetrics &metrics_item) -> std::ostream &;
};

struct SIIRLTSRerooterSearchOutput {
    [[nodiscard]] auto to_metric_items(int bootstrap_iter, const std::string &puzzle_name) const
        -> std::vector<SIIRLTSRerooterWeightMetrics>;

    double cw_cluster;
    double cw_heuristic;
    std::vector<double> solution_path_w{};
    std::vector<double> solution_path_w_cluster{};
    std::vector<double> solution_path_w_heuristic{};
    std::vector<double> solution_path_cw{};
    std::vector<double> solution_path_cw_cluster{};
    std::vector<double> solution_path_cw_heuristic{};
};

template <libpts::algorithm::rlts::IsEnv EnvT>
class SIIRLTSRerooter {
public:
    SIIRLTSRerooter(
        RobustMode _robust_mode,
        ClusterLevel _cluster_level,
        WeightMode _weight_mode,
        double _graph_update_factor,
        double _alpha,
        double _beta,
        int _seed
    )
        : robust_mode(_robust_mode),
          cluster_level(_cluster_level),
          weight_mode(_weight_mode),
          graph_update_factor(_graph_update_factor),
          alpha(_alpha),
          beta(_beta),
          seed(_seed)
    {}

    void reset()
    {
        cw_cluster = 0;
        cw_heuristic = 0;
        batch_inference_counter = -1;
        next_graph_update = 1;
        id_edges.clear();
        node_ids.clear();
        open_node_ids.clear();
        closed_node_ids.clear();
        node_id_to_cluster.clear();
        node_id_to_w.clear();
        node_id_to_w_cluster.clear();
        node_id_to_w_heuristic.clear();
        node_id_to_cw.clear();
        node_id_to_cw_cluster.clear();
        node_id_to_cw_heuristic.clear();
        cluster_id_expansion_count_map.clear();
        solution_path_w.clear();
        solution_path_w_cluster.clear();
        solution_path_w_heuristic.clear();
        solution_path_cw.clear();
        solution_path_cw_cluster.clear();
        solution_path_cw_heuristic.clear();
    }

    void init(const libpts::algorithm::rlts::Node<EnvT> &root_node)
    {
        node_ids.insert(root_node.id);
        open_node_ids.insert(root_node.id);
        base_h = std::max(root_node.h, 1.0);
    }

    void expanded(const libpts::algorithm::rlts::Node<EnvT> &node)
    {
        assert(open_node_ids.contains(node.id));
        open_node_ids.erase(node.id);
        closed_node_ids.insert(node.id);
    }

    void generated(
        const libpts::algorithm::rlts::Node<EnvT> &current_node,
        const libpts::algorithm::rlts::Node<EnvT> &child_node
    )
    {
        node_ids.insert(child_node.id);
        open_node_ids.insert(child_node.id);
        // Link underlying graph
        id_edges.emplace_back(current_node.id, child_node.id);
    }

    void prev_generated(
        const libpts::algorithm::rlts::Node<EnvT> &current_node,
        const libpts::algorithm::rlts::Node<EnvT> &prev_generated_node
    )
    {
        // If we did previously generate, we have another path through the node,
        // and thus we need to link underlying graph
        id_edges.emplace_back(current_node.id, prev_generated_node.id);
    }

    auto operator()(const libpts::algorithm::rlts::Node<EnvT> &node) -> double
    {
        auto compute_clustering_weight = [&]() -> double {
            bool has_cluster = node_id_to_cluster.contains(node.id);
            auto cluster_id = has_cluster ? node_id_to_cluster[node.id] : -1;
            double parent_w_cluster = node_id_to_w_cluster[node.parent->id];

            switch (weight_mode) {
            // All non-root nodes get weight of zero
            case WeightMode::Zeros:
                return 0;
            // 1/count if has cluster id, 0 otherwise
            case WeightMode::ClosedCount:
            case WeightMode::OpenCount:
            case WeightMode::AllCount:
                return !has_cluster ? 0 : 1.0 / cluster_id_expansion_count_map[cluster_id];
            // Incremented 1/count if has cluster id, otherwise 0
            case WeightMode::ClosedCountIncrement:
            case WeightMode::OpenCountIncrement:
            case WeightMode::AllCountIncrement:
                return !has_cluster ? 0 : 1.0 / (++cluster_id_expansion_count_map[cluster_id]);
            // Incremented 1/count if has cluster id, otherwise incremented parents weight
            case WeightMode::AllCountParentIncrement:
            case WeightMode::OpenCountParentIncrement:
            case WeightMode::ClosedCountParentIncrement:
                // Optimization trick
                return !has_cluster ? parent_w_cluster / (parent_w_cluster + 1)
                                    : 1.0 / (++cluster_id_expansion_count_map[cluster_id]);
            case WeightMode::AllCountParent:
                return !has_cluster ? parent_w_cluster : 1.0 / cluster_id_expansion_count_map[cluster_id];
            }
            std::unreachable();
        };

        // Weights (ensure root gets weight of 1)
        bool is_root = !node.parent;
        double wa = is_root ? 1.0 : alpha * compute_clustering_weight();
        double wb = is_root ? 1.0 : beta * std::clamp(1.0 - node.h / base_h, 0.0, 1.0);
        cw_cluster += wa;
        cw_heuristic += wb;

        // Store
        node_id_to_w[node.id] = is_root ? 1.0 : wa + wb;
        node_id_to_w_cluster[node.id] = wa;
        node_id_to_w_heuristic[node.id] = wb;
        node_id_to_cw[node.id] = is_root ? 1.0 : cw_cluster + cw_heuristic;
        node_id_to_cw_cluster[node.id] = cw_cluster;
        node_id_to_cw_heuristic[node.id] = cw_heuristic;

        // Robustness on individual components
        double w = [&]() {
            switch (robust_mode) {
            case RobustMode::None:
                return wa + wb;
            case RobustMode::Cluster:
                return wa / cw_cluster + wb;
            case RobustMode::Heuristic:
                return wa + wb / cw_heuristic;
            case RobustMode::Both:
                return wa / cw_cluster + wb / cw_heuristic;
            }
            std::unreachable();
        }();

        return is_root ? 1.0 : w;
    }

    void batch_inferenced()
    {
        ++batch_inference_counter;
        // Check if new graph needs to be made
        if (batch_inference_counter >= next_graph_update) {
            update_cluster_data();
            next_graph_update =
                static_cast<int>(std::ceil(graph_update_factor * static_cast<double>(next_graph_update)));
        }
    }

    auto get_search_output() const -> SIIRLTSRerooterSearchOutput
    {
        return {
            .cw_cluster = cw_cluster,
            .cw_heuristic = cw_heuristic,
            .solution_path_w = solution_path_w,
            .solution_path_w_cluster = solution_path_w_cluster,
            .solution_path_w_heuristic = solution_path_w_heuristic,
            .solution_path_cw = solution_path_cw,
            .solution_path_cw_cluster = solution_path_cw_cluster,
            .solution_path_cw_heuristic = solution_path_cw_heuristic,
        };
    }

    void solution_found(
        const libpts::algorithm::rlts::Node<EnvT> &node,
        [[maybe_unused]] const libpts::algorithm::rlts::NodeSet<EnvT> &tree_nodes
    )
    {
        auto current = &node;
        while (current->parent) {
            int node_id = current->parent->id;
            solution_path_w.push_back(node_id_to_w.at(node_id));
            solution_path_w_cluster.push_back(node_id_to_w_cluster.at(node_id));
            solution_path_w_heuristic.push_back(node_id_to_w_heuristic.at(node_id));
            solution_path_cw.push_back(node_id_to_cw.at(node_id));
            solution_path_cw_cluster.push_back(node_id_to_cw_cluster.at(node_id));
            solution_path_cw_heuristic.push_back(node_id_to_cw_heuristic.at(node_id));
            current = current->parent;
        }
        std::ranges::reverse(solution_path_w);
        std::ranges::reverse(solution_path_w_cluster);
        std::ranges::reverse(solution_path_w_heuristic);
        std::ranges::reverse(solution_path_cw);
        std::ranges::reverse(solution_path_cw_cluster);
        std::ranges::reverse(solution_path_cw_heuristic);
    }

private:
    auto make_graph() -> libpts::clustering::ClusterGraphs
    {
        int NUM_VERTICES = static_cast<int>(node_ids.size());
        std::vector<int> edges;
        edges.reserve(id_edges.size() * 2);
        for (const auto &[id_from, id_to] : id_edges) {
            edges.push_back(id_from);
            edges.push_back(id_to);
        }
        return {NUM_VERTICES, edges, static_cast<std::size_t>(seed)};
    }

    int get_cluster_level(const libpts::clustering::ClusterGraphs &graph_wrapper)
    {
        switch (cluster_level) {
        case ClusterLevel::Min:
            return 0;
        case ClusterLevel::Half:
            return static_cast<int>(graph_wrapper.hierarchy_size() / 2);
        case ClusterLevel::Max:
            return graph_wrapper.hierarchy_size() - 1;
        }
        std::unreachable();
    }

    void update_cluster_data()
    {
        // New graph clustering
        const libpts::clustering::ClusterGraphs graph_wrapper = make_graph();
        const auto c_level = get_cluster_level(graph_wrapper);

        // Set node cluster color IDs
        node_id_to_cluster.clear();
        for (auto &node_id : node_ids) {
            node_id_to_cluster[node_id] = static_cast<int>(graph_wrapper.get_cluster_id(node_id, c_level));
        }

        // Weight mode specific updates
        auto count_node_colors = [&](const std::unordered_set<int> &_node_ids) {
            cluster_id_expansion_count_map.clear();
            for (const auto &node_id : _node_ids) {
                assert(node_id_to_cluster.contains(node_id));
                ++cluster_id_expansion_count_map[node_id_to_cluster[node_id]];
            }
        };
        switch (weight_mode) {
        case WeightMode::Zeros:
            break;
        case WeightMode::ClosedCount:
        case WeightMode::ClosedCountIncrement:
        case WeightMode::ClosedCountParentIncrement:
            count_node_colors(closed_node_ids);
            break;
        case WeightMode::OpenCount:
        case WeightMode::OpenCountIncrement:
        case WeightMode::OpenCountParentIncrement:
            count_node_colors(open_node_ids);
            break;
        case WeightMode::AllCount:
        case WeightMode::AllCountIncrement:
        case WeightMode::AllCountParentIncrement:
        case WeightMode::AllCountParent:
            count_node_colors(node_ids);
            break;
        }
    }

    // Args during construction
    RobustMode robust_mode;
    ClusterLevel cluster_level;
    WeightMode weight_mode;
    double graph_update_factor;
    double alpha;
    double beta;
    int seed;
    // Args during construction
    double cw_cluster = 0;
    double cw_heuristic = 0;
    double base_h = 1;
    int batch_inference_counter = -1;    // How many times we've done inference
    int next_graph_update = 1;           // Step when the next graph update to be done
    std::unordered_set<int> node_ids;
    std::unordered_set<int> open_node_ids;
    std::unordered_set<int> closed_node_ids;
    std::unordered_map<int, int> node_id_to_cluster;
    std::unordered_map<int, double> node_id_to_w;
    std::unordered_map<int, double> node_id_to_w_cluster;
    std::unordered_map<int, double> node_id_to_w_heuristic;
    std::unordered_map<int, double> node_id_to_cw;
    std::unordered_map<int, double> node_id_to_cw_cluster;
    std::unordered_map<int, double> node_id_to_cw_heuristic;
    std::vector<std::tuple<int, int>> id_edges;                     // par-child graph edges with actions
    std::unordered_map<int, int> cluster_id_expansion_count_map;    // M in algorithm
    std::vector<double> solution_path_w{};
    std::vector<double> solution_path_w_cluster{};
    std::vector<double> solution_path_w_heuristic{};
    std::vector<double> solution_path_cw{};
    std::vector<double> solution_path_cw_cluster{};
    std::vector<double> solution_path_cw_heuristic{};
};

#endif    // SIIRLTS_REROOTER_H_
