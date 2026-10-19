#include <vector>
#include <queue>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

using std::vector;
using std::unordered_map;

namespace ParallelClusteredBlockGorder {

static constexpr unsigned INVALID_ID =
    std::numeric_limits<unsigned>::max();

struct GraphInfo {


    vector<vector<uint64_t>> reverse_graph;

    vector<unsigned> in_degree;
    vector<unsigned> out_degree;
    vector<unsigned> degree;
};

struct LocalScore {
    double cn = 0.0;
    double mnn = 0.0;
    double rrf = 0.0;
};

inline uint64_t packReverseEntry(unsigned src, unsigned rank) {
    return (static_cast<uint64_t>(rank) << 32) |
           static_cast<uint64_t>(src);
}

inline unsigned unpackSrc(uint64_t x) {
    return static_cast<unsigned>(x & 0xffffffffu);
}

inline unsigned unpackRank(uint64_t x) {
    return static_cast<unsigned>(x >> 32);
}


GraphInfo preprocessGraphSparse(
    const vector<vector<unsigned>>& graph,
    unsigned max_reverse_entries_per_node = 0
) {
    const unsigned n = static_cast<unsigned>(graph.size());

    GraphInfo info;
    info.reverse_graph.resize(n);
    info.in_degree.assign(n, 0);
    info.out_degree.assign(n, 0);
    info.degree.assign(n, 0);


    for (unsigned u = 0; u < n; ++u) {
        unsigned valid_out = 0;

        for (unsigned v : graph[u]) {
            if (v >= n) {
                throw std::runtime_error("graph contains invalid node id.");
            }

            if (v == u) {
                continue;
            }

            ++info.in_degree[v];
            ++valid_out;
        }

        info.out_degree[u] = valid_out;
    }


    for (unsigned v = 0; v < n; ++v) {
        unsigned reserve_size = info.in_degree[v];

        if (max_reverse_entries_per_node > 0) {
            reserve_size = std::min(
                reserve_size,
                max_reverse_entries_per_node
            );
        }

        info.reverse_graph[v].reserve(reserve_size);
    }


    for (unsigned u = 0; u < n; ++u) {
        unsigned rank = 1;

        for (unsigned v : graph[u]) {
            if (v == u) {
                continue;
            }

            info.reverse_graph[v].push_back(
                packReverseEntry(u, rank)
            );

            ++rank;
        }
    }

    auto rev_cmp = [](uint64_t a, uint64_t b) {
        unsigned ra = unpackRank(a);
        unsigned rb = unpackRank(b);

        if (ra != rb) {
            return ra < rb;
        }

        return unpackSrc(a) < unpackSrc(b);
    };


    for (unsigned v = 0; v < n; ++v) {
        auto& rev = info.reverse_graph[v];

        if (max_reverse_entries_per_node > 0 &&
            rev.size() > max_reverse_entries_per_node) {

            std::nth_element(
                rev.begin(),
                rev.begin() + max_reverse_entries_per_node,
                rev.end(),
                rev_cmp
            );

            rev.resize(max_reverse_entries_per_node);
        }

        std::sort(rev.begin(), rev.end(), rev_cmp);
    }

    for (unsigned v = 0; v < n; ++v) {


        info.degree[v] = info.in_degree[v] + info.out_degree[v];
    }

    return info;
}


inline bool isCandidateInCurrentCluster(
    unsigned v,
    unsigned current_cluster,
    const vector<unsigned>& cluster_id,
    const vector<unsigned>& local_pos,
    const vector<char>& local_unvisited
) {
    if (cluster_id[v] != current_cluster) {
        return false;
    }

    unsigned pos = local_pos[v];

    if (pos == INVALID_ID) {
        return false;
    }

    return local_unvisited[pos] != 0;
}


vector<std::pair<unsigned, double>> computeDirectedRRFForNodeInCluster(
    unsigned u,
    unsigned current_cluster,
    const vector<vector<unsigned>>& graph,
    const GraphInfo& info,
    const vector<unsigned>& cluster_id,
    const vector<unsigned>& local_pos,
    const vector<char>& local_unvisited,
    double rrf_c,
    bool use_cn,
    bool use_mnn
) {
    unordered_map<unsigned, LocalScore> cand;


    size_t reserve_size = graph[u].size();

    if (use_cn) {
        for (unsigned z : graph[u]) {
            if (z < info.reverse_graph.size()) {
                reserve_size += info.reverse_graph[z].size();
            }
        }
    }

    if (use_mnn) {
        reserve_size += info.reverse_graph[u].size();
    }

    const size_t max_reserve = static_cast<size_t>(1) << 20;
    cand.reserve(std::min(reserve_size, max_reserve));


    if (use_cn) {
        unsigned rank_u_z = 1;

        for (unsigned z : graph[u]) {
            if (z == u) {
                ++rank_u_z;
                continue;
            }

            const auto& rev_z = info.reverse_graph[z];

            for (uint64_t packed : rev_z) {
                unsigned v = unpackSrc(packed);
                unsigned rank_v_z = unpackRank(packed);

                if (v == u) {
                    continue;
                }

                if (!isCandidateInCurrentCluster(
                        v,
                        current_cluster,
                        cluster_id,
                        local_pos,
                        local_unvisited
                    )) {
                    continue;
                }

                double add =
                    static_cast<double>(info.degree[z]) /
                    std::sqrt(
                        static_cast<double>(rank_u_z) *
                        static_cast<double>(rank_v_z)
                    );


                cand[v].cn += add;
            }

            ++rank_u_z;
        }
    }


    if (use_mnn) {
        unsigned rank_u_v = 1;

        for (unsigned v : graph[u]) {
            if (v == u) {
                ++rank_u_v;
                continue;
            }

            if (isCandidateInCurrentCluster(
                    v,
                    current_cluster,
                    cluster_id,
                    local_pos,
                    local_unvisited
                )) {
                cand[v].mnn += 1.0 / static_cast<double>(rank_u_v);
            }

            ++rank_u_v;
        }


        const auto& rev_u = info.reverse_graph[u];

        for (uint64_t packed : rev_u) {
            unsigned v = unpackSrc(packed);
            unsigned rank_v_u = unpackRank(packed);

            if (v == u) {
                continue;
            }

            if (!isCandidateInCurrentCluster(
                    v,
                    current_cluster,
                    cluster_id,
                    local_pos,
                    local_unvisited
                )) {
                continue;
            }

            cand[v].mnn += 1.0 / static_cast<double>(rank_v_u);
        }
    }

    vector<std::pair<double, unsigned>> cn_order;
    vector<std::pair<double, unsigned>> mnn_order;

    if (use_cn) {
        cn_order.reserve(cand.size());
    }

    if (use_mnn) {
        mnn_order.reserve(cand.size());
    }

    for (const auto& kv : cand) {
        unsigned v = kv.first;
        const LocalScore& s = kv.second;

        if (v == u) {
            continue;
        }

        if (!isCandidateInCurrentCluster(
                v,
                current_cluster,
                cluster_id,
                local_pos,
                local_unvisited
            )) {
            continue;
        }

        if (use_cn && s.cn > 0.0) {
            cn_order.emplace_back(s.cn, v);
        }

        if (use_mnn && s.mnn > 0.0) {
            mnn_order.emplace_back(s.mnn, v);
        }
    }

    auto cmp = [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }

        return a.second < b.second;
    };

    if (use_cn) {
        std::sort(cn_order.begin(), cn_order.end(), cmp);

        for (unsigned rank = 1; rank <= cn_order.size(); ++rank) {
            unsigned v = cn_order[rank - 1].second;
            cand[v].rrf += 1.0 / (rrf_c + static_cast<double>(rank));
        }
    }

    if (use_mnn) {
        std::sort(mnn_order.begin(), mnn_order.end(), cmp);

        for (unsigned rank = 1; rank <= mnn_order.size(); ++rank) {
            unsigned v = mnn_order[rank - 1].second;
            cand[v].rrf += 1.0 / (rrf_c + static_cast<double>(rank));
        }
    }

    vector<std::pair<unsigned, double>> result;
    result.reserve(cand.size());

    for (const auto& kv : cand) {
        unsigned v = kv.first;
        const LocalScore& s = kv.second;

        if (s.rrf <= 0.0) {
            continue;
        }

        if (!isCandidateInCurrentCluster(
                v,
                current_cluster,
                cluster_id,
                local_pos,
                local_unvisited
            )) {
            continue;
        }

        result.emplace_back(v, s.rrf);
    }

    return result;
}


unsigned getNextSeedInCluster(
    const vector<unsigned>& nodes,
    size_t& seed_ptr,
    const vector<unsigned>& local_pos,
    const vector<char>& local_unvisited
) {
    while (seed_ptr < nodes.size()) {
        unsigned v = nodes[seed_ptr];
        ++seed_ptr;

        unsigned pos = local_pos[v];

        if (pos != INVALID_ID && local_unvisited[pos]) {
            return v;
        }
    }

    return INVALID_ID;
}


vector<unsigned> reorderOneCluster(
    unsigned current_cluster,
    const vector<unsigned>& nodes,
    const vector<vector<unsigned>>& graph,
    const GraphInfo& info,
    const vector<unsigned>& cluster_id,
    const vector<unsigned>& local_pos,
    unsigned vectors_per_block,
    double rrf_c,
    bool use_cn,
    bool use_mnn
) {
    vector<unsigned> local_order;
    local_order.reserve(nodes.size());

    if (nodes.empty()) {
        return local_order;
    }

    vector<char> local_unvisited(nodes.size(), 1);

    unsigned cluster_remaining =
        static_cast<unsigned>(nodes.size());

    size_t seed_ptr = 0;

    while (cluster_remaining > 0) {
        unsigned seed = getNextSeedInCluster(
            nodes,
            seed_ptr,
            local_pos,
            local_unvisited
        );

        if (seed == INVALID_ID) {
            break;
        }

        vector<unsigned> current_block;
        current_block.reserve(vectors_per_block);

        unordered_map<unsigned, double> block_candidate_score;
        block_candidate_score.reserve(vectors_per_block * 1024);

        auto addNodeToCurrentBlock = [&](unsigned u) {
            unsigned pos = local_pos[u];

            if (pos == INVALID_ID || !local_unvisited[pos]) {
                return;
            }

            local_unvisited[pos] = 0;
            --cluster_remaining;

            current_block.push_back(u);
            local_order.push_back(u);

            auto local_rrf = computeDirectedRRFForNodeInCluster(
                u,
                current_cluster,
                graph,
                info,
                cluster_id,
                local_pos,
                local_unvisited,
                rrf_c,
                use_cn,
                use_mnn
            );

            for (const auto& p : local_rrf) {
                unsigned v = p.first;
                double score = p.second;

                if (isCandidateInCurrentCluster(
                        v,
                        current_cluster,
                        cluster_id,
                        local_pos,
                        local_unvisited
                    )) {
                    block_candidate_score[v] += score;
                }
            }
        };

        addNodeToCurrentBlock(seed);

        while (current_block.size() < vectors_per_block &&
               cluster_remaining > 0) {

            unsigned best_v = INVALID_ID;
            double best_score = -1.0;

            for (const auto& kv : block_candidate_score) {
                unsigned v = kv.first;
                double score = kv.second;

                if (!isCandidateInCurrentCluster(
                        v,
                        current_cluster,
                        cluster_id,
                        local_pos,
                        local_unvisited
                    )) {
                    continue;
                }

                if (best_v == INVALID_ID ||
                    score > best_score ||
                    (score == best_score &&
                     (info.in_degree[v] > info.in_degree[best_v] ||
                      (info.in_degree[v] == info.in_degree[best_v] &&
                       v < best_v)))) {
                    best_v = v;
                    best_score = score;
                }
            }


            if (best_v == INVALID_ID) {
                best_v = getNextSeedInCluster(
                    nodes,
                    seed_ptr,
                    local_pos,
                    local_unvisited
                );

                if (best_v == INVALID_ID) {
                    break;
                }
            }

            addNodeToCurrentBlock(best_v);
        }
    }

    if (local_order.size() != nodes.size()) {
        throw std::runtime_error(
            "reorderOneCluster failed: local_order size mismatch."
        );
    }

    return local_order;
}


vector<unsigned> buildClusteredReorderMapParallel(
    const vector<vector<unsigned>>& graph,
    const vector<unsigned>& cluster_id,
    unsigned num_clusters,
    unsigned vectors_per_block,
    double rrf_c = 60.0,
    unsigned max_reverse_entries_per_node = 0,
    bool use_cn = true,
    bool use_mnn = true,
    int num_threads = 0
) {
    const unsigned n = static_cast<unsigned>(graph.size());

    if (cluster_id.size() != n) {
        throw std::runtime_error("cluster_id size must equal graph size.");
    }

    if (num_clusters == 0) {
        throw std::runtime_error("num_clusters must be positive.");
    }

    if (vectors_per_block == 0) {
        throw std::runtime_error("vectors_per_block must be positive.");
    }

    if (!use_cn && !use_mnn) {
        throw std::runtime_error(
            "at least one of use_cn or use_mnn must be true."
        );
    }

    if (n == 0) {
        return {};
    }

    for (unsigned i = 0; i < n; ++i) {
        if (cluster_id[i] >= num_clusters) {
            throw std::runtime_error(
                "cluster_id contains invalid cluster index."
            );
        }
    }


    GraphInfo info =
        preprocessGraphSparse(graph, max_reverse_entries_per_node);


    vector<vector<unsigned>> cluster_nodes(num_clusters);

    for (unsigned i = 0; i < n; ++i) {
        cluster_nodes[cluster_id[i]].push_back(i);
    }


    for (unsigned c = 0; c < num_clusters; ++c) {
        auto& nodes = cluster_nodes[c];

        std::sort(
            nodes.begin(),
            nodes.end(),
            [&](unsigned a, unsigned b) {
                if (info.in_degree[a] != info.in_degree[b]) {
                    return info.in_degree[a] > info.in_degree[b];
                }

                return a < b;
            }
        );
    }


    vector<unsigned> local_pos(n, INVALID_ID);

    for (unsigned c = 0; c < num_clusters; ++c) {
        const auto& nodes = cluster_nodes[c];

        for (unsigned pos = 0; pos < nodes.size(); ++pos) {
            unsigned old_id = nodes[pos];
            local_pos[old_id] = pos;
        }
    }


    vector<vector<unsigned>> cluster_orders(num_clusters);

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

#pragma omp parallel for schedule(dynamic)
#endif
    for (long long c_ll = 0;
         c_ll < static_cast<long long>(num_clusters);
         ++c_ll) {

        unsigned c = static_cast<unsigned>(c_ll);

        cluster_orders[c] = reorderOneCluster(
            c,
            cluster_nodes[c],
            graph,
            info,
            cluster_id,
            local_pos,
            vectors_per_block,
            rrf_c,
            use_cn,
            use_mnn
        );
    }


    vector<unsigned> order;
    order.reserve(n);

    for (unsigned c = 0; c < num_clusters; ++c) {
        const auto& local_order = cluster_orders[c];
        const size_t full_size =
            local_order.size() / vectors_per_block * vectors_per_block;
        order.insert(order.end(), local_order.begin(), local_order.begin() + full_size);
    }

    for (unsigned c = 0; c < num_clusters; ++c) {
        const auto& local_order = cluster_orders[c];
        const size_t full_size =
            local_order.size() / vectors_per_block * vectors_per_block;
        order.insert(order.end(), local_order.begin() + full_size, local_order.end());
    }

    if (order.size() != n) {
        throw std::runtime_error(
            "global reorder failed: order size != graph size."
        );
    }

    vector<unsigned> old_to_new(n);

    for (unsigned new_id = 0; new_id < n; ++new_id) {
        unsigned old_id = order[new_id];
        old_to_new[old_id] = new_id;
    }

    return old_to_new;
}

}

void Load(const char *filename, vector<vector<unsigned>>& final_graph_) {
  std::ifstream in(filename, std::ios::binary);
  if (!in) {
    throw std::runtime_error(std::string("failed to open graph: ") + filename);
  }

  unsigned width, ep_;
  if (!in.read((char *)&width, sizeof(unsigned)) ||
      !in.read((char *)&ep_, sizeof(unsigned))) {
    throw std::runtime_error(std::string("truncated graph header: ") + filename);
  }

  while (true) {
    unsigned k;
    in.read((char *)&k, sizeof(unsigned));
    if (in.eof()) {
      break;
    }
    if (!in) {
      throw std::runtime_error(std::string("failed to read graph degree: ") + filename);
    }
    std::vector<unsigned> tmp(k);
    if (k > 0 && !in.read((char *)tmp.data(), k * sizeof(unsigned))) {
      throw std::runtime_error(std::string("truncated graph row: ") + filename);
    }
    final_graph_.push_back(tmp);
  }
}

void PrintUsage(const char *program) {
    std::cerr
        << "Usage: " << program
        << " graph cluster_label out_order num_clusters vectors_per_block"
           " [rrf_c] [max_reverse_entries] [num_threads] [use_cn=1] [use_mnn=1]\n"
        << "Output: out_order[new_id] = old_id\n";
}

int main(int argc, char **argv) {
    using namespace ParallelClusteredBlockGorder;
    if (argc < 6 || argc > 11) {
        PrintUsage(argv[0]);
        return 1;
    }

    const char *graph_path = argv[1];
    const char *cluster_path = argv[2];
    const char *output_path = argv[3];
    unsigned num_clusters =
        static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10));
    unsigned vectors_per_block =
        static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));
    double rrf_c = argc > 6 ? std::strtod(argv[6], nullptr) : 60.0;
    unsigned max_reverse_entries =
        argc > 7 ? static_cast<unsigned>(std::strtoul(argv[7], nullptr, 10))
                 : 1024;
    int num_threads = argc > 8 ? std::atoi(argv[8]) : 8;

    auto parse_bool = [](const char *value) {
        if (std::string(value) == "0") return false;
        if (std::string(value) == "1") return true;
        throw std::invalid_argument("use_cn and use_mnn must be 0 or 1");
    };
    bool use_cn = true, use_mnn = true;
    try {
        if (argc > 9) use_cn = parse_bool(argv[9]);
        if (argc > 10) use_mnn = parse_bool(argv[10]);
    } catch (const std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    vector<vector<unsigned>> graph;
    Load(graph_path, graph);

    vector<unsigned> cluster_id;
    std::ifstream inFile(cluster_path);
    if (!inFile) {
        std::cerr << "failed to open cluster labels: " << cluster_path << std::endl;
        return 1;
    }
    unsigned val;
    while (inFile >> val) {
        cluster_id.push_back(val);
    }
    std::cout << "读取完成，共 " << cluster_id.size() << " 个元素" << std::endl;

    vector<unsigned> old_to_new =
        buildClusteredReorderMapParallel(
            graph,
            cluster_id,
            num_clusters,
            vectors_per_block,
            rrf_c,
            max_reverse_entries,
            use_cn,
            use_mnn,
            num_threads
        );

    std::ofstream outFile(output_path);
	if (!outFile) {
		std::cerr << "failed to open output: " << output_path << std::endl;
		return 1;
	}

	vector<unsigned> new_to_old(old_to_new.size());
	for (unsigned old_id = 0; old_id < old_to_new.size(); old_id++) {
		new_to_old[old_to_new[old_id]] = old_id;
	}

	for (unsigned new_id = 0; new_id < new_to_old.size(); new_id++) {
		outFile << new_to_old[new_id];
		if (new_id < new_to_old.size() - 1) outFile << " ";
	}
	outFile.close();

    return 0;
}
