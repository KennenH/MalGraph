import random
from trueskill import Rating, rate_1vs1
import networkx as nx
import os


# noinspection DuplicatedCode
def __get_big_sccs(g):
    num_big_sccs = 0
    big_sccs = []
    for sub in (g.subgraph(c).copy() for c in nx.strongly_connected_components(g)):
        number_of_nodes = sub.number_of_nodes()
        if number_of_nodes >= 2:
            # strongly connected components
            num_big_sccs += 1
            big_sccs.append(sub)
    # print(" # big sccs: %d" % (num_big_sccs))
    return big_sccs


# noinspection DuplicatedCode
def __nodes_in_scc(sccs):
    scc_nodes = []
    scc_edges = []
    for scc in sccs:
        scc_nodes += list(scc.nodes())
        scc_edges += list(scc.edges())

    # print("# nodes in big sccs: %d" % len(scc_nodes))
    # print("# edges in big sccs: %d" % len(scc_edges))
    return scc_nodes


def __scores_of_nodes_in_scc(sccs, players):
    scc_nodes = __nodes_in_scc(sccs)
    scc_nodes_score_dict = {}
    for node in scc_nodes:
        scc_nodes_score_dict[node] = players[node]
    # print("# scores of nodes in scc: %d" % (len(scc_nodes_score_dict)))
    return scc_nodes_score_dict


def __filter_big_scc(g, edges_to_be_removed):
    # Given a graph g and edges to be removed
    # Return a list of big scc subgraphs (# of nodes >= 2)
    g.remove_edges_from(edges_to_be_removed)
    sub_graphs = filter(lambda scc: scc.number_of_nodes() >= 2,
                        [g.subgraph(c).copy() for c in nx.strongly_connected_components(g)])
    return sub_graphs


def __remove_cycle_edges_by_agony_iterately(sccs, players, edges_to_be_removed):
    while True:
        graph = sccs.pop()
        pair_max_agony = None
        max_agony = -1
        for pair in graph.edges():
            u, v = pair
            agony = max(players[u] - players[v], 0)
            if agony >= max_agony:
                pair_max_agony = (u, v)
                max_agony = agony
        edges_to_be_removed.append(pair_max_agony)
        # print("graph: (%d,%d), edge to be removed: %s, agony: %0.4f" % (graph.number_of_nodes(),graph.number_of_edges(),pair_max_agony,max_agony))
        graph.remove_edges_from([pair_max_agony])
        # print("graph: (%d,%d), edge to be removed: %s" % (graph.number_of_nodes(),graph.number_of_edges(),pair_max_agony))
        sub_graphs = __filter_big_scc(graph, [pair_max_agony])
        if sub_graphs:
            for index, sub in enumerate(sub_graphs):
                sccs.append(sub)
        if not sccs:
            return


def __compute_trueskill(pairs, players):
    if not players:
        for u, v in pairs:
            if u not in players:
                players[u] = Rating()
            if v not in players:
                players[v] = Rating()

    random.shuffle(pairs)
    for u, v in pairs:
        players[v], players[u] = rate_1vs1(players[v], players[u])

    return players


def __get_players_score(players, n_sigma):
    relative_score = {}
    for k, v in players.items():
        relative_score[k] = players[k].mu - n_sigma * players[k].sigma
    return relative_score


def __measure_pairs_agreement(pairs, nodes_score):
    # whether nodes in pairs agree with their ranking scores
    num_correct_pairs = 0
    num_wrong_pairs = 0
    total_pairs = 0
    for u, v in pairs:
        if u in nodes_score and v in nodes_score:
            if nodes_score[u] <= nodes_score[v]:
                num_correct_pairs += 1
            else:
                num_wrong_pairs += 1
            total_pairs += 1
    if total_pairs != 0:
        acc = num_correct_pairs * 1.0 / total_pairs
    # print("correct pairs: %d, wrong pairs: %d, total pairs: %d, accuracy: %0.4f" % (num_correct_pairs,num_wrong_pairs,total_pairs,num_correct_pairs*1.0/total_pairs))
    else:
        acc = 1
    # print("total pairs: 0, accuracy: 1")
    return acc


def __trueskill_ratings(pairs, iter_times=15, n_sigma=3, threshold=0.85):
    players = {}
    for i in range(iter_times):
        players = __compute_trueskill(pairs, players)
        relative_scores = __get_players_score(players, n_sigma=n_sigma)
        accu = __measure_pairs_agreement(pairs, relative_scores)
        if accu >= threshold:
            return relative_scores
    # end = datetime.now()
    # time_used = end - start
    # print("time used in computing true skill: %0.4f s, iteration time is: %i" % (time_used.seconds, (i + 1)))
    return relative_scores


# noinspection DuplicatedCode
# def breaking_cycles_by_TS(graph_path):
#     g = nx.read_edgelist(graph_path, create_using=nx.DiGraph(), nodetype=int)
#     players_score_dict = __trueskill_ratings(list(g.edges()), iter_times=15, n_sigma=3, threshold=0.95)
#     g.remove_edges_from(list(nx.selfloop_edges(g)))
#     big_sccs = __get_big_sccs(g)
#     scc_nodes_score_dict = __scores_of_nodes_in_scc(big_sccs, players_score_dict)
#     edges_to_be_removed = []
#     if len(big_sccs) == 0:
#         print("After removal of self loop edgs: %s" % nx.is_directed_acyclic_graph(g))
#         return
#
#     __remove_cycle_edges_by_agony_iterately(big_sccs, scc_nodes_score_dict, edges_to_be_removed)
#     g.remove_edges_from(edges_to_be_removed)
#     nx.write_edgelist(g, out_path)


# edgelist形式为[(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
def perform_breaking_edges(g):
    players_score_dict = __trueskill_ratings(list(g.edges()), iter_times=15, n_sigma=3, threshold=0.95)
    g.remove_edges_from(list(nx.selfloop_edges(g)))
    big_sccs = __get_big_sccs(g)
    scc_nodes_score_dict = __scores_of_nodes_in_scc(big_sccs, players_score_dict)
    edges_to_be_removed = []

    # 移除自环已经是DAG
    if len(big_sccs) == 0:
        return

    __remove_cycle_edges_by_agony_iterately(big_sccs, scc_nodes_score_dict, edges_to_be_removed)
    g.remove_edges_from(edges_to_be_removed)


if __name__ == '__main__':
    # for test only
    graph_path = 'D:\\hkn\\infected\\datasets\\text_only_nx\\text.edges'
    out_path = 'D:\\hkn\\infected\\datasets\\text_only_nx\\result.edges'

    # breaking_cycles_by_TS(graph_path)
