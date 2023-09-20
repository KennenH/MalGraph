import torch
import pymetis as metis
import numpy as np
import networkx as nx
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx, from_networkx
from typing import List

from src.utils.RemoveCycleEdgesTrueskill import perform_breaking_edges
from datetime import datetime


perform_MID = True


def create_real_batch_data(one_batch: Batch):
    if perform_MID:
        return create_MID_real_batch_data(one_batch)

    real = []
    position = [0]
    count = 0
    
    assert len(one_batch.external_list) == len(one_batch.function_edges) == len(one_batch.local_acfgs) == len(one_batch.hash), "size of each component must be equal to each other"
    
    for item in one_batch.local_acfgs:
        for acfg in item:
            real.append(acfg)
        count += len(item)
        position.append(count)
    
    if len(one_batch.local_acfgs) == 1 and len(one_batch.local_acfgs[0]) == 0:
        return (None for _ in range(6))
    else:
        real_batch = Batch.from_data_list(real)
        return real_batch, position, one_batch.hash, one_batch.external_list, one_batch.function_edges, one_batch.targets


# cfg的多实例分解batch
def create_MID_real_batch_data(one_batch: Batch):
    # 原始cfg列表
    real = []
    # 分解后的cfg列表，每个元素都是一个cfg分解后的子图列表list[Data]，因此它是二维的
    real_decomposed_cfgs = []
    position = [0]
    count = 0

    assert len(one_batch.external_list) == len(one_batch.function_edges) == len(one_batch.local_acfgs) == len(
        one_batch.hash), "size of each component must be equal to each other"

    for pe in one_batch.local_acfgs:
        # 遍历pe中的acfg
        for acfg in pe:
            # 多实例分解acfg，返回一个list[Data]
            sub_graphs = multi_instance_decompose(acfg)
            real_decomposed_cfgs.append(sub_graphs)
            real.append(acfg)
        # 一个exe中的所有acfg数量
        count += len(pe)
        # 记录每个exe中acfg的数量
        position.append(count)

    if len(one_batch.local_acfgs) == 1 and len(one_batch.local_acfgs[0]) == 0:
        return (None for _ in range(6))
    else:
        real_batch = Batch.from_data_list(real)
        real_batch.cfg_subgraph_loader = real_decomposed_cfgs
        return real_batch, position, one_batch.hash, one_batch.external_list, one_batch.function_edges, one_batch.targets


# CFG的多实例分解
# return list[Data]
def multi_instance_decompose(acfg: Data):
    # edge_index : torch.tensor([[0, 1, 2], [1, 2, 3]])
    # acfg.x是每个块的11维属性张量
    # 只有一个节点的图，所以没有边信息，edge_index长度为0，不需要处理
    # if len(acfg.x) == 1:
    #     return [acfg]
    #
    # g = nx.DiGraph()
    # g.add_edges_from(edge_index2edges(acfg.edge_index))

    return metis_MID(acfg)


def metis_MID(acfg):
    nparts = 3
    node_num = len(acfg.x)
    if node_num < 10:
        return [acfg]
    G = to_networkx(acfg).to_undirected()
    adjacency_list = [list(G.neighbors(node)) for node in sorted(G.nodes)]
    _, parts = metis.part_graph(nparts=nparts, adjacency=adjacency_list, recursive=False)  # 分解为3个子图
    sub_graphs: List[Data] = []
    subgraph_nodes: List[List[int]] = []
    for i, p in enumerate(parts):
        while p >= len(subgraph_nodes):
            subgraph_nodes.append([])
        subgraph_nodes[p].append(i)

    for sub_graph in subgraph_nodes:
        if len(sub_graph) == 0:
            continue
        indices = torch.unique(torch.tensor(sub_graph)).long()
        sub_G = G.subgraph(sub_graph)
        sub_data = from_networkx(sub_G)
        sub_data.x = acfg.x[indices]
        sub_graphs.append(sub_data)

    return sub_graphs


# 将循环结构和剩余的层次结构分别保存为Data，返回list[Data]
def structure_MID(acfg, g):
    result = []

    # 提取图中的自环结构
    # self_loop = nx.selfloop_edges(g)
    # result += [create_data(acfg.x, torch.tensor([[loop[0]], [loop[0]]])) for loop in self_loop]

    # 这里不能用self_loop，因为这个变量在被读取之后会被清空
    # g.remove_edges_from(nx.selfloop_edges(g))

    # 提取图中的循环结构
    # cycles = list(nx.simple_cycles(g))
    # if len(cycles) > 0:
    #     max_cycle = max(len(cycle) for cycle in cycles)
    #     max_cycle = max(cycles, key=len)
    #     print(max_cycle)
    #     result += [create_data(acfg.x, torch.tensor([path[:-1], path[1:]])) for path in cycles]

    # time_start = datetime.now()
    # 将图转换为DAG，尽可能保留原图的层次结构
    perform_breaking_edges(g)
    graph_index = edges2edge_index(g.edges)
    result.append(create_data(acfg.x, graph_index))
    # time_end = datetime.now()
    # print("process time = {}".format(time_end - time_start))

    return result


# 将图进行拓扑排序后进行dfs找出图中所有最长子路径，分别保存为Data，返回list[Data]
def topological_MID(acfg, g):
    # 将图转换为DAG，尽可能保留原图的层次结构
    perform_breaking_edges(g)
    # 拓扑排序
    topo_order = list(nx.topological_sort(g))
    # 初始化距离数组
    dist = {node: float('-inf') for node in g.nodes()}
    # 初始化前驱节点数组
    prev = {node: [] for node in g.nodes()}
    # 初始化起点节点的距离为0
    for node in g.nodes():
        if g.in_degree(node) == 0:
            dist[node] = 0
    # 遍历所有节点
    for node in topo_order:
        # 遍历所有后继节点
        for successor in g.successors(node):
            # 更新距离
            if dist[successor] < dist[node] + 1:
                dist[successor] = dist[node] + 1
                prev[successor] = [node]
            elif dist[successor] == dist[node] + 1:
                prev[successor].append(node)

    # 计算最长路径的长度
    max_length = max(dist.values())

    # 初始化最长路径数组
    longest_paths = []

    # 遍历所有终点节点
    for node in g.nodes():
        if g.out_degree(node) == 0 and dist[node] == max_length:
            dfs(node, [node], prev, longest_paths)

    # 将acfg中所有最长子图路径转换为Data集合，也就是说一个acfg被转换为一个Data列表
    return [create_data(acfg.x, torch.tensor([path[:-1], path[1:]])) for path in longest_paths]


def dfs(node, path, prev, longest_paths):
    if len(prev[node]) == 0:
        longest_paths.append(path)
    else:
        for predecessor in prev[node]:
            dfs(predecessor, [predecessor] + path, prev, longest_paths)


# 获取edge_index中出现过的所有元素，在x中仅保留这些元素所对应的索引
# 用于快速创建子图的x属性，注意x和edge_index都是torch.tensor
def create_data(x, edge_index):
    # 获取edge_index中出现过的元素
    indices = torch.unique(edge_index).long()
    return Data(x[indices], edge_index)


# torch.tensor([[1, 2, 3], [2, 3, 4]]) => [(1, 2), (2, 3), (3, 4)]
# 将edge_index张量转换为edges数组
def edge_index2edges(edge_index):
    return list(zip(*edge_index.tolist()))


# OutEdgeView([(1, 2), (2, 3), (3, 4)]) => torch.tensor([[1, 2, 3], [2, 3, 4]])
# 将edges数组转换为edge_index张量
def edges2edge_index(edges):
    edges = list(edges.items())
    return torch.tensor([list(edge[0]) for edge in edges]).t().contiguous()
