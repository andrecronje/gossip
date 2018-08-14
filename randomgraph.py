import networkx as nx
from math import log

NODES = 50
PROB = 2 * log(NODES) / NODES
# count = 0
# for i in range(20):
#     print(i)
#     g = nx.erdos_renyi_graph(NODES, PROB)
#     if not nx.is_connected(g):
#         count += 1
#
# print(count)
g = nx.erdos_renyi_graph(NODES, PROB)
nx.write_adjlist(g, 'test.adjlist')
