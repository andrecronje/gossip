import networkx as nx
from networkx.algorithms.distance_measures import diameter
from networkx.linalg.laplacianmatrix import normalized_laplacian_matrix
from typing import Type, ClassVar, Dict, Set, List
from math import log2, log
from math import sqrt
from numpy.linalg import eigvals
import scipy
import random

import pickle

debug = False


class Graph:
    def __init__(self, g: nx.Graph, dia: int=None):
        self.nodes: ClassVar[Dict[int, Node]] = {}
        self.sharp_nodes: int = -1
        self.diameter: int = dia if dia is not None else -1
        self.generate_graph(g)

    def generate_graph(self, g: nx.Graph):
        for uid in g:
            v = Node(uid)
            self.nodes[uid] = v

        self.sharp_nodes = len(self.nodes)

        for uid in g:
            node = self.nodes[uid]
            for eid in g.neighbors(uid):
                neighbor = self.nodes[eid]
                node.adj.add(neighbor)
                node.adjlist.append(neighbor)

        if self.diameter == -1:
            self.diameter = diameter(g)

    def iterate_with_discover(self):
        # round 1
        for uid, v in self.nodes.items():
            if not v.has_all():
                v.discover()
                v.push()

        # round 2
        for uid, v in self.nodes.items():
            if not v.has_all():
                v.pull()

        # round 3
        for uid, v in self.nodes.items():
            if not v.has_all():
                v.pull()

        # round 4
        for uid, v in self.nodes.items():
            if not v.has_all():
                v.push()

    def iterate_without_discover(self):
        for uid, v in self.nodes.items():
            v.push()
        for uid, v in self.nodes.items():
            v.pull()

    def inspect(self):
        for uid, v in self.nodes.items():
            rumor = list(v.rumor)
            rumor.sort()
            links = list(v.links)
            links.sort()
            print(uid, "Rumor:", rumor, "Links:", links)

    def spread_rumor(self):
        count = 0

        while not all(v.has_all() for uid, v in self.nodes.items()):
            count += 1
            self.iterate_with_discover()
            if debug: print("Discover:", count), self.inspect()

        for i in range(self.diameter - 1):
            self.iterate_without_discover()
            if all(v.know_all(self.sharp_nodes) for uid, v in self.nodes.items()):
                break
            if debug: print("Spread:", i+1), self.inspect()

        global_iteration = min(self.diameter - 1, i + 1)

        print("Every Node Knows All :", all(v.know_all(self.sharp_nodes) for uid, v in self.nodes.items()))
        print("Discovering Iteration:", count)
        print("Standard Discovering :", log2(self.sharp_nodes))
        print("Total Rounds:", (1+count)*count/2 + global_iteration * count)

    def random_exchange(self):
        count = 0
        while not all(v.know_all(self.sharp_nodes) for uid, v in self.nodes.items()):
            count += 1
            for uid, v in self.nodes.items():
                v.act()

        return count


class Node:
    def __str__(self):
        return str(self.uid)

    def __repr__(self):
        return str(self.uid)

    def __init__(self, uid: int):
        self.uid: int = uid
        self.adj: ClassVar[Set[Node]] = set()
        self.adjlist: ClassVar[List[Node]] = list()
        self.rumor: ClassVar[Set[Node]] = {self, }
        self.inbox: ClassVar[Set[Node]] = set()
        self.links: ClassVar[List[Node]] = []
        self.__has_all: bool = False

    def discover(self):
        self.links.append((self.adj - self.rumor).pop())

    def push(self):
        reverse = list(self.links)
        reverse.reverse()
        for i in reverse:
            self.exchange_rumor(i)

    def exchange_rumor(self, u: 'Node'):
        u.rumor |= self.rumor
        self.rumor |= u.rumor

    def pull(self):
        for i in self.links:
            self.exchange_rumor(i)

    def has_all(self):
        if self.__has_all:
            return self.__has_all
        else:
            self.__has_all = len(self.rumor & self.adj) == len(self.adj)
            return self.__has_all

    def know_all(self, total: int):
        return len(self.rumor) == total

    def __lt__(self, other):
        return self.uid < other.uid

    def act(self):
        i = random.randint(0, len(self.adjlist)-1)
        self.exchange_rumor(self.adjlist[i])


class GraphWithInfo:
    def __init__(self, g: nx.Graph, info: str, l2: float, d: int=None):
        self.g = g
        self.info = info
        self.l2 = l2
        self.dia = d if d is not None else diameter(self.g)


class Tester:
    def __init__(self):
        self.graphs: ClassVar[List['GraphWithInfo']] = []
        for i in range(3):
            self.graphs.append(self.erdos_renyi(2048, log(2048)/2048, random.randint(0, 65535), str(i)))

        self.graphs.append(self.barbell(1024, 12))
        self.graphs.append(self.c_barbell(256, 8))
        self.graphs.append(self.balanced_tree(2, 10))

        for g in self.graphs:
            print(g.info, "l2: %.20f", g.l2, "diameter:", g.dia)
            print("nodes:", len(g.g.nodes), "edges:", g.g.size())

    @staticmethod
    def l2(g: nx.Graph):
        norml = normalized_laplacian_matrix(G.g).todense()
        es = eigvals(norml)
        l2 = min(x for x in es if x > 0)
        return sqrt(2*l2)

    @staticmethod
    def erdos_renyi(nodes, p, seed, serial=''):
        count = 0
        while True:
            count += 1
            print('Looking for connected graph...', count)
            ng = nx.erdos_renyi_graph(nodes, p, seed=seed)
            seed += 1
            if nx.is_connected(ng):
                break

        return GraphWithInfo(ng, "({0}, {1})-Erdos-Renyi{2}".format(nodes, p, serial), Tester.l2(ng))

    @staticmethod
    def barbell(clique, path):
        bbell = nx.barbell_graph(clique, path)
        return GraphWithInfo(bbell, "({0}, {1})-Barbell".format(clique, path), Tester.l2(bbell), 2+path)

    @staticmethod
    def c_barbell(clique: int, quantity: int):
        temp = nx.empty_graph()
        for i in range(quantity):
            g = nx.Graph()
            g.add_nodes_from(range(i*clique, (i+1)*clique))
            g.add_edges_from([(u, v) for u in range(i*clique, (i+1)*clique) for v in range(u, (i+1)*clique)])
            temp = nx.compose(temp, g)

        for i in range(1, quantity):
            temp.add_edge(i*clique-1, i*clique)

        return GraphWithInfo(temp, "{0}-Barbell({1}clique)".format(quantity, clique), Tester.l2(temp), quantity*2)

    @staticmethod
    def balanced_tree(child: int, height: int):
        t = nx.generators.classic.balanced_tree(child, 10)
        return GraphWithInfo(t, "{0}-Node-{1}-Layer-Tree".format(2**height-1, height), Tester.l2(t), height*2)

    def test(self):
        for index, G in enumerate(self.graphs[3:]):
            print("*"*16)
            g, info, l2, dia = G.g, G.info, G.l2, G.dia
            print(info)
            print(nx.is_connected(g))
            if index == 0 or index ==1:
                deterministic_gossip = Graph(g, dia)
            else:
                deterministic_gossip = Graph(g, dia+2)
            print('l2:', l2)
            print('diameter:', dia)
            deterministic_gossip.spread_rumor()

            count = 0
            max_count = 0
            iteration = 5
            for i in range(iteration):
                push_pull = Graph(g, dia+2)
                r = push_pull.random_exchange()
                print('iteration', i, ':', r)
                max_count = max(max_count, r)
                count += r

            count /= iteration
            print('Random Avg:', count, 'Max:', max_count)


if __name__ == "__main__":
    with open('graphs.pkl', 'rb') as file:
        t = pickle.load(file)
        t.test()
