from collections import defaultdict
from functools import cached_property

import rustlib

# import rustworkx
# class Graph:
# def __init__(self) -> None:
#     self.graph = rustworkx.PyGraph()
#     self.name_to_node_id = {}
#     self.node_id_to_name = {}

# def add_node(self, name, **kwargs):
#     if name in self.name_to_node_id:
#         if kwargs:
#             obj = self.graph[self.name_to_node_id[name]]
#             obj.update(**kwargs)
#     else:
#         node_id = self.graph.add_node(dict(**kwargs))
#         self.name_to_node_id[name] = node_id
#         self.node_id_to_name[node_id] = name

# def add_edge(self, name1, name2):
#     node1 = self.name_to_node_id[name1]
#     node2 = self.name_to_node_id[name2]
#     self.graph.add_edge(node1, node2, None)

# def add_edges_from(self, pairs):
#     for pair in pairs:
#         self.add_node(pair[0])
#         self.add_node(pair[1])
#         self.add_edge(pair[0], pair[1])

# @cached_property
# def nodes(self):
#     return NodeView(self)

# def edges(self):
#     return [
#         (self.node_id_to_name[u], self.node_id_to_name[v]) for u, v in self.graph.edge_list()
#     ]

# def neighbors(self, name):
#     node = self.name_to_node_id[name]
#     return [self.node_id_to_name[n] for n in self.graph.neighbors(node)]


# def remove_node(self, name):
#     node = self.name_to_node_id[name]
#     self.graph.remove_node(node)
#     del self.name_to_node_id[name]
#     del self.node_id_to_name[node]
class DiGraph:
    def __init__(self) -> None:
        self.graph = rustlib.DiGraph()
        self.data = {}
        self.name_to_node_id = {}
        self.node_id_to_name = {}
        self.last_node_id = 0
        self.nodes2add = []
        self.edges2add = []
        self.tags2add = []

    def add_node(self, name, node_id=None, **kwargs):
        if name in self.name_to_node_id:
            if kwargs:
                obj = self.data[self.name_to_node_id[name]]
                obj.update(**kwargs)
        else:
            if node_id is None:
                node_id = self.graph.add_node()
            self.last_node_id = node_id
            self.data[node_id] = kwargs
            self.data[node_id]["node_id"] = node_id
            self.name_to_node_id[name] = node_id
            self.node_id_to_name[node_id] = name

    def add_node_cache(self, name, **kwargs):
        self.nodes2add.append((name, kwargs))

    def add_nodes(self, name_data_pairs):
        node_ids = self.graph.add_nodes(len(name_data_pairs))
        for i, (name, data) in enumerate(name_data_pairs):
            self.add_node(name, node_id=node_ids[i], **data)

    def update_nodes_from_cache(self):
        self.add_nodes(self.nodes2add)

    def add_edge(self, name1, name2):
        node1 = self.name_to_node_id[name1]
        node2 = self.name_to_node_id[name2]
        self.graph.add_edge(node1, node2)

    def add_edge_cache(self, name1, name2):
        self.edges2add.append((name1, name2))

    def update_edges_from_cache(self):
        self.graph.add_edges(
            [(self.name_to_node_id[a], self.name_to_node_id[b]) for a, b in self.edges2add]
        )

    def add_edges_from(self, pairs):
        for pair in pairs:
            self.add_node(pair[0])
            self.add_node(pair[1])
            self.add_edge(pair[0], pair[1])

    def add_edges_from_cache(self, pairs):
        for pair in pairs:
            # self.add_node_cache(pair[0])
            # self.add_node_cache(pair[1])
            # self.add_edge_cache(pair[0], pair[1])
            self.nodes2add.append((pair[0], {}))
            self.nodes2add.append((pair[1], {}))
            self.edges2add.append((pair[0], pair[1]))

    @cached_property
    def nodes(self):
        return NodeView(self)

    def node_names(self):
        return list(self.name_to_node_id.keys())

    def edges(self):
        return [
            (self.node_id_to_name[u], self.node_id_to_name[v]) for u, v in self.graph.edge_list()
        ]

    def outgoings(self, name):
        return self.__directions(name, "outgoing")

    def incomings(self, name):
        return self.__directions(name, "incoming")

    def __directions(self, name, direction):
        if name not in self.name_to_node_id:
            return []
        node = self.name_to_node_id[name]
        return [
            self.node_id_to_name[n]
            for n in (
                self.graph.outgoings(node)
                if direction == "outgoing"
                else self.graph.incomings(node)
            )
        ]

    def get_all_outgoings(self, src_tag):
        return {
            self.node_id_to_name[node_id]: [self.node_id_to_name[o] for o in outgoing]
            for node_id, outgoing in self.graph.outgoings_from(src_tag).items()
        }

    def get_all_incomings(self, src_tag):
        return {
            self.node_id_to_name[node_id]: [self.node_id_to_name[o] for o in outgoing]
            for node_id, outgoing in self.graph.incomings_from(src_tag).items()
        }

    def remove_node(self, name):
        if name not in self.name_to_node_id:
            return
        node = self.name_to_node_id[name]
        self.graph.remove_node(node)
        del self.name_to_node_id[name]
        self.name_to_node_id[self.node_id_to_name[self.last_node_id]] = node
        self.node_id_to_name[node] = self.node_id_to_name[self.last_node_id]
        self.data[node] = self.data[self.last_node_id]
        self.data[node]["node_id"] = node
        self.last_node_id -= 1

    def remove_nodes(self, names):
        for name in names:
            self.remove_node(name)

    def add_tag(self, name, tag):
        node = self.name_to_node_id[name]
        self.graph.update_node_data(node, tag)

    def add_tag_cache(self, name, tag):
        self.tags2add.append((name, tag))

    def update_tags_from_cache(self):
        self.graph.update_node_datas([(self.name_to_node_id[a], b) for a, b in self.tags2add])

    def get_tag(self, name):
        node = self.name_to_node_id[name]
        return self.graph.node_data(node)

    def __build_direction_map(self, tag, src_tag, direction):
        return {
            self.node_id_to_name[node_id]: [
                (self.node_id_to_name[a], self.node_id_to_name[b]) for a, b in neighbor_pair
            ]
            for node_id, neighbor_pair in (
                self.graph.build_outgoing_map(tag, src_tag)
                if direction == "outgoing"
                else self.graph.build_incoming_map(tag, src_tag)
            ).items()
        }

    def build_outgoing_map(self, tag, src_tag):
        return self.__build_direction_map(tag, src_tag, "outgoing")

    def build_incoming_map(self, tag, src_tag):
        return self.__build_direction_map(tag, src_tag, "incoming")

    # def build_descendant(self, name, tag):
    #     node = self.name_to_node_id[name]
    #     return [
    #         (self.node_id_to_name[a], self.node_id_to_name[b])
    #         for a, b in self.graph.get_ancestor_until(node, tag)
    #     ]

    def __len__(self):
        return len(self.name_to_node_id)

    @property
    def size(self):
        return len(self)

    def rename_node(self, old_name, new_name):
        node_id = self.name_to_node_id[old_name]
        self.name_to_node_id[new_name] = node_id
        self.node_id_to_name[node_id] = new_name
        del self.name_to_node_id[old_name]

    def has_node(self, name):
        return name in self.name_to_node_id

    # def toposort(self):
    #     return [self.node_id_to_name[node_id] for node_id in self.graph.toposort()]


class NodeView:
    def __init__(self, bind: DiGraph):
        self.bind = bind

    def __call__(self, data=None, type="list"):
        if data is None:
            if type == "list":
                pair = [
                    (n, self.bind.data[self.bind.name_to_node_id[n]])
                    for n in self.bind.name_to_node_id.keys()
                ]
            elif type == "dict":
                pair = {
                    n: self.bind.data[self.bind.name_to_node_id[n]]
                    for n in self.bind.name_to_node_id.keys()
                }
            else:
                raise ValueError("type must be list or dict")
        else:
            if type == "list":
                pair = [
                    (n, self.bind.data[self.bind.name_to_node_id[n]][data])
                    for n in self.bind.name_to_node_id.keys()
                ]
            elif type == "dict":
                pair = {
                    n: self.bind.data[self.bind.name_to_node_id[n]][data]
                    for n in self.bind.name_to_node_id.keys()
                }
            else:
                raise ValueError("type must be list or dict")
        return pair

    def __getitem__(self, name):
        return self.bind.data[self.bind.name_to_node_id[name]]


if __name__ == "__main__":
    graph = DiGraph()
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_edge("A", "B")
    print(graph.nodes())
    print(graph.neighbors("A"))
    print(graph.nodes())
    print(graph.edges())
    graph.add_tag("A", 1)
    # print(graph.get_ancestor_until("B", 1))
    # a.add_edge(0, 2);
    # a.add_edge(2, 4);
    # a.add_edge(2, 5);
    # a.add_edge(1, 2);
    # a.add_edge(2, 5);
    # a.add_edge(1, 3);
    # a.add_edge(3, 5);
    # a.update_node_data(0, 1);
    # a.update_node_data(1, 1);
