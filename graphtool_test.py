# from graph_tool.all import *

# g = Graph(directed=False)

# # g = Graph([("foo", "bar"), ("gnu", "gnat")], hashed=True)
# a = g.add_vertex()
# b = g.add_vertex()
# e = g.add_edge(a, b)
# print(g.vertex_index[a])
# neightbors = [x for x in g.vertex(0).out_neighbors()]
# print(neightbors)
import rustworkx

graph = rustworkx.PyGraph()

# Each time add node is called, it returns a new node index
a = graph.add_node("A")
b = graph.add_node("B")
c = graph.add_node("C")

# add_edges_from takes tuples of node indices and weights,
# and returns edge indices
graph.add_edges_from([(a, b, 1.5), (a, c, 5.0), (b, c, 2.5)])

# Returns the path A -> B -> C
# rustworkx.dijkstra_shortest_paths(graph, a, c, weight_fn=float)
print(graph[a])
print(graph.neighbors(a))