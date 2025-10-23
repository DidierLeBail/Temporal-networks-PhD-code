from main import *

t_ij = [
    [0, (1, 0)],
    [1, (1, 2)],
    [1, (0, 1)],
    [1, (0, 2)],
    [2, (2, 0)]
]

table = Table_interaction(t_ij, is_formatted=True)
graph = table.to_graph()

# sample edges
inclusions = Edge_obs().sample_inclusions(graph)
print(inclusions)
agg_net = sample_time_weight(inclusions)
print(agg_net)
tradi_edge_weight = agg_net.reduce()
print(tradi_edge_weight)
tradi_edge_weight = sample_time_weight(inclusions, reduce=True)
print(tradi_edge_weight)
nb_edges_per_time = sample_space_weight(inclusions)
print(nb_edges_per_time)
exit()


# sample NCTNs
nctn = NCTN_obs(depth=2)
inclusions = nctn.sample_inclusions(graph)
print(inclusions)
space_time_weight = sample_space_time_weight(inclusions)
print(space_time_weight)
