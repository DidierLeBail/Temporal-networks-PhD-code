from typing import List, Tuple, Union
import numpy as np
import networkx as nx

def get_info(temp_net):
	"""Compute the number of timestamps, nodes and temporal edges,
	as well as the parameters of the node activity distribution viewed as the exponential of a two sided truncated Gaussian variable
	"""
	info = {}
	info['T'] = len(temp_net.TN)
	info['nb of edges'] = len(temp_net.data)
	info['N'] = temp_net.N
	activity = np.zeros(temp_net.N, dtype=float)
	for events in temp_net.TN:
		for node in events.nodes:
			activity[node] += 1
	info['a_min'] = np.min(activity) / info['T']
	info['a_max'] = np.max(activity) / info['T']
	tab = np.log10(activity) - np.log10(info['T'])
	values, bins = np.histogram(tab, density=True)
	
	# estimate the parameters mu and sigma
	bin_min = len(bins) - 3
	while values[bin_min] > values[-1]:
		bin_min -= 1
	mu_tab = [el for el in tab if el >= bins[bin_min]]
	info['mu'] = np.mean(mu_tab)
	info['sigma'] = 2 * np.sqrt(np.var(mu_tab))
	return info

def group_by_time(t_ij):
	r"""Compute `data_by_time`.
	
	Parameters
	----------
	t_ij : array[int]
		The data of pairwise interactions.
		The time label should be in the first column, and should increase or stay constant from one row to the next.
	
	Returns
	-------
	data_by_time : array[int]
		Contains 2 columns, with `data_by_time[i] = [n_1, n_2]` with i the :math:`(i+1)^{\text{th}}` time appearing in `data`
		n_1 the first line of occurrence of i and n_2 - 1 the last one.
	"""
	data_by_time = []
	n1 = 0; n_max = len(t_ij)
	for n in range(1, n_max):
		if t_ij[n][0] > t_ij[n-1][0]:
			data_by_time.append([n1, n])
			n1 = n

	# take care of the last line of data
	data_by_time.append([n1, n_max])
	return data_by_time

class Graph_interaction:
	"""
	This view of a temporal network is more relevant for aggregation, (some) randomizations, sampling most observables, etc.
	"""
	def __init__(self, graphs:List[nx.Graph], nb_nodes:Union[int, None]=None):
		self._graphs = graphs
		if nb_nodes is None:
			self.nb_nodes = len(set().union(*(set(graph.nodes) for graph in self.graphs)))
		self.duration = len(graphs)

	def __iter__(self):
		def gen():
			for graph in self._graphs:
				yield graph
		return gen()

	def __getitem__(self, key):
		return self._graphs[key]

	def sliding_agg(self, agg:int):
		"""Return the temporal graph at aggregation level `agg` (sliding aggregation)."""
		# new_graphs[t] = aggregated graph of interactions on t^th time interval
		nb_time = len(self._graphs)
		new_nb = nb_time - agg + 1
		new_graphs = [nx.Graph() for _ in range(new_nb)]
		
		# initial graph
		G = nx.Graph()
		for k in range(agg):
			for edge in self._graphs[k].edges:
				if G.has_edge(*edge):
					G[edge[0]][edge[1]]['weight'] += 1
				else:
					G.add_edge(*edge, weight=1)
		
		# main loop
		for t in range(new_nb - 1):
			new_graphs[t].add_edges_from(G.edges(data=False))

			# remove the edges from time t
			for edge in self.graphs[t].edges:
				if G[edge[0]][edge[1]]['weight'] == 1:
					G.remove_edge(*edge)
				else:
					G[edge[0]][edge[1]]['weight'] -= 1
			
			# add the edges from time t + agg
			for edge in self.graphs[t + agg].edges:
				if G.has_edge(*edge):
					G[edge[0]][edge[1]]['weight'] += 1
				else:
					G.add_edge(*edge, weight=1)
		
		# last graph
		new_graphs[new_nb - 1].add_edges_from(G.edges(data=False))
		return Graph_interaction(new_graphs, nb_nodes=self.nb_nodes)

	def full_agg(self,
		is_weighted:bool=False
	):
		r"""Return the fully aggregated network (weighted or not).

		This network is static, and is invariant under time shuffling of the original temporal network.
		
		Parameters
		----------
		is_weighted : bool
			if True, the weight from i to j is the total number of interactions between i and j
			if False, an edge is drawn between i and j iif they have interacted at least once
		
		Returns
		-------
		static_net : nx.Graph
			the fully aggregated network, with the same nodes as the temporal network, but a single timestamp
		"""
		static_net = nx.Graph()
		if is_weighted:
			for graph in self._graphs:
				for edge in graph.edges:
					if static_net.has_edge(*edge):
						static_net[edge[0]][edge[1]]['weight'] += 1
					else:
						static_net.add_edge(*edge, weight=1)
		else:
			for graph in self._graphs:
				static_net.add_edges_from(graph.edges)
		return static_net

class Table_interaction:
	r"""A table containing a time sequence of undirected unweighted pairwise interactions.

	This view of a temporal network is more relevant for formatting interaction data, saving them, splitting them by day, etc.

	The first column contains the time label and the second column contains the edge active at that time.
	A correctly formatted table should satisfy the following properties:
	- node labels go from 0 to nb of nodes - 1
	- time starts from 0 and increases 1 by 1
	- self-loops are removed
	- the rows are ordered by increasing value of the time
	- two given nodes interact at most once with each other at each time step
	
	Parameters
	----------
	t_ij : List[List[int]]
		Has three columns: 'time'...
	"""
	def __init__(self,
		t_ij:Union[List[List[Union[int, Tuple[int, int]]]], str],
		is_formatted:bool=True,
		nb_nodes:Union[int, None]=None
	):
		try:
			self.data = [
				[row[0], tuple(row[1:])] for row in np.loadtxt(t_ij, dtype=int)
			]
		except TypeError:
			self.data = t_ij

		self.data_by_time = []
		if is_formatted:
			self._group_by_time()
		if nb_nodes is None:
			self.nb_nodes = len(self.all_nodes())
		self.duration = len(self.data_by_time)

	def __str__(self):
		return '\n'.join([str(row) for row in self.data])

	def _group_by_time(self):
		self.data_by_time = group_by_time(self.data)
		self.duration = len(self.data_by_time)
	
	def edges(self, t):
		"""Return the edges appearing at the (t+1)^{th} time in self.data"""
		n1, n2 = self.data_by_time[t]
		for row in self.data[n1: n2]:
			yield row[1]

	def nodes(self, t):
		"""Return the active nodes at (t+1)^{th} time in self.data"""
		n1, n2 = self.data_by_time[t]
		return set().union(*(set(row[1]) for row in self.data[n1: n2]))
	
	def all_nodes(self):
		"""return all nodes in self.data"""
		return set().union(*(set(row[1]) for row in self.data))

	def format(self,
		max_T:int=np.inf,
		is_sorted:bool=True,
		has_duplicata:bool=False,
		relabel_nodes:bool=False,
		relabel_times:bool=False
	):
		"""Replace `t_ij` by its formatted version.

		Does nothing by default.
		
		Parameters
		----------
		max_T : int
			Restrict to the times equal or lower than `max_T`.
		t_col : int
			Indicates which column of `self.data` contains the time label of the interactions.
		"""
		# ensure the rows are sorted by increasing time
		if not is_sorted:
			self.data.sort(key=lambda row: row[0])

		# group the interactions by time of occurrence
		self._group_by_time()
		
		# restrict to the times smaller than `max_T`
		last_line = min(max_T, len(self.data_by_time))
		self.data = self.data[:self.data_by_time[last_line - 1][1]]
		
		# remove self-loops and edge duplicata (active more than once per time step)
		if has_duplicata:
			new_data = []
			for el in self.data_by_time:
				all_edges = set()
				for row in self.data[el[0]: el[1]]:
					edge = row[1]
					if edge[0] < edge[1]:
						all_edges.add(edge)
					elif edge[0] > edge[1]:
						all_edges.add(edge[::-1])
				t = self.data[el[0]][0]
				new_data.extend([[t, edge] for edge in all_edges])
			self.data = new_data

			# recompute data_time
			self._group_by_time()
		
		# relabel the nodes
		if relabel_nodes:
			node_to_int = {node: num for num, node in enumerate(self.all_nodes())}
			for row in self.data:
				row[1] = tuple(node_to_int[k] for k in row[1])

		# relabel the times
		if relabel_times:
			for new_t, el in enumerate(self.data_by_time):
				for row in self.data[el[0]: el[1]]:
					row[0] = new_t
	
	def to_graph(self):
		"""Return self.data under the form of a sequence of undirected unweighted graphs."""
		return Graph_interaction(
			[nx.Graph(self.edges(t)) for t in range(len(self.data_by_time))],
			nb_nodes=self.nb_nodes
		)

	pass
	def split_per_days(self, **kwargs):
		"""Split `self.data` into multiple tables of interaction, one per identified day."""
		self.format(**kwargs)
		pass

class Temp_net:
	"""A temporal network contains a table view and a graph view.
	
	Depending on the operation we want to perform (aggregation, time shuffling, etc.)
	"""
	def __init__(self, init_data:Union[Table_interaction, Graph_interaction]):
		self.data = init_data
	
	def sliding_agg(self, agg):
		self.data_sliding_agg(agg)

if __name__ == "__main__":
	t_ij = [
		[-1, (10, 10)],
		[-1, (1, 10)],
		[-1, (10, 1)],
		[-1, (10, 1)],
		[11, (2, 4)],
		[11, (2, 4)],
		[5, (6, 4)],
		[5, (6, 10)]
	]

	test = Table_interaction(t_ij, is_formatted=False)
	test.format(is_sorted=False, has_duplicata=True, relabel_nodes=True, relabel_times=True)
	print(test)
	print()
	temp_graph = test.to_graph()
	for graph in temp_graph.graphs:
		print(list(graph.edges))
