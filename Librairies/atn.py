import numpy as np
from math import *
import random as rd
import networkx as nx

class Gen_ATN:
	"""any generator of temporal networks inherits from this class"""
	def __init__(self,N,duree):
		self.N = N
		self.duree = duree
		self.events = []
		self.activated = np.zeros(N,dtype=int)
		self.t_min = 0

	def step(self,t):
		pass

	def refresh(self):
		self.activated[:] = 0
		self.events = []

	def edge_to_ind(self,i,j):
		if i>j:
			i,j = j,i
		return i*(2*self.N-i-1)//2 + j-i-1

	def evolve(self,t_min=0,max_count=3,max_locount=5):
		self.t_min = t_min
		#nb of nodes that have activated during the run
		nb_nodes = 0; reinitialize = False; count = 0
		while nb_nodes<self.N and count<max_count:
			#reinitialize the TN
			if reinitialize:
				self.refresh()
			#start the time flow
			t = 0; loc_count = 0
			while t<self.duree+self.t_min and loc_count<max_locount:
				if t*100%(self.duree+self.t_min)==0:
					print(str(t*100//(self.duree+self.t_min))+" % computed")
				newt = self.step(t)
				if newt==t:
					loc_count += 1
				else:
					t = newt
					loc_count = 0
			if loc_count==max_locount:
				reinitialize = True; count += 1
			else:
				#nb of nodes that have activated during the run
				nb_nodes = np.sum(self.activated)
				reinitialize = True; count += 1
		if nb_nodes==self.N:
			#return the events data
			return np.array(self.events,dtype=int)
		else:
			print('nb of nodes that activated: '+str(nb_nodes))
			return None

	#record the events and identify the activated nodes
	def record_event(self,t,event):
		#check the recording has started
		if t<self.t_min:
			pass
		else:
			#update the activated nodes
			for i in event.nodes:
				self.activated[i] = 1
			#update self.events
			for edge in event.edges:
				self.events.append([t-self.t_min,*edge])


class Min_EW(Gen_ATN):
	"""
	- the proba that an edge is active once more is equal to the fraction of times it has been active
	in the past
	- if the edge is created w=1 at t0, then P(w+1,t+1) = 1/(t-t0+2)
	- the nb of newborn edges per time is constant
	- frac = fraction of the nb of the total edges that has been activated after duree time steps
	(0<frac<1)
	"""
	def __init__(self,N,duree,frac,shift=False,removal=None,var_act=False,delay=0,n_inj=0,newborn='random'):
		super(Min_EW, self).__init__(N,duree)
		self.delay = delay
		self.n_inj = n_inj
		#social bond graph: gives the activation probability of old edges
		#social_graph[i][j]['weight'] = weight of i--j
		#social_graph[i][j]['birth'] = birth time of i--j
		self.social_graph = nx.Graph()
		#the adjacent graph gathers all the edges that can become newborn edges
		#it could be directed and weighted
		#initially it is the fully connected undirected graph
		self.adjacent_graph = nx.Graph()
		#ind_to_edge[ind] = (i,j) = edge of identifier ind, with i<j
		self.ind_to_edge = [(i,j) for i in range(N-1) for j in range(i+1,N)]
		self.adjacent_graph.add_edges_from(self.ind_to_edge)
		#the nb of newborn edges per time step follows a Poisson law of mean self.nb_new
		self.nb_new = frac*N*(N-1)/(2*duree)
		self.free_param = {}
		if shift:
			self.old_update = self.shift_old_update
			self.born_update = self.shift_born_update
		else:
			self.old_update = self.no_shift_old_update
			self.born_update = self.no_shift_born_update
		if removal=='node_unif':
			self.free_param['p_d'] = 1e-2
			self.prune = self.prune_node_unif
		elif removal=='edge_unif':
			self.free_param['p_d'] = 2e-2
			self.prune = self.prune_edge_unif
		else:
			self.prune = self.prune_none
		if var_act:
			self.step = self.step_var_act
		else:
			self.step = self.step_cst_act

	def prune_none(self):
		pass
	def prune_node_unif(self):
		#reset some nodes drawn at random
		node_to_remove = set(())
		for node in self.social_graph.nodes:
			if rd.random()<self.free_param['p_d']:
				node_to_remove.add(node)
		new_edges = set(())
		for node in node_to_remove:
			for j in self.social_graph.neighbors(node):
				if node>j:
					new_edges.add((j,node))
				else:
					new_edges.add((node,j))
		self.social_graph.remove_nodes_from(node_to_remove)
		self.adjacent_graph.add_edges_from(new_edges)
		#remove the nodes that have become isolated in the social bond graph because of the pruning
		node_to_remove = set(())
		for node in self.social_graph.nodes:
			if self.social_graph.degree[node]==0:
				node_to_remove.add(node)
		self.social_graph.remove_nodes_from(node_to_remove)
	def prune_edge_unif(self):
		nb_max = len(self.social_graph.edges)
		nb_to_remove = min(np.random.default_rng().poisson(self.free_param['p_d']*nb_max,1)[0],nb_max)
		if nb_to_remove>0:
			edge_to_remove = rd.sample(list(self.social_graph.edges),nb_to_remove)
			self.social_graph.remove_edges_from(edge_to_remove)
			self.adjacent_graph.add_edges_from(edge_to_remove)
	def no_shift_old_update(self,t):
		event = nx.Graph()
		for edge in self.social_graph.edges:
			w = self.social_graph[edge[0]][edge[1]]['weight']
			if rd.random()<w/(t+1):
				event.add_edge(*edge)
				self.social_graph[edge[0]][edge[1]]['weight'] += 1
		return event
	def no_shift_born_update(self,t,newborn):
		for edge in newborn:
			self.social_graph.add_edge(*edge,weight=1)
		self.adjacent_graph.remove_edges_from(newborn)
	def shift_old_update(self,t):
		event = nx.Graph()
		for edge in self.social_graph.edges:
			w = self.social_graph[edge[0]][edge[1]]['weight']
			if rd.random()<w/(t+1-self.social_graph[edge[0]][edge[1]]['birth']):
				event.add_edge(*edge)
				self.social_graph[edge[0]][edge[1]]['weight'] += 1
		return event
	def shift_born_update(self,t,newborn):
		for edge in newborn:
			self.social_graph.add_edge(*edge,weight=1,birth=t)
		self.adjacent_graph.remove_edges_from(newborn)

	#erase run data to be ready for a new run
	def refresh(self):
		self.activated[:] = 0
		self.events = []
		self.social_graph = nx.Graph()
		self.adjacent_graph = nx.Graph()
		self.adjacent_graph.add_edges_from(self.ind_to_edge)

	#evolve the network for time step t
	def step_var_act(self,t):
		#if the activity is imposed, we inject a current every delay, meaning n_inj edges are activated
		#at random
		if t%self.delay==0:
			active_edges = rd.sample(self.ind_to_edge,self.n_inj)
			newborn = []
			for i,j in active_edges:
				if self.social_graph.has_edge(i,j):
					self.social_graph[i][j]['weight'] += 1
				else:
					newborn.append((i,j))
			self.born_update(t,newborn)
			self.prune()
			#record the events and identify the activated nodes
			event = nx.Graph(); event.add_edges_from(active_edges)
			self.record_event(t,event)
			return t+1
		else:
			return self.step_cst_act(t)

	#evolve the network for time step t
	def step_cst_act(self,t):
		#event = interaction graph at t; first determine the edges to be activated again
		event = self.old_update(t)
		#determine the newborn edges
		possible_edges = list(self.adjacent_graph.edges)
		actual_nb_new = min(np.random.default_rng().poisson(self.nb_new,1)[0],len(possible_edges))
		if actual_nb_new>0:
			newborn = rd.sample(possible_edges,actual_nb_new)
			event.add_edges_from(newborn)
			self.born_update(t,newborn)
		###
		if not event:
			return t
		#pruning process: remove some parts of the social bond graph and modifies the adjacent graph
		#accordingly
		self.prune()
		#record the events and identify the activated nodes
		self.record_event(t,event)
		return t+1

class Dyn_EW(Gen_ATN):
	"""
	- the proba that an edge is active once more is equal to the fraction of times it has been active
	in the past
	- if the edge is created w=1 at t0, then P(w+1,t+1) = 1/(t-t0+2)
	- the nb of newborn edges per time is constant
	- frac = fraction of the nb of the total edges that has been activated after duree time steps
	(0<frac<1)
	- ext_act = external activity imposed for the TN (the nb of interactions at t is ext_act[t])
	"""
	def __init__(self,N,frac,ext_act):
		duree = len(ext_act)
		super(Dyn_EW,self).__init__(N,duree)
		self.ext_act = ext_act.copy()
		#social bond graph: gives the activation probability of old edges
		#social_graph[i][j]['weight'] = weight of i--j
		#social_graph[i][j]['birth'] = birth time of i--j
		self.social_graph = nx.Graph()
		#the adjacent graph gathers all the edges that can become newborn edges
		#it could be directed and weighted
		#initially it is the fully connected undirected graph
		self.adjacent_graph = nx.Graph()
		#ind_to_edge[ind] = (i,j) = edge of identifier ind, with i<j
		self.ind_to_edge = [(i,j) for i in range(N-1) for j in range(i+1,N)]
		self.adjacent_graph.add_edges_from(self.ind_to_edge)
		#the nb of newborn edges per time step follows a Poisson law of mean self.nb_new
		self.nb_new = frac*N*(N-1)/(2*duree)
		self.free_param = {}
		self.free_param['p_d'] = 2e-2

	def prune(self):
		nb_max = len(self.social_graph.edges)
		nb_to_remove = min(np.random.default_rng().poisson(self.free_param['p_d']*nb_max,1)[0],nb_max)
		if nb_to_remove>0:
			edge_to_remove = rd.sample(list(self.social_graph.edges),nb_to_remove)
			self.social_graph.remove_edges_from(edge_to_remove)
			self.adjacent_graph.add_edges_from(edge_to_remove)

	def draw_alive(self,t):
		event = nx.Graph()
		for edge in self.social_graph.edges:
			w = self.social_graph[edge[0]][edge[1]]['weight']
			if rd.random()<w/(t+1-self.social_graph[edge[0]][edge[1]]['birth']):
				event.add_edge(*edge)
		return event

	#aggregate the past interactions to update the social bond graph and the adjacent graph
	def hebbian_step(self,t,event):
		newborn = []
		for edge in event.edges:
			if self.social_graph.has_edge(*edge):
				self.social_graph[edge[0]][edge[1]]['weight'] += 1
			else:
				self.social_graph.add_edge(*edge,weight=1,birth=t)
				newborn.append(edge)
		self.adjacent_graph.remove_edges_from(newborn)

	#erase run data to be ready for a new run
	def refresh(self):
		self.activated[:] = 0
		self.events = []
		self.social_graph = nx.Graph()
		self.adjacent_graph = nx.Graph()
		self.adjacent_graph.add_edges_from(self.ind_to_edge)

	#evolve the network for time step t
	def step(self,t):
		#event = interaction graph at t
		#first determine the edges to be activated again
		event = self.draw_alive(t)
		#second determine the newborn edges
		possible_edges = list(self.adjacent_graph.edges)
		actual_nb_new = min(np.random.default_rng().poisson(self.nb_new,1)[0],len(possible_edges))
		if actual_nb_new>0:
			newborn = rd.sample(possible_edges,actual_nb_new)
			event.add_edges_from(newborn)

		if t>=self.t_min:
			#third remove or add additional interactions to match the required activity self.ext_act[t]
			if len(event.edges)<self.ext_act[t-self.t_min]:
				available = [edge for edge in self.ind_to_edge if not event.has_edge(*edge)]
				add = rd.sample(available,self.ext_act[t-self.t_min]-len(event.edges))
				event.add_edges_from(add)
			elif len(event.edges)>self.ext_act[t-self.t_min]:
				remove = rd.sample(list(event.edges),len(event.edges)-self.ext_act[t-self.t_min])
				event.remove_edges_from(remove)

		#fourth Hebbian step
		self.hebbian_step(t,event)
		#pruning step: remove some parts of the social bond graph
		#and modifies the adjacent graph accordingly
		self.prune()
		#sixth and last record the events and identify the activated nodes
		self.record_event(t,event)
		return t+1


class ADM_class(Gen_ATN):
	"""Revisited Activity Driven model with Memory
	basis version :
	p_g=1e-2,c=1,p_c=0.3,p_u=1e-3
	"""
	def __init__(self,readable_param,m='random',a='lognorm',update='alpha,i',context='neutral',c_ij=True,egonet_growth='cst',remove='edge'):
		super(ADM_class,self).__init__(readable_param['N'],readable_param['T'])
		#minimum value for alpha or beta
		self.alpha_min = 1e-3
		#maximum value for alpha or beta
		self.alpha_max = 1
		#parameters of the node activity distribution
		self.node_mu = readable_param['mu']
		self.node_sig = readable_param['sigma']
		#free parameters are determined by a genetic algo
		self.free_param = {}
		#save the initialization parameters to correctly refresh
		self.refresh_param = [m,a,update]
		#number of emitted interactions per active node
		if m=='random':
			self.free_param['m_max'] = 1
			self.m = []
			self.get_m = self.get_m_var
		elif m=='cst':
			self.get_m = self.get_m_cst
			self.free_param['m'] = 0
		#node activity
		if a=='lognorm':
			self.act = []
			self.get_act = self.get_act_var
			#minimum node activity
			self.a_min = readable_param['a_min']
			#maximum node activity
			self.a_max = readable_param['a_max']
		elif a=='power':
			self.act = []
			self.get_act = self.get_act_var
			self.free_param['a_min'] = 0
			self.free_param['a_max'] = 0
		elif a=='empirical':
			self.act = []
			self.get_act = self.get_act_var
		elif a=='cst':
			self.free_param['a'] = 0
			self.get_act = self.get_act_cst
		#rate of weight change for social bond ties
		if update=='alpha,i':
			self.get_alpha = self.get_alpha_i
			self.get_beta = self.get_alpha_i
			self.alpha = []
			self.update_egonet = self.update_egonet_Alpha
		elif update=='alpha':
			self.get_alpha = self.get_alpha_cst
			self.get_beta = self.get_alpha_cst
			self.free_param['alpha'] = 0
			self.update_egonet = self.update_egonet_Alpha
		elif update=='alpha,beta,i':
			self.get_alpha = self.get_alpha_i
			self.get_beta = self.get_beta_i
			self.alpha = []
			self.beta = []
			self.update_egonet = self.update_egonet_Alpha
		elif update=='alpha,beta,ij':
			self.get_alpha = self.get_alpha_ij
			self.get_beta = self.get_beta_ij
			self.alpha = []
			self.beta = []
			self.update_egonet = self.update_egonet_Alpha
		elif update=='linear':
			self.update_egonet = self.update_egonet_Linear
		elif update=='linear,decay':
			self.update_egonet = self.update_egonet_Linear_with_decay
		elif update=='no_memory':
			self.update_egonet = self.update_egonet_no_memory
		#choose the function computing the probability of growing the egonet
		if egonet_growth=='cst':
			self.grow_egonet = self.grow_egonet_cst
			self.free_param['p_g'] = 0
			self.free_param['p_u'] = 0
		elif egonet_growth=='var':
			self.grow_egonet = self.grow_egonet_var
			self.free_param['c'] = 0
		#choose the function computing the sets R_event and W_event :
		#how the weight of contextual interactions in the social bond graph should be updated
		if context=='equivalent':
			self.context = self.context_equivalent
			self.free_param['p_c'] = 0
			self.add_contint = self.add_contint_True
		elif context=='neutral':
			self.context = self.context_neutral
			self.free_param['p_c'] = 0
			self.add_contint = self.add_contint_True
		elif context=='noise':
			self.context = self.context_noise
			self.free_param['p_c'] = 0
			self.add_contint = self.add_contint_True
		elif context==None:
			self.context = self.context_equivalent
			self.add_contint = self.add_contint_False
		#choose the function removing ties from the social bond graph
		if remove=='edge':
			self.pruning = self.prune_edge
			self.free_param['lambda'] = 0
		elif remove=='node':
			self.pruning = self.prune_node
			self.free_param['p_d'] = 0
		#context memory : if True, the weight is modified by the number of common neighbours in the
		#interaction graph
		if c_ij==True:
			self.context_coeff = self.context_coeff_True
			self.compute_comngh = self.compute_comngh_True
		elif c_ij==False:
			self.context_coeff = self.context_coeff_False
			self.compute_comngh = self.compute_comngh_False
		#social bond graph (egonet connections) : directed weighted graph
		self.social_graph = nx.DiGraph()
		#commngh[ind(i,j)] = nb of common neighbours btw i and j at previous time step
		self.comngh = {}

	def power_law_act(self):
		return self.free_param['a_min']*(self.free_param['a_max']/self.free_param['a_min'])**rd.random()

	def power_law_alpha(self):
		return self.alpha_min*(self.alpha_max/self.alpha_min)**rd.random()

	def get_m_cst(self,i):
		return self.free_param['m']

	def get_m_var(self,i):
		return self.m[i]

	def get_act_cst(self,i):
		return self.free_param['a']

	def get_act_var(self,i):
		return self.act[i]

	def get_alpha_i(self,i,j):
		return self.alpha[i]

	def get_beta_i(self,i,j):
		return self.beta[i]

	def get_alpha_cst(self,i,j):
		return self.free_param['alpha']

	def get_alpha_ij(self,i,j):
		return self.alpha[i,j]

	def get_beta_ij(self,i,j):
		return self.beta[i,j]

	def grow_egonet_cst(self,i):
		return self.free_param['p_g']

	def grow_egonet_var(self,i):
		return self.free_param['c']/(self.free_param['c']+self.social_graph.out_degree(i))

	def coeff_ngh(self,i,k):
		ind = self.edge_to_ind(i,k)
		if ind in self.comngh:
			return 1+self.comngh[ind]
		return 1

	def context_coeff_False(self,i,nodes):
		x = rd.random()
		norm = sum([self.social_graph[i][k]['weight'] for k in nodes])
		s = 0; j = -1
		while s<x:
			j += 1
			s += self.social_graph[i][nodes[j]]['weight']/norm
		return nodes[j]

	#same as context_coeff_False
	#but the weight is modulated by the number of common ngh at previous step
	def context_coeff_True(self,i,nodes):
		x = rd.random()
		norm = sum([self.social_graph[i][k]['weight']*self.coeff_ngh(i,k) for k in nodes])
		s = 0; j = -1
		while s<x:
			j += 1
			s += self.social_graph[i][nodes[j]]['weight']*self.coeff_ngh(i,nodes[j])/norm
		return nodes[j]

	def add_contint_False(self,event,active_nodes):
		return nx.Graph()

	def add_contint_True(self,event,active_nodes):
		#browse the interaction graph event to collect open triangles
		#first collect nodes with degree >= 2
		centers = {node for node in event.nodes if event.degree(node)>=2}
		open_triangles = set(()); doubling_event = nx.Graph()
		for j in centers:
			#collect the neighbours of j that are not connected to each other
			#and such that at least one of them is active
			list_ngh = list(event[j])
			for ind1 in range(len(list_ngh)-1):
				i = list_ngh[ind1]
				for ind2 in range(ind1+1,len(list_ngh)):
					k = list_ngh[ind2]
					cond = (i in active_nodes) or (j in active_nodes)
					if cond and not event.has_edge(i,k):
						open_triangles.add((i,j,k))
		all_nodes = {node for el in open_triangles for node in el}
		norms = {}
		for q in all_nodes:
			norms[q] = sum([self.social_graph[q][l]['weight']*self.coeff_ngh(q,l) for l in self.social_graph[q]])
		#close the triangles
		for (i,j,k) in open_triangles:
			#p_ik = proba that i decides to close the triangle
			#p_ki = proba that k decides to close the triangle
			if i in active_nodes:
				if self.social_graph.has_edge(i,j):
					p_ij = self.social_graph[i][j]['weight']*self.coeff_ngh(i,j)/norms[i]
				else:
					p_ij = self.grow_egonet(i)
				if self.social_graph.has_edge(j,k):
					p_jk = self.social_graph[j][k]['weight']*self.coeff_ngh(j,k)/norms[j]
				else:
					p_jk = self.grow_egonet(j)
				x = p_ij*p_jk*self.free_param['p_c']
				norm = (1 + x*(self.coeff_ngh(i,k) - 1))
				p_ik = x*self.coeff_ngh(i,k)/norm
			else:
				p_ik = 0

			if k in active_nodes:
				if self.social_graph.has_edge(k,j):
					p_kj = self.social_graph[k][j]['weight']*self.coeff_ngh(j,k)/norms[k]
				else:
					p_kj = self.grow_egonet(k)
				if self.social_graph.has_edge(j,i):
					p_ji = self.social_graph[j][i]['weight']*self.coeff_ngh(i,j)/norms[j]
				else:
					p_ji = self.grow_egonet(j)
				x = p_kj*p_ji*self.free_param['p_c']
				norm = (1 + x*(self.coeff_ngh(i,k) - 1))
				p_ki = x*self.coeff_ngh(i,k)/norm
			else:
				p_ki = 0

			if rd.random()<1-(1-p_ik)*(1-p_ki):
				doubling_event.add_edge(i,k)
		return doubling_event

	#i can emit self.m links at most
	#at each try, he creates a new link with proba self.eps
	#or reactivates a known link
	#Each succesful interaction is accompanied by a bonus interaction (proba self.double)
	#with a 2nd order ngh.
	def play(self,i):
		inst_ngh = set() #nodes with whom i succeeds to interact
		unknown_nodes = [k for k in range(self.N) if not self.social_graph.has_edge(i,k) and k!=i]
		known_nodes = [k for k in self.social_graph[i]]
		#i tries to emit self.get_m(i) intentional interactions
		for _ in range(self.get_m(i)):
			#if the egonet is empty, choose a partner uniformly at random
			if self.social_graph.out_degree(i)==0 and unknown_nodes:
				j = rd.choice(unknown_nodes)
				inst_ngh.add(j)
			#grow the egonet, if possible
			elif rd.random()<self.grow_egonet(i) and unknown_nodes:
				#choose an unknown node uniformly at random
				if 'p_u' in self.free_param:
					cond = (rd.random()<self.free_param['p_u'])
				else:
					cond = False
				if cond:
					j = rd.choice(unknown_nodes)
					inst_ngh.add(j)
				#choose an unkwown node by triadic closure, if possible
				else:
					#choose a 1st order ngh
					j = self.context_coeff_False(i,list(self.social_graph[i]))
					#choose a 2nd order ngh
					nodes = [k for k in self.social_graph[j] if k!=i and k not in inst_ngh and not self.social_graph.has_edge(i,k)]
					if nodes:
						inst_ngh.add(self.context_coeff(j,nodes))
					else:
						j = rd.choice(unknown_nodes)
						inst_ngh.add(j)
			#interact with a known node, if possible
			elif known_nodes:
				j = self.context_coeff(i,known_nodes)
				inst_ngh.add(j)
			#update known and unknown nodes
			unknown_nodes = [k for k in unknown_nodes if k not in inst_ngh]
			known_nodes = [k for k in known_nodes if k not in inst_ngh]
		return inst_ngh

	def prune_edge(self,R_event,W_event):
		edge_to_remove = set()
		for i in R_event:
			m = self.get_m(i)
			#number of neighbours of i
			d = self.social_graph.out_degree(i)
			norm = self.social_graph.out_degree(i,weight='weight')
			for j in self.social_graph[i]:
				if not W_event.has_edge(i,j):
					#proba that i selects j in the abscence of social context
					p = self.social_graph[i][j]['weight']/norm
					#if all partners of i have equal weight, their individual proba of activation
					#given i is in the active state, is 1/d
					if p<=0:
						edge_to_remove.add((i,j))
					elif d>m:
						if rd.random()<exp(-p*d*self.free_param['lambda']):
							edge_to_remove.add((i,j))
		self.social_graph.remove_edges_from(edge_to_remove)
		self.social_graph.add_nodes_from(range(self.N))

	def prune_node(self,R_event,W_event):
		zero_ties = []
		#first remove the zero ties
		for i in R_event:
			for j in self.social_graph[i]:
				if not W_event.has_edge(i,j):
					if self.social_graph[i][j]['weight']<=0:
						zero_ties.append((i,j))
		self.social_graph.remove_edges_from(zero_ties)
		self.social_graph.add_nodes_from(set(range(self.N)))
		#second remove nodes at random
		nb_sup = min(np.random.poisson(lam=self.free_param['p_d']*self.N),self.N)
		removed_nodes = rd.sample(range(self.N),nb_sup)
		self.social_graph.remove_nodes_from(removed_nodes)
		self.social_graph.add_nodes_from(removed_nodes)

	#update self.social_graph
	#R_event are the events which participate to the egonet edges reinforcement
	#edges in W_event cannot be weakened
	#idea : initialize weight with random weights or with transitive weight, i.e. for
	#(i,k) resulting from triadic closure of (i,j) and (j,k) : w_{i,k} = w_{i,j}*w_{j,k}
	def update_egonet_Alpha(self,R_event,W_event):
		#ties reinforcement
		for i,j in R_event.edges:
			for k,l in zip([i,j],[j,i]):
				alpha = self.get_alpha(k,l)
				if self.social_graph.has_edge(k,l):
					self.social_graph[k][l]['weight'] += alpha*(1-self.social_graph[k][l]['weight'])
				else:
					self.social_graph.add_edge(k,l,weight=alpha)
		#ties decay
		for i in R_event:
			for j in self.social_graph[i]:
				if not W_event.has_edge(i,j):
					self.social_graph[i][j]['weight'] -= self.get_beta(i,j)*self.social_graph[i][j]['weight']

	#update self.social_graph in case of a linear reinforcement process
	#and a linear decay process
	def update_egonet_Linear_with_decay(self,R_event,W_event):
		#ties reinforcement
		for i,j in R_event.edges:
			for k,l in zip([i,j],[j,i]):
				if self.social_graph.has_edge(k,l):
					self.social_graph[k][l]['weight'] += 1
				else:
					self.social_graph.add_edge(k,l,weight=1)
		#ties decay
		for i in R_event:
			for j in self.social_graph[i]:
				if not W_event.has_edge(i,j):
					self.social_graph[i][j]['weight'] -= 1

	#update self.social_graph in case of no memory: all social weights have a value of 1
	def update_egonet_no_memory(self,R_event,W_event):
		for i,j in R_event.edges:
			for k,l in zip([i,j],[j,i]):
				if not self.social_graph.has_edge(k,l):
					self.social_graph.add_edge(k,l,weight=1)

	#update self.social_graph in case of a linear reinforcement process
	#and no linear decay process
	def update_egonet_Linear(self,R_event,W_event):
		#ties reinforcement
		for i,j in R_event.edges:
			for k,l in zip([i,j],[j,i]):
				if self.social_graph.has_edge(k,l):
					self.social_graph[k][l]['weight'] += 1
				else:
					self.social_graph.add_edge(k,l,weight=1)

	def context_equivalent(self,event,tot_event):
		return tot_event,tot_event

	def context_neutral(self,event,tot_event):
		return event,tot_event

	def context_noise(self,event,tot_event):
		return event,event

	#compute self.comngh
	def compute_comngh_True(self,event):
		self.comngh = {}
		active_nodes = list(event.nodes)
		nb_nodes = len(active_nodes)
		for ind1 in range(nb_nodes-1):
			i = active_nodes[ind1]; ngh1 = set(event[i])
			for ind2 in range(ind1+1,nb_nodes):
				j = active_nodes[ind2]
				ngh2 = set(event[j])
				self.comngh[self.edge_to_ind(i,j)] = len(ngh1.intersection(ngh2))

	#compute self.comngh
	def compute_comngh_False(self,event):
		pass

	#evolve the network for time step t
	def step(self,t):
		active_nodes = set(())
		for i in range(self.N):
			if rd.random()<self.get_act(i):
				active_nodes.add(i)
		#if there is no node active, no interaction can occur
		if len(active_nodes)==0:
			return t
		#graph of intentional interactions
		event = nx.Graph()
		for i in active_nodes:
			inst_ngh = self.play(i)
			for j in inst_ngh:
				event.add_edge(i,j)
		if not event:
			return t
		#add contextual interactions (dynamic closure)
		doubling_event = self.add_contint(event,active_nodes)
		tot_event = nx.Graph()
		tot_event.add_edges_from(event.edges)
		tot_event.add_edges_from(doubling_event.edges)
		#update self.egonet
		R_event,W_event = self.context(event,tot_event)
		#Hebbian step
		self.update_egonet(R_event,W_event)
		#pruning step
		self.pruning(R_event,W_event)
		#compute self.comngh if necessary
		self.compute_comngh(tot_event)
		#record the events and identify the activated nodes
		self.record_event(t,tot_event)
		return t+1

	#erase run data to be ready for a new run
	def refresh(self):
		self.events = []
		self.activated[:] = 0
		self.social_graph = nx.DiGraph()
		self.social_graph.add_nodes_from(range(self.N))
		self.comngh = {}
		m,a,update = self.refresh_param
		if m=='random':
			self.m = np.random.randint(1,self.free_param['m_max']+1,self.N)
		if a=='lognorm':
			#draw N realizations of a Gaussian variable of parameters node_mu and node_sig
			#truncated between np.log10(a_min) and np.log10(a_max)
			start = np.log10(self.a_min); end = np.log10(self.a_max)
			aa,bb = (start-self.node_mu)/self.node_sig,(end-self.node_mu)/self.node_sig
			tab = truncnorm.rvs(aa,bb,loc=self.node_mu,scale=self.node_sig,size=self.N)
			self.act = np.power(10,tab)
		elif a=='empirical':
			#draw N realizations from the empirical distribution for the observed activities
			pass
		elif a=='power':
			if self.free_param['a_min']>self.free_param['a_max']:
				self.free_param['a_min'],self.free_param['a_max'] = self.free_param['a_max'],self.free_param['a_min']
			self.act = [self.power_law_act() for _ in range(self.N)]
		if update=='alpha,i':
			self.alpha = [self.power_law_alpha() for _ in range(self.N)]
		elif update=='alpha,beta,i':
			self.alpha = [self.power_law_alpha() for _ in range(self.N)]
			self.beta = [self.power_law_alpha() for _ in range(self.N)]
		elif update=='alpha,beta,ij':
			self.alpha = np.asarray([[self.power_law_alpha() for i in range(self.N)] for j in range(self.N)])
			self.beta = np.asarray([[self.power_law_alpha() for i in range(self.N)] for j in range(self.N)])
