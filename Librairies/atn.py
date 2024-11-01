import numpy as np
from math import *
import random as rd
import networkx as nx
#from scipy.stats import truncnorm

#return a number drawn from the power law with fixed bounds and exponent -gamma (P(x)=x^{-gamma})
def Power_law(bounds,gamma,size=1):
	if gamma==1:
		return bounds[0]*(bounds[1]/bounds[0])**np.random.random(size)
	return bounds[0]*(1 + np.random.random(size)*((bounds[1]/bounds[0])**(1-gamma) - 1))**(1/(1-gamma))

class Gen_ATN:
	"""any constructor of temporal networks inherits from this class"""
	def __init__(self,N,duree):
		self.N = N
		self.duree = duree
		self.events = []
		self.activated = np.zeros(N,dtype=int)
		self.t_min = 0

	def Edge_to_ind(self,i,j):
		if i>j:
			i,j = j,i
		return i*(2*self.N-i-1)//2 + j-i-1

	def Evolve(self,t_min=0,max_count=3,max_locount=5):
		self.t_min = t_min
		#nb of nodes that have activated during the run
		nb_nodes = 0; reinitialize = False; count = 0
		while nb_nodes<self.N and count<max_count:
			#reinitialize the TN
			if reinitialize:
				self.Refresh()
			#start the time flow
			t = 0; loc_count = 0
			while t<self.duree+self.t_min and loc_count<max_locount:
				if t*100%(self.duree+self.t_min)==0:
					print(str(t*100//(self.duree+self.t_min))+" % computed")
				newt = self.Step(t)
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
	def Record_event(self,t,event):
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

class Life_game:
	"""simulate the game of life and convert the grid world into a temporal network, the intuition behind
	is that the game of life is Turing complete and that this property may account for many of the fractal
	behaviour of empirical temporal networks.
	The world is a torus.
	"""
	def __init__(self,N,first_image=None):
		#game of life variables
		self.N = N
		self.size = floor(sqrt(N))
		if first_image is not None:
			self.grid = first_image
		else:
			self.grid = np.zeros((self.size,self.size),dtype=int)
		#TN variables
		self.activated = np.zeros(self.size,dtype=int)
		self.events = []

	#update the grid state
	def Update(self):
		new_grid = np.zeros(self.grid.shape,dtype=int)
		for i in range(self.size):
			for j in range(self.size):
				#count the nb of neighbors
				nb_ngh = np.sum(self.grid[i-1:(i+2)%self.size,j-1:(j+2)%self.size])-self.grid[i,j]
				new_grid[i,j] = int(nb_ngh==3 or (self.grid[i,j]==1 and nb_ngh==2))
		self.grid[:,:] = new_grid[:,:]

	def Refresh(self,first_image):
		self.activated = np.zeros(self.size,dtype=int)
		self.events = []
		self.grid = first_image

	#convert the grid state into an interaction graph, the idea being that the resulting graph has at much
	#info as possible about the grid state
	#first possibility: nodes are living cells; interaction iif adjacence
	#second possibility: we introduce a detection radius and a proba of interaction btw two living cells
	#that decreases with distance, taking the Manhattan metric
	#other possibility: we consider the whole grid as the adjacency matrix of the interaction graph but
	#we have to symmetrize it and the nb of nodes becomes self.size instead of self.N
	#to symmetrize, some possibilities are: (1) G_i,j = grid[i,j] xor grid[j,i], which automatically
	#removes the self-loops (2) G_i,j = grid[i,j] or grid[j,i] (3) grid[i,j] and grid[j,i] (4) etc
	def Grid_to_net(self):
		net = nx.Graph()
		for i in range(self.size-1):
			for j in range(i+1,self.size):
				if self.grid[i,j]^self.grid[j,i]:
					net.add_edge(i,j)
		return net

	def Step_no_recording(self,t):
		event = self.Grid_to_net()
		self.Update()
		if not event:
			return t
		return t+1

	def Step(self,t):
		event = self.Grid_to_net()
		self.Update()
		if not event:
			return t
		for i in event.nodes:
			self.activated[i] = 1
		#update self.events
		for edge in event.edges:
			self.events.append([t,*edge])
		return t+1

	#obtain the temporal network from Game of Life dynamics
	#starting_time is the time needed to achieve stationary dynamics
	#data recording begins only after that time
	def Evolve(self,duree,max_locount=5,starting_time=0):
		#wait for the first interaction
		t = 0; loc_count = 0
		while t==0 and loc_count<max_locount:
			t = self.Step_no_recording(t)
			if t==0:
				loc_count += 1
		if loc_count==max_locount:
			return None
		#start the time flow
		while t<starting_time and loc_count<max_locount:
			if t*100%(duree+starting_time)==0:
				print(str(t*100//(duree+starting_time))+" % computed")
			newt = self.Step_no_recording(t)
			if newt==t:
				loc_count += 1
			else:
				t = newt
				loc_count = 0
		#start the time flow
		while t<duree+starting_time and loc_count<max_locount:
			if t*100%(duree+starting_time)==0:
				print(str(t*100//(duree+starting_time))+" % computed")
			newt = self.Step(t)
			if newt==t:
				loc_count += 1
			else:
				t = newt
				loc_count = 0
		print('actual duration of the TN:',t-starting_time)
		print('actual nb of agents:',np.sum(self.activated))
		return np.asarray(self.events,dtype=int)

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
			self.Old_update = self.Shift_old_update
			self.Born_update = self.Shift_born_update
		else:
			self.Old_update = self.No_shift_old_update
			self.Born_update = self.No_shift_born_update
		if removal=='node_unif':
			self.free_param['p_d'] = 1e-2
			self.Prune = self.Prune_node_unif
		elif removal=='edge_unif':
			self.free_param['p_d'] = 2e-2
			self.Prune = self.Prune_edge_unif
		else:
			self.Prune = self.Prune_none
		if var_act:
			self.Step = self.Step_var_act
		else:
			self.Step = self.Step_cst_act

	def Prune_none(self):
		pass
	def Prune_node_unif(self):
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
	def Prune_edge_unif(self):
		nb_max = len(self.social_graph.edges)
		nb_to_remove = min(np.random.default_rng().poisson(self.free_param['p_d']*nb_max,1)[0],nb_max)
		if nb_to_remove>0:
			edge_to_remove = rd.sample(list(self.social_graph.edges),nb_to_remove)
			self.social_graph.remove_edges_from(edge_to_remove)
			self.adjacent_graph.add_edges_from(edge_to_remove)
	def No_shift_old_update(self,t):
		event = nx.Graph()
		for edge in self.social_graph.edges:
			w = self.social_graph[edge[0]][edge[1]]['weight']
			if rd.random()<w/(t+1):
				event.add_edge(*edge)
				self.social_graph[edge[0]][edge[1]]['weight'] += 1
		return event
	def No_shift_born_update(self,t,newborn):
		for edge in newborn:
			self.social_graph.add_edge(*edge,weight=1)
		self.adjacent_graph.remove_edges_from(newborn)
	def Shift_old_update(self,t):
		event = nx.Graph()
		for edge in self.social_graph.edges:
			w = self.social_graph[edge[0]][edge[1]]['weight']
			if rd.random()<w/(t+1-self.social_graph[edge[0]][edge[1]]['birth']):
				event.add_edge(*edge)
				self.social_graph[edge[0]][edge[1]]['weight'] += 1
		return event
	def Shift_born_update(self,t,newborn):
		for edge in newborn:
			self.social_graph.add_edge(*edge,weight=1,birth=t)
		self.adjacent_graph.remove_edges_from(newborn)

	#erase run data to be ready for a new run
	def Refresh(self):
		self.activated[:] = 0
		self.events = []
		self.social_graph = nx.Graph()
		self.adjacent_graph = nx.Graph()
		self.adjacent_graph.add_edges_from(self.ind_to_edge)

	#evolve the network for time step t
	def Step_var_act(self,t):
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
			self.Born_update(t,newborn)
			self.Prune()
			#record the events and identify the activated nodes
			event = nx.Graph(); event.add_edges_from(active_edges)
			self.Record_event(t,event)
			return t+1
		else:
			return self.Step_cst_act(t)

	#evolve the network for time step t
	def Step_cst_act(self,t):
		#event = interaction graph at t; first determine the edges to be activated again
		event = self.Old_update(t)
		#determine the newborn edges
		possible_edges = list(self.adjacent_graph.edges)
		actual_nb_new = min(np.random.default_rng().poisson(self.nb_new,1)[0],len(possible_edges))
		if actual_nb_new>0:
			newborn = rd.sample(possible_edges,actual_nb_new)
			event.add_edges_from(newborn)
			self.Born_update(t,newborn)
		###
		if not event:
			return t
		#pruning process: remove some parts of the social bond graph and modifies the adjacent graph
		#accordingly
		self.Prune()
		#record the events and identify the activated nodes
		self.Record_event(t,event)
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

	def Prune(self):
		nb_max = len(self.social_graph.edges)
		nb_to_remove = min(np.random.default_rng().poisson(self.free_param['p_d']*nb_max,1)[0],nb_max)
		if nb_to_remove>0:
			edge_to_remove = rd.sample(list(self.social_graph.edges),nb_to_remove)
			self.social_graph.remove_edges_from(edge_to_remove)
			self.adjacent_graph.add_edges_from(edge_to_remove)

	def Draw_alive(self,t):
		event = nx.Graph()
		for edge in self.social_graph.edges:
			w = self.social_graph[edge[0]][edge[1]]['weight']
			if rd.random()<w/(t+1-self.social_graph[edge[0]][edge[1]]['birth']):
				event.add_edge(*edge)
		return event

	#aggregate the past interactions to update the social bond graph and the adjacent graph
	def Hebbian_step(self,t,event):
		newborn = []
		for edge in event.edges:
			if self.social_graph.has_edge(*edge):
				self.social_graph[edge[0]][edge[1]]['weight'] += 1
			else:
				self.social_graph.add_edge(*edge,weight=1,birth=t)
				newborn.append(edge)
		self.adjacent_graph.remove_edges_from(newborn)

	#erase run data to be ready for a new run
	def Refresh(self):
		self.activated[:] = 0
		self.events = []
		self.social_graph = nx.Graph()
		self.adjacent_graph = nx.Graph()
		self.adjacent_graph.add_edges_from(self.ind_to_edge)

	#evolve the network for time step t
	def Step(self,t):
		#event = interaction graph at t
		#first determine the edges to be activated again
		event = self.Draw_alive(t)
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
		self.Hebbian_step(t,event)
		#pruning step: remove some parts of the social bond graph
		#and modifies the adjacent graph accordingly
		self.Prune()
		#sixth and last record the events and identify the activated nodes
		self.Record_event(t,event)
		return t+1

class Toy_egonet:
	"""all nodes have same activity in the ADM sense, they also emit the same number of
	intentional interactions (m), they have no explicit memory but they have a fixed egonet.
	The distribution of the egonet size across nodes is fixed.
	We want to be able to impose the instantaneous node degree distribution, as well as
	the instantaneous total edge activity distribution.
	"""
	def __init__(self,N,m,a,duree):
		self.activated = np.zeros(N,dtype=int)
		self.social_graph = nx.Graph()
		self.events = []
		self.N = N
		self.m = m
		self.a = a
		self.duree = duree
		#degree sequence we want to impose on the social bond graph
		self.deg_seq = np.ones(self.N,dtype=int)
		#save the choice of degree sequence
		self.choice = None

	#compute the degree sequence (i.e. the sequence of egonet sizes)
	#choice is of the form ('type',parameters)
	def Compute_deg_seq(self,choice):
		#uniform degree
		if choice[0]=='cst':
			self.deg_seq = [choice[1]]*self.N
		#power law P(B) \propto B^{-\gamma}; \gamma = choice[1]
		elif choice[0]=='power':
			self.deg_seq = [int(np.round(el)) for el in Power_law((1,self.N-1),choice[1],size=self.N)]
		#geometric distribution P(B) \propto (1-p)^{B-1}*p; B_c = choice[1]
		#we have B_c = \frac{-1}{\log(1-p)} so p = 1-\exp(-\frac{1}{B_c})
		elif choice[0]=='exp':
			p = 1-exp(-1/float(choice[1]))
			self.deg_seq = np.random.default_rng().geometric(p=p,size=self.N)
		#Poisson law
		elif choice[0]=='poisson':
			self.deg_seq = np.random.default_rng().poisson(choice[1],self.N)
		for ind in range(self.N):
			deg = self.deg_seq[ind]
			if deg==0:
				self.deg_seq[ind] = 1
			elif deg>self.N-1:
				self.deg_seq[ind] = self.N-1
		self.choice = choice
		if np.sum(self.deg_seq)%2==1:
			ind = 0
			while self.deg_seq[ind]==self.N-1:
				ind += 1
			self.deg_seq[ind] += 1

	#compute the egonets of each node such that the degree sequence is close to self.deg_seq
	def Compute_egonet(self):
		self.social_graph = nx.configuration_model(self.deg_seq)
		self.social_graph = nx.Graph(self.social_graph)
		self.social_graph.remove_edges_from(nx.selfloop_edges(self.social_graph))

	#erase run data to be ready for a new run
	def Refresh(self):
		self.activated[:] = 0
		self.events = []
		self.Compute_deg_seq(self.choice)
		self.Compute_egonet()

	def Play(self,i):
		inst_ngh = set([]) #nodes with whom i succeeds to interact
		known_nodes = [k for k in self.social_graph[i]]
		nb = min(self.m,len(known_nodes))
		return set(rd.sample(known_nodes,nb))

	#evolve the network for time step t
	def Step(self,t):
		nb_act = min(np.random.poisson(lam=self.a*self.N),self.N)
		active_nodes = set(rd.sample(range(self.N),nb_act))
		#if there is no node active, no interaction can occur
		if len(active_nodes)==0:
			return t
		#graph of intentional interactions
		event = nx.Graph()
		for i in active_nodes:
			inst_ngh = self.Play(i)
			for j in inst_ngh:
				event.add_edge(i,j)
		if not event:
			return t
		#update the activated nodes
		for i in event.nodes:
			self.activated[i] = 1
		#update self.events
		for edge in event.edges:
			self.events.append([t,*edge])
		return t+1

	def Evolve(self,max_count=3,max_locount=5):
		#nb of nodes that have activated during the run
		nb_nodes = 0; reinitialize = False; count = 0
		while nb_nodes<self.N and count<max_count:
			#reinitialize the TN
			if reinitialize:
				self.Refresh()
			#wait for the first interaction
			t = 0; loc_count = 0
			while t==0 and loc_count<max_locount:
				t = self.Step(t)
				if t==0:
					loc_count += 1
			if loc_count==max_locount:
				reinitialize = True; count += 1
			else:
				#start the time flow
				while t<self.duree and loc_count<max_locount:
					newt = self.Step(t)
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
			return np.asarray(self.events,dtype=int)
		else:
			return None

#minimal toy model reproducing empirical distributions for edge activity, interactivity
#and aggregated weight distribution
class Toy_modelV3:
	"""At each time step, nb_new newborn edges are drawn uniformly at random.
	We assume that each node is sensitive to the global activity level, which translates
	in a reinforcement process for the edge activity level :
	Then the number nb_act(t) of edges activated among the not newborn edges at time t
	satisfies :
	nb_act(t)-nb_act(t-1) = 1 or -1 with proba following a reinforcement process
	Not newborn edges are chosen according statistical weights B_{i,j}.
	"""
	def __init__(self,N,nb_new,delta):
		#nb of social agents
		self.N = N
		#events[k] = [t_k,i_k,j_k] = k^th interaction
		self.events = []
		#ind_to_edge[ind] = (i,j) = edge of identifier ind, with i<j
		self.ind_to_edge = []
		for i in range(N-1):
			for j in range(i+1,N):
				self.ind_to_edge.append((i,j))
		#reinforcement rate for social bonds
		self.delta = delta
		#number of newborn edges per time step
		self.nb_new = nb_new
		#number of active not newborn edges at current time step
		self.nb_act = 10
		#born = set of edges already active once
		self.born = set(())
		#bond[i][j]['weight'] = social bond from i to j
		self.bond = nx.Graph()
		
	def Edge_to_ind(self,i,j):
		if i>j:
			i,j = j,i
		return i*(2*self.N-i-1)//2 + j-i-1

	#evolve the network for time step t
	def Step(self,t):
		event = nx.Graph()
		#first determine the edge activity
		#note that edge activity cannot vanish or exceed the network capacity
		self.nb_act = max(1,np.random.poisson(lam=self.nb_act))
		self.nb_act = min(self.nb_act,self.N*(self.N-1)//2)
		#draw the active edges
		active_edges = []
		if len(self.born)>self.nb_act:
			vec = []; proba = []; s = 0
			for ind in self.born:
				i,j = self.ind_to_edge[ind]
				vec.append(ind)
				proba.append(self.bond[i][j]['weight'])
				s += self.bond[i][j]['weight']
			for k in range(len(proba)):
				proba[k] /= s
			active_edges = np.random.choice(vec,size=self.nb_act,replace=False,p=proba)
			active_edges = [self.ind_to_edge[ind] for ind in active_edges]
		elif len(self.born)>0:
			active_edges = [self.ind_to_edge[ind] for ind in self.born]
		for i,j in active_edges:
			event.add_edge(i,j)
			self.bond[i][j]['weight'] += self.delta

		#draw the newborn edges
		available = [ind for ind in range(self.N*(self.N-1)//2) if ind not in self.born]
		#add the newborn edges to event and self.born_active
		if available:
			newborn = rd.sample(available,self.nb_new)
			for ind in newborn:
				event.add_edge(*self.ind_to_edge[ind])
				self.born.add(ind)
				i,j = self.ind_to_edge[ind]
				if self.bond.has_edge(i,j):
					self.bond[i][j]['weight'] += self.delta
				else:
					self.bond.add_edge(i,j,weight=self.delta)	
		#record the events
		for edge in event.edges:
			self.events.append([t,*edge])
		return t+1

	#produce a TN of duration duree and save it
	def Evolve(self,duree,name):
		#wait for the first interaction
		t = 0
		while t==0:
			t = self.Step(t)
		#start the time flow
		while t<duree:
			t = self.Step(t)
			if t*100%duree==0:
				print(str(t*100//duree)+" % computed")
		#print the effective number of nodes
		all_nodes = set([])
		for line in self.events:
			all_nodes.add(line[1])
			all_nodes.add(line[2])
		print('effective number of nodes :',len(all_nodes))
		dic_nodes = {node:i for i,node in enumerate(all_nodes)}
		correvent = np.asarray(self.events)
		for n in range(len(correvent)):
			for k in [1,2]:
				correvent[n,k] = dic_nodes[self.events[n][k]]
		#save the events data
		np.savetxt('data/atn/'+name+'.txt',correvent,fmt='%d')

class Avalanche:
	"""Avalanche Temporal Network
	Each descendent node will try to establish n links, where n follows a Poisson
	law of parameter equal to the degree of their parent.
	At each time step we select n ancestors, n following a Poisson law. Then there are two ways
	to select the n ancestors among all the possible nodes, knowing that the activity and
	interactivity distributions for nodes are power laws.
	Way 1 : to each possible node we associate its social bond out-degree d_{i}(t) = \sum_{j} w_{i,j}(t)
	Way 2 : to each node we associate a variable a_{i}(t) such that a_{i} increases if i is active at t
	and decreases else.
	In either way, ancestors are drawn from the possible nodes with the non-uniform distribution
	generated by d_{i}(t) or a_{i}(t).
	To mimick an external perturbation likely to account for the periods of high edge activity
	we stimulate the TN at each time with a small probability (excitatory perturbation) as follows :
	we randomly activate groups of size larger than 2 among the inactive nodes
	"""
	def __init__(self,N,nb_ancestor,eps,p_random,p_stimulus):
		#nb of ancestors at previous time step
		self.prev_anc = 0
		#probability of excitatory stimulation
		self.p_stimulus = p_stimulus
		#commngh[ind(i,j)] = nb of common neighbours btw i and j at previous time step
		self.comngh = {}
		#act[i] = potential for i to be an ancestor
		self.act = np.zeros(N)
		self.alpha_act = np.zeros(N)
		#proba, when growing the egonet, to choose a partner at random
		self.p_random = p_random
		#proba of growing the egonet
		self.eps = eps
		#average nb of ancestors popping per time step
		#this nb follows a Poisson law
		self.nb_anc = nb_ancestor
		#nb of social agents
		self.N = N
		#descendents[i] = instantaneous degree of i (for i st if it is > 0)
		self.descendents = {}
		#egonet[i][j]['weight'] = social bond from i to j
		self.egonet = nx.DiGraph()
		self.egonet.add_nodes_from(range(N))
		self.alpha = np.zeros(N)
		for i in range(N):
			self.alpha[i] = Power_law(-1,1e-3)
			#self.alpha_act[i] = self.alpha[i]
		#events[k] = [t_k,i_k,j_k] = k^th interaction
		self.events = []
		#ind_to_edge[ind] = (i,j) = edge of identifier ind, with i<j
		self.ind_to_edge = []
		for i in range(N-1):
			for j in range(i+1,N):
				self.ind_to_edge.append((i,j))

	def Edge_to_ind(self,i,j):
		if i>j:
			i,j = j,i
		return i*(2*self.N-i-1)//2 + j-i-1

	#compute self.comngh
	def Compute_comngh(self,event):
		self.comngh = {}
		active_nodes = list(event.nodes)
		nb_nodes = len(active_nodes)
		for ind1 in range(nb_nodes-1):
			i = active_nodes[ind1]; ngh1 = set(event[i])
			for ind2 in range(ind1+1,nb_nodes):
				j = active_nodes[ind2]
				ngh2 = set(event[j])
				self.comngh[self.Edge_to_ind(i,j)] = len(ngh1.intersection(ngh2))

	def Choose(self,i,nodes):
		x = rd.random()
		norm = sum([self.egonet[i][k]['weight'] for k in nodes])
		s = 0; j = -1
		while s<x:
			j += 1
			s += self.egonet[i][nodes[j]]['weight']/norm
		return nodes[j]

	def Coeff_ngh(self,i,k):
		ind = self.Edge_to_ind(i,k)
		if ind in self.comngh:
			return 1+self.comngh[ind]
		return 1

	#same as Choose, but the weight is modulated by the number of common ngh at previous step
	def Choose_ngh(self,i,nodes):
		x = rd.random()
		norm = sum([self.egonet[i][k]['weight']*self.Coeff_ngh(i,k) for k in nodes])
		s = 0; j = -1
		while s<x:
			j += 1
			s += self.egonet[i][nodes[j]]['weight']*self.Coeff_ngh(i,nodes[j])/norm
		return nodes[j]

	#proba of removing a node j from self.egonet[i] with
	#w_{i,j} = w and len(self.egonet[i]) = d
	def P_removal(self,w,d,i):
		return exp(-w*d/self.egonet.out_degree(i,weight='weight'))

	#draw one node from available
	#with P(i) \propto self.act[i]
	def Draw(self,available):
		dic = {i:self.act[i] for i in available}
		norm = sum(list(dic.values()))
		if norm==0:
			return rd.choice(list(dic.keys()))
		x = rd.random()
		s = 0; j = -1
		keys = list(dic.keys())
		while s<x:
			j += 1
			s += dic[keys[j]]/norm
		return keys[j]

	#i tries to establish m interactions
	def Play(self,i,m):
		inst_ngh = set([]) #nodes with whom i succeeds to interact
		unknown_nodes = [k for k in range(self.N) if not self.egonet.has_edge(i,k) and k!=i]
		known_nodes = [k for k in self.egonet[i]]
		#i tries to emit m intentional interactions
		for _ in range(m):
			choice = rd.random()
			cond1 = (choice<self.eps) and unknown_nodes
			cond2 = (choice>=self.eps) and known_nodes
			#add a node to the egonet
			if cond1 or not known_nodes:
				#choose a ngh at random
				if rd.random()<self.p_random or self.egonet.out_degree(i)==0:
					j = rd.choice(unknown_nodes)
					inst_ngh.add(j)
				#choose a ngh wisely
				else:
					#choose a 1st order ngh
					j = self.Choose(i,list(self.egonet[i]))
					#choose a 2nd order ngh
					nodes = [k for k in self.egonet[j] if k!=i and k not in inst_ngh]
					if nodes:
						inst_ngh.add(self.Choose_ngh(j,nodes))
					else:
						inst_ngh.add(rd.choice(unknown_nodes))
			#interact with a node from the egonet
			if cond2 or not unknown_nodes:
				#choose an active 1st order ngh to interact with
				j = self.Choose_ngh(i,known_nodes)
				inst_ngh.add(j)
			#update the known and unknown nodes
			unknown_nodes = [k for k in unknown_nodes if k not in inst_ngh]
			known_nodes = [k for k in known_nodes if k not in inst_ngh]
		return inst_ngh

	#ancestor i selects a partner among available nodes to establish a new lineage
	def Anc_play(self,i,available):
		unknown_nodes = [k for k in available if not self.egonet.has_edge(i,k)]
		known_nodes = [k for k in available if k in self.egonet[i]]
		choice = rd.random()
		cond1 = (choice<self.eps) and unknown_nodes
		cond2 = (choice>=self.eps) and known_nodes
		#add a node to the egonet
		if cond1 or not known_nodes:
			#choose a ngh at random
			if rd.random()<self.p_random or self.egonet.out_degree(i)==0:
				return rd.choice(unknown_nodes)
			#choose a ngh wisely
			else:
				#choose a 1st order ngh
				j = self.Choose(i,list(self.egonet[i]))
				#choose a 2nd order ngh
				nodes = [k for k in self.egonet[j] if k in available]
				if nodes:
					return self.Choose_ngh(j,nodes)
				else:
					return rd.choice(unknown_nodes)
		#interact with a node from the egonet
		if cond2 or not unknown_nodes:
			#choose an active 1st order ngh to interact with
			return self.Choose_ngh(i,known_nodes)

	#update self.egonet and self.act
	#R_event are the events which participate to the egonet edges reinforcement
	def Update_egonet(self,R_event):
		for i,j in R_event.edges:
			for k,l in zip([i,j],[j,i]):
				if self.egonet.has_edge(k,l):
					diff = self.alpha[k]*(1-self.egonet[k][l]['weight'])
					self.egonet[k][l]['weight'] += diff
				else:
					self.egonet.add_edge(k,l,weight=self.alpha[k])
		edge_to_remove = set([])
		for i in R_event:
			for j in self.egonet[i]:
				if not R_event.has_edge(i,j):
					if rd.random()<self.P_removal(self.egonet[i][j]['weight'],self.egonet.out_degree(i),i):
						edge_to_remove.add((i,j))
					else:
						diff = self.alpha[i]*self.egonet[i][j]['weight']
						self.egonet[i][j]['weight'] -= diff
		self.egonet.remove_edges_from(edge_to_remove)
		#update self.act
		for i in range(self.N):
			if i in R_event:
				self.act[i] += self.alpha_act[i]*(1-self.act[i])
			else:
				self.act[i] -= self.alpha_act[i]*self.act[i]

	#evolve the network for time step t
	def Step(self,t):
		event = nx.Graph()
		#first descendents establish relations
		for i,deg in self.descendents.items():
			m = np.random.poisson(lam=deg)
			if m>0:
				event.add_edges_from([(i,j) for j in self.Play(i,m)])
		#second ancestors pop (note that ancestors are necessarily in even number)
		#available nodes are inactive ones
		available = set(range(self.N)).difference(set(event.nodes))
		nb_ancestor = 2*np.random.poisson(lam=self.nb_anc/2)
		#self.prev_anc = nb_ancestor
		#select the ancestors among the available nodes :
		#We choose one node i on the basis on either Way 1 or 2. Then this node plays with m = 1,
		#interacting with node j. We remove i and j from the available nodes and we add (i,j)
		#to the event graph.
		for _ in range(nb_ancestor//2):
			i = self.Draw(available)
			available.remove(i)
			j = self.Anc_play(i,available)
			available.remove(j)
			event.add_edge(i,j)

		#external excitatory stimulation :
		#we assume that the nb_sup edges with highest weight
		#in the social bond graph are activated
		if rd.random()<self.p_stimulus:
			nb_sup = 100
			event.add_edges_from(sorted([edge for edge in self.egonet.edges if not event.has_edge(*edge)],key=lambda edge:self.egonet[edge[0]][edge[1]]['weight'])[:nb_sup])

		if not event:
			return t
		#update the descendents
		self.descendents = {i:event.degree(i) for i in event.nodes}
		nodes_to_remove = set([])
		for i,val in self.descendents.items():
			if val%2:
				if rd.random()<0.5:
					self.descendents[i] = floor(val/2)
					if val==1:
						nodes_to_remove.add(i)
				else:
					self.descendents[i] = ceil(val/2)
			else:
				self.descendents[i] = val//2
		for i in nodes_to_remove:
			del self.descendents[i]
		#update the social bond graph and self.act
		self.Update_egonet(event)
		#update the common neighbors
		self.Compute_comngh(event)
		#record the events
		for edge in event.edges:
			self.events.append([t,*edge])
		return t+1

	#produce a TN of duration duree and save it
	def Evolve(self,duree,name):
		#wait for the first interaction
		t = 0
		while t==0:
			t = self.Step(t)
		#start the time flow
		while t<duree:
			t = self.Step(t)
			if t*100%duree==0:
				print(str(t*100//duree)+" % computed")
		#print the effective number of nodes
		all_nodes = set([])
		for line in self.events:
			all_nodes.add(line[1])
			all_nodes.add(line[2])
		print('effective number of nodes :',len(all_nodes))
		dic_nodes = {node:i for i,node in enumerate(all_nodes)}
		correvent = np.asarray(self.events)
		for n in range(len(correvent)):
			for k in [1,2]:
				correvent[n,k] = dic_nodes[self.events[n][k]]
		#save the events data
		np.savetxt('data/atn/'+name+'.txt',correvent,fmt='%d')

class ADM_class:
	"""Revisited Activity Driven model with Memory
	basis version :
	p_g=1e-2,c=1,p_c=0.3,p_u=1e-3
	"""
	def __init__(self,readable_param,m='random',a='lognorm',update='alpha,i',context='neutral',c_ij=True,egonet_growth='cst',remove='edge'):
		#minimum value for alpha or beta
		self.alpha_min = 1e-3
		#maximum value for alpha or beta
		self.alpha_max = 1
		#number of nodes
		self.N = readable_param['N']
		#temporal network duration
		self.duree = readable_param['T']
		#parameters of the node activity distribution
		self.node_mu = readable_param['mu']
		self.node_sig = readable_param['sigma']
		#activated[i] = 1 if the node i has activated at least once during the simulation
		self.activated = np.zeros(self.N,dtype=int)
		#free parameters are determined by a genetic algo
		self.free_param = {}
		#save the initialization parameters to correctly refresh
		self.refresh_param = [m,a,update]
		#number of emitted interactions per active node
		if m=='random':
			self.free_param['m_max'] = 1
			self.m = []
			self.Get_m = self.Get_m_var
		elif m=='cst':
			self.Get_m = self.Get_m_cst
			self.free_param['m'] = 0
		#node activity
		if a=='lognorm':
			self.act = []
			self.Get_act = self.Get_act_var
			#minimum node activity
			self.a_min = readable_param['a_min']
			#maximum node activity
			self.a_max = readable_param['a_max']
		elif a=='power':
			self.act = []
			self.Get_act = self.Get_act_var
			self.free_param['a_min'] = 0
			self.free_param['a_max'] = 0
		elif a=='empirical':
			self.act = []
			self.Get_act = self.Get_act_var
		elif a=='cst':
			self.free_param['a'] = 0
			self.Get_act = self.Get_act_cst
		#rate of weight change for social bond ties
		if update=='alpha,i':
			self.Get_alpha = self.Get_alpha_i
			self.Get_beta = self.Get_alpha_i
			self.alpha = []
			self.Update_egonet = self.Update_egonet_Alpha
		elif update=='alpha':
			self.Get_alpha = self.Get_alpha_cst
			self.Get_beta = self.Get_alpha_cst
			self.free_param['alpha'] = 0
			self.Update_egonet = self.Update_egonet_Alpha
		elif update=='alpha,beta,i':
			self.Get_alpha = self.Get_alpha_i
			self.Get_beta = self.Get_beta_i
			self.alpha = []
			self.beta = []
			self.Update_egonet = self.Update_egonet_Alpha
		elif update=='alpha,beta,ij':
			self.Get_alpha = self.Get_alpha_ij
			self.Get_beta = self.Get_beta_ij
			self.alpha = []
			self.beta = []
			self.Update_egonet = self.Update_egonet_Alpha
		elif update=='linear':
			self.Update_egonet = self.Update_egonet_Linear
		elif update=='linear,decay':
			self.Update_egonet = self.Update_egonet_Linear_with_decay
		elif update=='no_memory':
			self.Update_egonet = self.Update_egonet_no_memory
		#choose the function computing the probability of growing the egonet
		if egonet_growth=='cst':
			self.Grow_egonet = self.Grow_egonet_cst
			self.free_param['p_g'] = 0
			self.free_param['p_u'] = 0
		elif egonet_growth=='var':
			self.Grow_egonet = self.Grow_egonet_var
			self.free_param['c'] = 0
		#choose the function computing the sets R_event and W_event :
		#how the weight of contextual interactions in the social bond graph should be updated
		if context=='equivalent':
			self.Context = self.Context_equivalent
			self.free_param['p_c'] = 0
			self.Add_contint = self.Add_contint_True
		elif context=='neutral':
			self.Context = self.Context_neutral
			self.free_param['p_c'] = 0
			self.Add_contint = self.Add_contint_True
		elif context=='noise':
			self.Context = self.Context_noise
			self.free_param['p_c'] = 0
			self.Add_contint = self.Add_contint_True
		elif context==None:
			self.Context = self.Context_equivalent
			self.Add_contint = self.Add_contint_False
		#choose the function removing ties from the social bond graph
		if remove=='edge':
			self.Pruning = self.Prune_edge
			self.free_param['lambda'] = 0
		elif remove=='node':
			self.Pruning = self.Prune_node
			self.free_param['p_d'] = 0
		#context memory : if True, the weight is modified by the number of common neighbours in the
		#interaction graph
		if c_ij==True:
			self.Context_coeff = self.Context_coeff_True
			self.Compute_comngh = self.Compute_comngh_True
		elif c_ij==False:
			self.Context_coeff = self.Context_coeff_False
			self.Compute_comngh = self.Compute_comngh_False

		#social bond graph (egonet connections) : directed weighted graph
		self.social_graph = nx.DiGraph()
		#commngh[ind(i,j)] = nb of common neighbours btw i and j at previous time step
		self.comngh = {}
		#events[k] = [t_k,i_k,j_k] = k^th interaction
		self.events = []

	def Power_law_act(self):
		return self.free_param['a_min']*(self.free_param['a_max']/self.free_param['a_min'])**rd.random()

	def Power_law_alpha(self):
		return self.alpha_min*(self.alpha_max/self.alpha_min)**rd.random()

	def Get_m_cst(self,i):
		return self.free_param['m']

	def Get_m_var(self,i):
		return self.m[i]

	def Get_act_cst(self,i):
		return self.free_param['a']

	def Get_act_var(self,i):
		return self.act[i]

	def Get_alpha_i(self,i,j):
		return self.alpha[i]

	def Get_beta_i(self,i,j):
		return self.beta[i]

	def Get_alpha_cst(self,i,j):
		return self.free_param['alpha']

	def Get_alpha_ij(self,i,j):
		return self.alpha[i,j]

	def Get_beta_ij(self,i,j):
		return self.beta[i,j]

	def Grow_egonet_cst(self,i):
		return self.free_param['p_g']

	def Grow_egonet_var(self,i):
		return self.free_param['c']/(self.free_param['c']+self.social_graph.out_degree(i))

	def Coeff_ngh(self,i,k):
		ind = self.Edge_to_ind(i,k)
		if ind in self.comngh:
			return 1+self.comngh[ind]
		return 1

	def Context_coeff_False(self,i,nodes):
		x = rd.random()
		norm = sum([self.social_graph[i][k]['weight'] for k in nodes])
		s = 0; j = -1
		while s<x:
			j += 1
			s += self.social_graph[i][nodes[j]]['weight']/norm
		return nodes[j]

	#same as Context_coeff_False
	#but the weight is modulated by the number of common ngh at previous step
	def Context_coeff_True(self,i,nodes):
		x = rd.random()
		norm = sum([self.social_graph[i][k]['weight']*self.Coeff_ngh(i,k) for k in nodes])
		s = 0; j = -1
		while s<x:
			j += 1
			s += self.social_graph[i][nodes[j]]['weight']*self.Coeff_ngh(i,nodes[j])/norm
		return nodes[j]

	def Add_contint_False(self,event,active_nodes):
		return nx.Graph()

	def Add_contint_True(self,event,active_nodes):
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
			norms[q] = sum([self.social_graph[q][l]['weight']*self.Coeff_ngh(q,l) for l in self.social_graph[q]])
		#close the triangles
		for (i,j,k) in open_triangles:
			#p_ik = proba that i decides to close the triangle
			#p_ki = proba that k decides to close the triangle
			if i in active_nodes:
				if self.social_graph.has_edge(i,j):
					p_ij = self.social_graph[i][j]['weight']*self.Coeff_ngh(i,j)/norms[i]
				else:
					p_ij = self.Grow_egonet(i)
				if self.social_graph.has_edge(j,k):
					p_jk = self.social_graph[j][k]['weight']*self.Coeff_ngh(j,k)/norms[j]
				else:
					p_jk = self.Grow_egonet(j)
				x = p_ij*p_jk*self.free_param['p_c']
				norm = (1 + x*(self.Coeff_ngh(i,k) - 1))
				p_ik = x*self.Coeff_ngh(i,k)/norm
			else:
				p_ik = 0

			if k in active_nodes:
				if self.social_graph.has_edge(k,j):
					p_kj = self.social_graph[k][j]['weight']*self.Coeff_ngh(j,k)/norms[k]
				else:
					p_kj = self.Grow_egonet(k)
				if self.social_graph.has_edge(j,i):
					p_ji = self.social_graph[j][i]['weight']*self.Coeff_ngh(i,j)/norms[j]
				else:
					p_ji = self.Grow_egonet(j)
				x = p_kj*p_ji*self.free_param['p_c']
				norm = (1 + x*(self.Coeff_ngh(i,k) - 1))
				p_ki = x*self.Coeff_ngh(i,k)/norm
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
	def Play(self,i):
		inst_ngh = set() #nodes with whom i succeeds to interact
		unknown_nodes = [k for k in range(self.N) if not self.social_graph.has_edge(i,k) and k!=i]
		known_nodes = [k for k in self.social_graph[i]]
		#i tries to emit self.Get_m(i) intentional interactions
		for _ in range(self.Get_m(i)):
			#if the egonet is empty, choose a partner uniformly at random
			if self.social_graph.out_degree(i)==0 and unknown_nodes:
				j = rd.choice(unknown_nodes)
				inst_ngh.add(j)
			#grow the egonet, if possible
			elif rd.random()<self.Grow_egonet(i) and unknown_nodes:
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
					j = self.Context_coeff_False(i,list(self.social_graph[i]))
					#choose a 2nd order ngh
					nodes = [k for k in self.social_graph[j] if k!=i and k not in inst_ngh and not self.social_graph.has_edge(i,k)]
					if nodes:
						inst_ngh.add(self.Context_coeff(j,nodes))
					else:
						j = rd.choice(unknown_nodes)
						inst_ngh.add(j)
			#interact with a known node, if possible
			elif known_nodes:
				j = self.Context_coeff(i,known_nodes)
				inst_ngh.add(j)
			#update known and unknown nodes
			unknown_nodes = [k for k in unknown_nodes if k not in inst_ngh]
			known_nodes = [k for k in known_nodes if k not in inst_ngh]
		return inst_ngh

	def Prune_edge(self,R_event,W_event):
		edge_to_remove = set()
		for i in R_event:
			m = self.Get_m(i)
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

	def Prune_node(self,R_event,W_event):
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
	def Update_egonet_Alpha(self,R_event,W_event):
		#ties reinforcement
		for i,j in R_event.edges:
			for k,l in zip([i,j],[j,i]):
				alpha = self.Get_alpha(k,l)
				if self.social_graph.has_edge(k,l):
					self.social_graph[k][l]['weight'] += alpha*(1-self.social_graph[k][l]['weight'])
				else:
					self.social_graph.add_edge(k,l,weight=alpha)
		#ties decay
		for i in R_event:
			for j in self.social_graph[i]:
				if not W_event.has_edge(i,j):
					self.social_graph[i][j]['weight'] -= self.Get_beta(i,j)*self.social_graph[i][j]['weight']

	#update self.social_graph in case of a linear reinforcement process
	#and a linear decay process
	def Update_egonet_Linear_with_decay(self,R_event,W_event):
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
	def Update_egonet_no_memory(self,R_event,W_event):
		for i,j in R_event.edges:
			for k,l in zip([i,j],[j,i]):
				if not self.social_graph.has_edge(k,l):
					self.social_graph.add_edge(k,l,weight=1)

	#update self.social_graph in case of a linear reinforcement process
	#and no linear decay process
	def Update_egonet_Linear(self,R_event,W_event):
		#ties reinforcement
		for i,j in R_event.edges:
			for k,l in zip([i,j],[j,i]):
				if self.social_graph.has_edge(k,l):
					self.social_graph[k][l]['weight'] += 1
				else:
					self.social_graph.add_edge(k,l,weight=1)

	def Context_equivalent(self,event,tot_event):
		return tot_event,tot_event

	def Context_neutral(self,event,tot_event):
		return event,tot_event

	def Context_noise(self,event,tot_event):
		return event,event

	def Edge_to_ind(self,i,j):
		if i>j:
			i,j = j,i
		return i*(2*self.N-i-1)//2 + j-i-1

	#compute self.comngh
	def Compute_comngh_True(self,event):
		self.comngh = {}
		active_nodes = list(event.nodes)
		nb_nodes = len(active_nodes)
		for ind1 in range(nb_nodes-1):
			i = active_nodes[ind1]; ngh1 = set(event[i])
			for ind2 in range(ind1+1,nb_nodes):
				j = active_nodes[ind2]
				ngh2 = set(event[j])
				self.comngh[self.Edge_to_ind(i,j)] = len(ngh1.intersection(ngh2))

	#compute self.comngh
	def Compute_comngh_False(self,event):
		pass

	#evolve the network for time step t
	def Step(self,t):
		active_nodes = set(())
		for i in range(self.N):
			if rd.random()<self.Get_act(i):
				active_nodes.add(i)
		#if there is no node active, no interaction can occur
		if len(active_nodes)==0:
			return t
		#graph of intentional interactions
		event = nx.Graph()
		for i in active_nodes:
			inst_ngh = self.Play(i)
			for j in inst_ngh:
				event.add_edge(i,j)
		if not event:
			return t
		#add contextual interactions (dynamic closure)
		doubling_event = self.Add_contint(event,active_nodes)
		tot_event = nx.Graph()
		tot_event.add_edges_from(event.edges)
		tot_event.add_edges_from(doubling_event.edges)
		#update self.egonet
		R_event,W_event = self.Context(event,tot_event)
		#Hebbian step
		self.Update_egonet(R_event,W_event)
		#pruning step
		self.Pruning(R_event,W_event)
		#compute self.comngh if necessary
		self.Compute_comngh(tot_event)
		#update the activated nodes
		for i in tot_event.nodes:
			self.activated[i] = 1
		#update self.events
		for edge in tot_event.edges:
			self.events.append([t,*edge])
		return t+1

	#erase run data to be ready for a new run
	def Refresh(self):
		self.events = []
		self.social_graph = nx.DiGraph()
		self.social_graph.add_nodes_from(range(self.N))
		self.comngh = {}
		self.activated[:] = 0
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
			self.act = [self.Power_law_act() for _ in range(self.N)]
		if update=='alpha,i':
			self.alpha = [self.Power_law_alpha() for _ in range(self.N)]
		elif update=='alpha,beta,i':
			self.alpha = [self.Power_law_alpha() for _ in range(self.N)]
			self.beta = [self.Power_law_alpha() for _ in range(self.N)]
		elif update=='alpha,beta,ij':
			self.alpha = np.asarray([[self.Power_law_alpha() for i in range(self.N)] for j in range(self.N)])
			self.beta = np.asarray([[self.Power_law_alpha() for i in range(self.N)] for j in range(self.N)])

	#produce a TN of duration self.duree
	def Evolve(self,max_count=3,max_locount=5):
		#nb of nodes that have activated during the run
		nb_nodes = 0; reinitialize = False; count = 0
		while nb_nodes<self.N and count<max_count:
			#reinitialize the TN
			if reinitialize:
				self.Refresh()
			#wait for the first interaction
			t = 0; loc_count = 0
			while t==0 and loc_count<max_locount:
				t = self.Step(t)
				if t==0:
					loc_count += 1
			if loc_count==max_locount:
				reinitialize = True; count += 1
			else:
				#start the time flow
				while t<self.duree and loc_count<max_locount:
					newt = self.Step(t)
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
			return np.asarray(self.events,dtype=int)
		else:
			return None
