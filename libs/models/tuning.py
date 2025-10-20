"""Allows a temporal network generator to be tuned, so that
the generated networks are close to a given reference.

The proximity here is defined as the cosine similarity between temporal motifs
at multiple levels of temporal aggregation.
"""

from typing import List, Union
import os
import networkx as nx
import numpy as np
import json

from libs.temporal_network import Table_interaction
import libs.observables.centered_motifs as ctn
import defs
ROOT_DIR = os.path.dirname(__file__)
for _ in range(2):
	ROOT_DIR = os.path.dirname(ROOT_DIR)

__all__ = ["Metric_tensor", "analyze_ref"]

class Metric_tensor:
	pass

class Reference_observables:
	r"""Define the observables that will be targetted by the tuning of a model.

	The corresponding configuration dictionary should be as follows:
	{
		"obs_name": Literal['NCTN', 'ECTN'],
		"agg_levels": List[int],
		"truncate": Union[int, None]
	}

	The default configuration is:
	{
		"obs_name": 'NCTN',
		"agg_levels": [i for i in range(1, 11)],
		"truncate": 20
	}
	"""
	def __init__(self):
		self.config_path = os.path.join(ROOT_DIR, "analysis", "model_tuning", "config_obs.json")
	
	def load(self) -> dict:
		"""Returns the current config for the reference observables.
		
		By default, these are the NCTN distribution computed at the aggregation levels from 1 to 10 included,
		and truncated to the 20 most frequent motifs.
		"""
		return json.load(self.config_path)

	def _check_format(self, config: dict[str, Union[str, List[int]]]):
		if not "agg_levels" in config:
			raise AttributeError("config should include 'agg_levels' as a key")
		try:
			if not config["obs_name"] in ['NCTN', 'ECTN']:
				raise ValueError("the value associated to 'obs_name' should be either 'NCTN' or 'ECTN'")
		except KeyError:
			raise AttributeError("config should include 'obs_name' as a key")
		try:
			if not isinstance(config['agg_levels'][0], int):
				raise ValueError("the value associated to 'agg_levels' should be a list of ints")
		except TypeError or IndexError:
			raise ValueError("the value associated to 'agg_levels' should be a list of ints")
		try:
			if not isinstance(config["truncate"], int):
				raise ValueError("the value associated to 'truncate' should be either an int or None")
		except KeyError:
			raise AttributeError("config should include 'truncate' as a key")

	def set(self, config: dict[str, Union[str, List[int]]]):
		# check the format
		self._check_format()

		# overwrite the configuration file
		json.dump(config, self.config_path)

def analyze_ref(ref_name:str):
	"""
	Compute the readable parameters for the specified reference,
	and save these data in data/analysis/refs/<ref_name>/readable_params.txt
	Also compute the reference observables (used for fitness computation) and save them in data/analysis/refs/<ref_name>/ref_obs.txt
	"""
	ref_path = os.path.join(ROOT_DIR, "data", "temporal_networks", "formatted", ref_name + '.txt')
	ref_dir_out = os.path.join(ROOT_DIR, "data", "analysis", "refs", ref_name)
	os.makedirs(os.path.join(ref_dir_out), exist_ok=True)

	# compute the readable parameters of each reference
	ref = Table_interaction(ref_path)

	# collect descriptive information:
	# nb of nodes, number of timestamps, number of temporal edges,
	# minimum node activity, maximum node activity, mean and std of
	# the distribution of log(node activity)

	global_info = [
		['nb_nodes', 'duration', 'nb_temp_edges', 'a_min', 'a_max', 'a_mu', 'a_sigma'],
		[ref.nb_nodes, ref.duration, len(ref.data), None, None, None, None]
	]

	# estimate the parameters a_min and a_max
	activity = np.zeros(ref.nb_nodes, dtype=float)
	for t in range(ref.duration):
		for node in ref.nodes(t):
			activity[node] += 1

	global_info[1][3] = np.min(activity) / ref.duration
	global_info[1][4] = np.max(activity) / ref.duration

	# estimate the parameters a_mu and a_sigma
	tab = np.log10(activity) - np.log10(ref.duration)
	values, bins = np.histogram(tab, density=True)
	bin_min = len(bins) - 3
	while values[bin_min] > values[-1]:
		bin_min -= 1
	mu_tab = [el for el in tab if el >= bins[bin_min]]
	global_info[1][5] = np.mean(mu_tab)
	global_info[1][6] = 2*np.sqrt(np.var(mu_tab))
	
	# collect the reference observables (motifs at multiple levels of aggregation):
	# each row of `ref_obs` corresponds to a different value of aggregation
	ref_config = Reference_observables().load()
	ref_obs = []
	for agg in ref_config['agg_levels']:
		ref.sliding_agg(agg)
		distr = ctn.sample_motifs(
			ref,
			ref_config['obs_name'],
			truncate=ref_config['truncate']
		)


	# store the results:
	
	# global_info
	np.savetxt(os.path.join(ref_dir_out, 'global_info.txt'), np.array(global_info, dtype=str), fmt='%s', delimiter=',')
	
	# reference observables
	np.savetxt(os.path.join(ref_dir_out, 'ref_obs.txt'), np.array(global_info, dtype=int), fmt='%d', delimiter=',')

def tune_models():
	"""
	define the models considered and their hyperparameters, and tune each model instance with respect to the references
	then save the best parameters, the evolution of the fitness over the generations, the hyperparameters of the genetic algorithm
	and the best genome
	data are saved at data/analysis/tuned/
	"""
	pass

def get_metric_tensor():
	"""
	compute the metric tensor between each tuned model and its references
	then save the data at data/analysis/metric_tensor/
	"""
	pass

#compute the distributions of the reference observables
#as well as the number of nodes and the duration of the temporal network
#of name name
def get_ref_data(name,dataset=None,supath=''):
	#load the raw data and convert it into a temporal network
	if dataset is None:
		dataset = np.loadtxt(name+'.txt',dtype=int)
	temp_net = Temp_net.Temp_net(dataset)
	#prepare the temporal network for analysis
	temp_net.Prepare()
	#collect descriptive information :
	#nb of nodes, number of timestamps, number of temporal edges
	#minimum node activity, maximum node activity
	X,Y = zip(*temp_net.info.items())
	global_info = [list(X),[str(y) for y in Y]]
	#compute the reference observables
	temp_net.Ref_obs()
	#store results
	#types of observables are point, distribution and vector
	np.savetxt('analysis/'+name+supath+'/global_info.txt',np.asarray(global_info,dtype=str),fmt='%s',delimiter=',')
	for type_obs,dic in temp_net.dic_obs.items():
		for name_obs,val in dic.items():
			Save_obs[type_obs]('analysis/'+name+supath,name_obs,val)

#to determine the free parameters that best match empirical distributions we use a genetic algo
#however it is time-consuming so we use several tricks to speed up the research :
# - to compute the fitness of a sequence, we use just a subset of the observables of reference,
#   using the fact that only a fraction of the observables are independent of each other
# - before computing the distributions of the observables resulting from a sequence,
#   we attribute a fitness of zero (minimal value) if the number of temporal edges of the model
#   is significantly different from the empirical case (i.e. either 10 times too high or 10 times too low)
# - to avoid computing the fitness of the same sequence multiple times, we store the fitness of each
#   encountered sequence in a dictionary, speeding up the algorithm when the genetic diversity drops

class Pop_model:
	"""
	genetic coding (depends on the free parameters),
	two types of parameters:
	 - m and c, which are of the order of 1
	 - parameters corresponding to probabilities
	for coding proba, we use a binary exponential coding :
	the coding has ten bits, to code proba ranging from 1e-3 to 1
	integer parameters range from 1 to 10 so we code them on 4 bits
	"""
	def __init__(self,version_nb,name_XP,pop_size,p_mut,p_cross,breed_size):
		#empirical realizations of the observables used for fitness computation
		self.eval_obs = {'vector':{'ETN3':{}}}
		self.XP_path = 'analysis/'+name_XP+'/'
		#number of reproducers
		self.breed_size = breed_size
		#version of the ADM class we compare with name_XP
		self.ADM_version = version_nb
		#model = atn.ADM_class(N,T,**versions[version_nb])
		self.model = 0
		self.pop_size = pop_size
		#population[num] = genetic sequence of the member of identifier num
		self.population = {}
		#fitness_dic[seq] = fitness of the model with seq as genetic sequence
		self.fitness_dic = {}
		#fitness_pop[num] = fitness of the individual of identtifier num
		self.fitness_pop = [0]*pop_size
		#mutation probability
		self.p_mut = p_mut
		#crossover probability
		self.p_cross = p_cross
		#Decode[param] = function used to convert a genetic sequence into a numerical value for param
		self.Decode = {}
		#number of genes, i.e. number of free parameters coded by a genetic sequence
		self.nb_genes = 0
		#gene_name[gene_num] = (param,length), where length is the number of nucleotides
		#coding for the parameter param. The genetic info is located at the gene number gen_num.
		self.gene_name = []
		#total length of the genetic sequence
		self.seq_length = 0
		#best individual of the former generation
		self.best_num = 0

	#generate a random genetic sequence
	def Generate_seq(self):
		return ''.join([str(_) for _ in np.random.randint(0,2,self.seq_length)])

	#determine the genetic parameters : number of genes and their length
	#initialize the population with random genetic sequences
	#we also prepare to the fitness computation by loading the realizations
	#of the evaluation observables in the reference dataset
	#also tune the fixed parameters of the model
	def Init_pop(self):
		#load the parameters of the dataset of reference (i.e. the reference parameters)
		global_info = np.loadtxt(self.XP_path+'global_info.txt',dtype=str,delimiter=',')
		self.XP_info = {}
		for i in range(len(global_info[0,:])):
			x = global_info[0,i]; y = global_info[1,i]
			if x in {'N','T','nb of edges'}:
				self.XP_info[x] = int(y)
			else:
				self.XP_info[x] = float(y)
		self.model = atn.ADM_class(self.XP_info,**versions[self.ADM_version])
		for type_obs,dic in self.eval_obs.items():
			for name_obs in dic.keys():
				self.eval_obs[type_obs][name_obs] = Load_obs[type_obs](self.XP_path,name_obs,agg_max=10)
		#determine the genetic parameters
		self.seq_length = 0
		for param in self.model.free_param.keys():
			if param in {'m','c','m_max'}:
				self.Decode[param] = self.Decode_integer
				self.gene_name.append((param,2))
				self.seq_length += 2
			elif param in {'a','a_min'}:
				self.Decode[param] = lambda seq : self.Decode_proba(seq,p_min=self.XP_info['a_min'],p_max=self.XP_info['a_max'])
				self.gene_name.append((param,10))
				self.seq_length += 10
			elif param=='lambda':
				self.Decode[param] = lambda seq : self.Decode_proba(seq,p_min=0.01,p_max=10)
				self.gene_name.append((param,10))
				self.seq_length += 10
			elif param=='a_max':
				self.Decode[param] = lambda seq : self.Decode_proba(seq,p_min=self.XP_info['a_min'],p_max=1)
				self.gene_name.append((param,10))
				self.seq_length += 10
			else:
				self.Decode[param] = self.Decode_proba
				self.gene_name.append((param,10))
				self.seq_length += 10
			self.nb_genes += 1
		#intialize the genetic pool of the population
		for num in range(self.pop_size):
			self.population[num] = self.Generate_seq()

	#obtain the free parameters from the genetic sequence
	def Sequence_to_model(self,seq):
		#current location on the genetic sequence
		loc = 0
		#number of the gene currently translated
		gene_num = 0
		while gene_num<self.nb_genes:
			param,length = self.gene_name[gene_num]
			self.model.free_param[param] = self.Decode[param](seq[loc:loc+length])
			loc += length
			gene_num += 1

	#convert seq into an integer in [1,2**4]
	def Decode_integer(self,seq):
		res = 1
		for i,s in enumerate(seq):
			res += int(s)*2**i
		return res

	#convert seq into a float in [p_min,p_max]
	def Decode_proba(self,seq,p_min=1e-3,p_max=1):
		res = 0
		for i,s in enumerate(seq):
			res += int(s)*2**i
		return p_min + (p_max-p_min)*res/(2**10-1)

	#compute the fitness of the genetic sequence seq
	#the fitness is the product of the similarity between the model instance coded by seq
	#and the reference dataset self.XP_path for each observable of evaluation
	#each similarity is a number comprised between 0 and 1
	def Fitness(self,seq):
		#tune the model parameters to those coded by seq
		self.Sequence_to_model(seq)
		#refresh the model data
		self.model.refresh()
		#compute the interaction graph
		model_data = self.model.Evolve()
		if model_data is None:
			return 0
		temp_net = Temp_net.Temp_net(model_data)
		#prepare the temporal network for partial analysis
		temp_net.Partial_prepare(self.model.N,self.model.duree)	
		if temp_net.info['nb of edges']>self.XP_info['nb of edges']*10:
			return 0
		#compute the evaluation observables for the model instance
		temp_net.Evaluation_obs()
		res = 1
		#compute the similarity between the model and the reference dataset
		#for each observable
		for type_obs,dic in self.eval_obs.items():
			for name_obs,realization in dic.items():
				res *= 1-Distance_obs[type_obs](temp_net.eval_obs[type_obs][name_obs],realization)
		return res
	
	#mutate the sequence seq
	def Mutate(self,seq):
		#determine the number of mutations
		nb_mut = np.random.poisson(lam=self.seq_length*self.p_mut)
		if nb_mut>0:
			list_seq = list(seq)
			for i in rd.sample(range(self.seq_length),nb_mut):
				list_seq[i] = str(1-int(list_seq[i]))
			seq = ''.join(list_seq)
		return seq

	#combine the individuals num1 and num2 to produce two new individuals
	def Combine(self,seq1,seq2):
		if rd.random()<self.p_cross:
			#choose whether this is a simple or double crossover
			if rd.random()<0.7:
				#double crossover
				locus1 = rd.randint(0,self.seq_length-3)
				locus2 = rd.randint(locus1+1,self.seq_length-1)
				newseq1 = seq1[:locus1] + seq2[locus1:locus2] + seq1[locus2:]
				newseq2 = seq2[:locus1] + seq1[locus1:locus2] + seq2[locus2:]
			else:
				#simple crossover
				locus = rd.randint(1,self.seq_length-2)
				newseq1 = seq1[:locus] + seq2[locus:]
				newseq2 = seq2[:locus] + seq1[locus:]
		else:
			newseq1 = seq1
			newseq2 = seq2
		#mutate the two sequences
		return self.Mutate(newseq1),self.Mutate(newseq2)

	#breed a new generation by sexual reproduction
	#between the reproducers individuals
	#we impose also that the new generation contains the best individual of the former
	def Breed(self,reproducers):
		deg = (self.pop_size-1)//self.breed_size
		#initialize the breeding graph with a regular graph
		breeding_graph = nx.generators.random_graphs.random_regular_graph(deg,self.breed_size)
		newpop = {0:self.population[self.best_num]}; num = 1
		for num1,num2 in breeding_graph.edges:
			newpop[num],newpop[num+1] = self.Combine(reproducers[num1],reproducers[num2])
			num += 2
		#complete the population with random combinations
		nb_edges = self.pop_size-1-deg*self.breed_size
		if nb_edges%2:
			nb_edges = nb_edges//2 + 1
		for _ in range(nb_edges):
			seq1,seq2 = rd.sample(reproducers,2)
			newpop[num],newpop[num+1] = self.Combine(seq1,seq2)
			num += 2
		if len(newpop)>self.pop_size:
			del newpop[num-1]
		self.population = newpop

	#determine the next generation
	def Next_generation(self):
		#first evaluate all the current individuals
		for num,seq in self.population.items():
			if seq in self.fitness_dic:
				self.fitness_pop[num] = self.fitness_dic[seq]
			else:
				self.fitness_dic[seq] = self.Fitness(seq)
				self.fitness_pop[num] = self.fitness_dic[seq]
		#sort the individuals by increasing fitness
		list_num = sorted(range(self.pop_size),key=lambda num:self.fitness_pop[num],reverse=True)
		#second gather the reproducers :
		#nb_best best individuals + nb_rand new random individuals
		nb_best = self.breed_size - max(self.breed_size//10,1)
		nb_rand = max(self.breed_size//10,1)
		reproducers = [self.population[num] for num in list_num[:nb_best]]
		for _ in range(nb_rand):
			reproducers.append(self.Generate_seq())
		#third breed the next generation, imposing to keep
		#the best individual of the former generation
		self.best_num = list_num[0]
		self.Breed(reproducers)

#determine and then analyze the closest instance of the model version_nb from the dataset name
def Closest_instance(name,version_nb):
	print(name)
	print(version_nb)
	pop_size = 40
	p_mut = 0.02
	p_cross = 0.7
	breed_size = 8
	pop_model = Pop_model(version_nb,name,pop_size,p_mut,p_cross,breed_size)

	#run multiple generations
	#keep track of the highest fitness and the average fitness
	nb_epochs = 20
	pop_model.Init_pop()
	best_fit = []
	avg_fit = []
	for _ in range(nb_epochs):
		pop_model.Next_generation()
		best_fit.append(np.max(pop_model.fitness_pop))
		avg_fit.append(np.mean(pop_model.fitness_pop))
	#store the results
	#save the evolution of the fitness accross generations
	np.savetxt('analysis/ADM_class_V'+str(version_nb)+'/'+name+'/best_fit.txt',best_fit)
	np.savetxt('analysis/ADM_class_V'+str(version_nb)+'/'+name+'/avg_fit.txt',avg_fit)
	#save the genetic sequence of the best individual
	best_seq = pop_model.population[0]
	np.savetxt('analysis/ADM_class_V'+str(version_nb)+'/'+name+'/best_seq.txt',[int(_)for _ in list(best_seq)],fmt='%d')
	#save the corresponding free parameters
	pop_model.Sequence_to_model(best_seq)
	best_param = np.array(list(zip(*pop_model.model.free_param.items())),dtype=str)
	np.savetxt('analysis/ADM_class_V'+str(version_nb)+'/'+name+'/best_param.txt',best_param,delimiter=',',fmt='%s')
	#generate an instance of the best model
	pop_model.model.Refresh()
	instance = pop_model.model.Evolve()
	#analyze it
	Analyze('ADM_class_V'+str(version_nb),dataset=instance,supath='/'+name)

#determine the closest instance of the version version_nb of the ADM model from the empirical dataset name
#then analyze this instance

