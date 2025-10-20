import numpy as np
import networkx as nx
import random as rd
from Librairies.utils import PROJECT_ROOT

#load TN data of a given name, this TN is not a randomization of another
def Load_TN_no_random(name):
	if type(name)==tuple:
		if type(name[0])==str:
			#name should be of the form ('ADM',version_nb,reference)
			if name[0]=='ADM':
				return Load_TN_ADM(name[1],name[2])
			#name should be of the form ('min_EW',version_nb)
			elif name[0]=='min_EW':
				return Load_TN_Min_EW(name[1])
			elif name[0]=='min_ADM':
				return Load_TN_min_ADM(name[1])
		#name should be of the form (dataset,name of the dataset)
		elif len(name)==2 and type(name[0])==np.ndarray:
			return name[0]
	return np.loadtxt(PROJECT_ROOT+'/data/'+name+'.txt',dtype=int)

#load TN data of a given name
def Load_TN(name):
	if type(name)==tuple:
		if type(name[-1])==str:
			if 'randomized' in name[-1]:
				if len(name)==2:
					data = Load_TN_no_random(name[0])
				else:
					data = Load_TN_no_random(name[:-1])
				#randomization type
				num_random = int(name[-1][10:])
				#strong time shuffling: the timeline of each edge is shuffled
				if num_random==1:
					#collect the number of times each edge is active, i.e. the time aggregated network
					time_agg = nx.Graph()
					for n in range(np.size(data,0)):
						i,j = data[n,1:]
						if time_agg.has_edge(i,j):
							time_agg[i][j]['weight'] += 1
						else:
							time_agg.add_edge(i,j,weight=1)
					#draw new activity times for each edge
					list_times = list(range(data[-1,0]+1)) #list of available times
					new_TN = {} #new_TN[t] = set of edges active at time t
					for edge in time_agg.edges:
						new_times = rd.sample(list_times,time_agg[edge[0]][edge[1]]['weight'])
						for t in new_times:
							if t in new_TN:
								new_TN[t].add(edge)
							else:
								new_TN[t] = {edge}
					#put new_TN under the form of a formatted measurement table
					new_data = []
					for num,t in enumerate(sorted(new_TN.keys())):
						for edge in new_TN[t]:
							new_data.append([num,*edge])
					return np.asarray(new_data,dtype=int)
				#weaker time shuffling: the global edge activity timeline is preserved
				#however, co-occurent edges before this shuffling may not be after
				elif num_random==2:
					#get data_time
					data_time = []
					n1 = 0; n_max = np.size(data,0)
					for n in range(1,n_max):
						t = data[n-1,0]
						if data[n,0]>t:
							data_time += [[n1,n]]
							n1 = n
					#take care of the last line of data
					t = data[-1,0]
					data_time += [[n1,n_max]]
					#shuffle the measurement table
					rng = np.random.default_rng()
					new_data = rng.shuffle(data)
					for t,el in enumerate(data_time):
						new_data[el[0]:el[1],0] = t
					return new_data
				elif num_random==3:
					pass
					#keep contact and intercontact the same ; i.e. shuffle edge events
				elif num_random==4:
					pass
					#reattribute the edge activity profiles among edges
			return Load_TN_no_random(name)
		return Load_TN_no_random(name)
	return Load_TN_no_random(name)

#an observable is written as 'object'0'object property we want to sample', e.g.
#'NCTN0nb_diff' or 'edge0event_duration' or 'ICC' or 'node_space_weight'
def Obs_to_chunks(obs):
	if '0' in obs:
		obj1,prop = obs.split('0')
		if 'duration' in prop or 'events' in prop:
			obj2 = 'train'
		elif 'time_weight' in prop or 'motif' in prop or 'entropy' in prop or 'nb' in prop:
			obj2 = 'time_weight'
		return ([obj1,obj2],prop)
	return ([],obs)

#return how we should call the dataset corresponding to name in a figure or a data file
def Get_savename(name):
	if type(name)==tuple:
		if type(name[-1])==str:
			if 'randomized' in name[-1]:
				if len(name)==2:
					return Get_savename_no_random(name[0])+'_'+name[-1]
				else:
					return Get_savename_no_random(name[:-1])+'_'+name[-1]
			return Get_savename_no_random(name)
		return Get_savename_no_random(name)
	return Get_savename_no_random(name)

#return how we should call the dataset corresponding to name in a figure or a data file
#given the dataset is not a randomization of another
def Get_savename_no_random(name):
	if type(name)==tuple:
		if type(name[0])==str:
			if name[0]=='ADM':
				savename = name[0]+str(name[1])+name[2]
			elif name[0]=='min_EW':
				savename = name[0]+str(name[1])
			elif name[0]=='min_ADM' or name[0]=='min_V7':
				savename = 'min_ADM'+str(name[1])
		elif type(name[0])==np.ndarray:
			savename = name[1]
	else:
		savename = name
	return savename

#for not randomized datasets
def Savename_to_name_no_random(savename):
	if savename[:7]=='min_ADM':
		name = ('min_ADM',int(savename[7:]))
	elif savename[:6]=='min_EW':
		name = ('min_EW',int(savename[6:]))
	elif savename[:3]=='ADM':
		ind = 0; set_int = {str(i) for i in range(10)}
		while savename[ind+3] in set_int:
			ind += 1
		name = ('ADM',int(savename[3:3+ind]),savename[3+ind:])
	else:
		name = savename
	return name

#include randomized datasets
def Savename_to_name(savename):
	if '_randomized' in savename:
		ind = 0
		while savename[ind:ind+11]!='_randomized':
			ind += 1
		name_no_random = Savename_to_name_no_random(savename[:ind])
		if type(name_no_random)==tuple:
			return name_no_random + (savename[ind+1:],)
		return (name_no_random,savename[ind+1:])
	return Savename_to_name_no_random(savename)

def Load_TN_Min_EW(version_nb):
	return atn.Min_EW(138,3635,0.79,**Get_versions(choice="min_EW")[version_nb]).Evolve()

def Load_TN_min_ADM(version_nb):
	version = Get_versions(choice='ADM')[7]
	#load XP info of conf16
	XP_info = {'N':138,'T':3635,'nb of edges':153371,'sigma':0.34,'mu':-0.56}
	#modifies the version
	version['m'] = 'cst'
	version['a'] = 'cst'
	version['c_ij'] = False
	version['update'] = 'linear'
	version['context'] = None
	version['remove'] = 'node'
	#decide of the parameters
	dic_param = {}
	dic_param['a'] = 0.3
	dic_param['m'] = 1
	if version_nb==2:
		p_d = 0.1
	elif version_nb==1:
		p_d = 0.02
	dic_param['p_d'] = p_d
	dic_param['p_u'] = 1
	dic_param['p_g'] = 0.08498
	#generate the model instance
	model = atn.ADM_class(XP_info,**version)
	for param in model.free_param.keys():
		model.free_param[param] = dic_param[param]
	model.Refresh()
	return model.Evolve()

def Get_versions(choice='ADM'):
	versions = {}
	if choice=='ADM':
		#version 1 = basis version
		#versions 2 to 13 are adjacent to 1
		#versions 14 and beyond are combinations of multiple adjacent versions

		#basis version :
		#m_{i}, a_{i}, alpha_{i}, contextual interactions are neutral, c_{i,j} used, constant egonet growth,
		#removal of edges depending on their weight, Alpha update process
		versions[1] = {'m':'random','a':'power','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#2nd version : linear reinforcement process and no gradual decay
		versions[2] = {'m':'random','a':'power','update':'linear','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#3rd version : random removal of nodes
		versions[3] = {'m':'random','a':'power','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'node'}

		#4th version : varying egonet growth
		versions[4] = {'m':'random','a':'power','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'var','remove':'edge'}

		#5th version : c_{i,j} = 1
		versions[5] = {'m':'random','a':'power','update':'alpha,i','context':'neutral','c_ij':False,'egonet_growth':'cst','remove':'edge'}

		#6th version : intentional and contextual interactions are equivalent
		versions[6] = {'m':'random','a':'power','update':'alpha,i','context':'equivalent','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#7th version : contextual interactions are pure noise
		versions[7] = {'m':'random','a':'power','update':'alpha,i','context':'noise','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#8th version : no contextual interactions
		versions[8] = {'m':'random','a':'power','update':'alpha,i','context':None,'c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#9th version : alpha
		versions[9] = {'m':'random','a':'power','update':'alpha','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#10th version : alpha_{i}, beta_{i}
		versions[10] = {'m':'random','a':'power','update':'alpha,beta,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#11th version : alpha_{i,j}, beta_{i,j}
		versions[11] = {'m':'random','a':'power','update':'alpha,beta,ij','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#12th version : a
		versions[12] = {'m':'random','a':'cst','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#13th version : m
		versions[13] = {'m':'cst','a':'power','update':'alpha,i','context':'neutral','c_ij':True,'egonet_growth':'cst','remove':'edge'}

		#14th version : ori_ADM, i.e. the following combination : versions 2+3+4+5+8+13
		versions[14] = {'m':'cst','a':'power','update':'linear','context':None,'c_ij':False,'egonet_growth':'var','remove':'node'}

		#15th version : 2+5+8+13
		versions[15] = {'m':'cst','a':'power','update':'linear','context':None,'c_ij':False,'egonet_growth':'cst','remove':'edge'}

		#16th version : 5+8+11+13
		versions[16] = {'m':'cst','a':'power','update':'alpha,beta,ij','context':None,'c_ij':False,'egonet_growth':'cst','remove':'edge'}

		#17th version : 3+5+8+9+12+13 (simplest version with exponential Hebbian process)
		versions[17] = {'m':'cst','a':'cst','update':'alpha','context':None,'c_ij':False,'egonet_growth':'cst','remove':'node'}

		#18th version : 2+3+5+8+12+13 (simplest version with linear Hebbian process)
		versions[18] = {'m':'cst','a':'cst','update':'linear','context':None,'c_ij':False,'egonet_growth':'cst','remove':'node'}

		#19th version : 7+9+13 (best expected version)
		versions[19] = {'m':'cst','a':'power','update':'alpha','context':'noise','c_ij':True,'egonet_growth':'cst','remove':'edge'}
		#15th version : linear reinforcement process and linear decay process
		#16th version : transitive initialization for weights of social ties
	elif choice=='min_EW':
		#if shift=True and removal'=None then
		#the number of temporal edges is O(duree**2) instead of O(duree)
		#so the analysis is barely doable. One way to do it would be to parallelize the computation of
		#observables by using the CPT clusters
		#On the contrary, if shift=False and 'removal'!=None then
		#the data set will only contain newborn activations in the stationary state
		versions[1] = {'shift':False,'removal':None,'newborn':'random'}
		versions[2] = {'shift':True,'removal':'node_unif','newborn':'random'}
		versions[3] = {'shift':True,'removal':'edge_unif','newborn':'random'}
	return versions

def Load_instance_param(version_nb,name):
	model = 'ADM_class_V'+str(version_nb)
	set_int_param = {'m_max','m','c'}
	dic_param = {}
	#load parameters
	best_param = np.loadtxt(os.path.join(ADM_DIR,model+'/'+name+'/best_param.txt'),dtype=str,delimiter=',')
	for i,param in enumerate(best_param[0,:]):
		if param in set_int_param:
			dic_param[param] = int(best_param[1,i])
		else:
			dic_param[param] = float(best_param[1,i])
	return dic_param

def Load_XP_info(name):
	global_info = np.loadtxt(os.path.join(ADM_DIR,name+'/global_info.txt'),dtype=str,delimiter=',')
	XP_info = {}
	for i in range(len(global_info[0,:])):
		x = global_info[0,i]; y = global_info[1,i]
		if x in {'N','T','nb of edges'}:
			XP_info[x] = int(y)
		else:
			XP_info[x] = float(y)
	return XP_info

#load TN data from the ADM class
def Load_TN_ADM(version_nb,name):
	versions = Get_versions(choice='ADM')
	#load instance parameters
	dic_param = Load_instance_param(version_nb,name)
	#load XP info
	XP_info = Load_XP_info(name)
	#generate the model instance
	Model = atn.ADM_class(XP_info,**versions[version_nb])
	for param in Model.free_param.keys():
		Model.free_param[param] = dic_param[param]
	Model.Refresh()
	return Model.Evolve()

#return the list of non randomized TN (savename format)
def Get_tot_TN_not_randomized():
	tot_TN = []
	#list_XP
	for i in range(16,20):
		tot_TN.append('conf'+str(i))
	for i in range(1,4):
		tot_TN.append('highschool'+str(i))
	for i in range(1,3):
		tot_TN.append('work'+str(i))
	tot_TN += ['utah','french','baboon','hospital','malawi']
	#list_raha
	tot_TN += ['ABP2pi','ABPpi4','brownD001','brownD01','brownD1','Vicsek2pi','Vicsekpi4']
	#list_model
	tot_TN += [('ADM',9,'conf16'),('ADM',18,'conf16'),('min_ADM',1),('min_ADM',2),('min_EW',1),('min_EW',2),('min_EW',3)]
	return [Get_savename(name) for name in tot_TN]

#return the list of empirical data sets
def Get_tot_XPTN():
	tot_TN = []
	for i in range(16,20):
		tot_TN.append('conf'+str(i))
	for i in range(1,4):
		tot_TN.append('highschool'+str(i))
	for i in range(1,3):
		tot_TN.append('work'+str(i))
	tot_TN += ['baboon','hospital','malawi','utah','french']
	return [Get_savename(name) for name in tot_TN]


class Motifs_tp:
	"""
    input: string, path or table of ints
	"""
	def __init__(self,savename,where=None):
		if where is not None:
			self.data = np.loadtxt(where,dtype=int)
		elif type(savename)==str:
			self.data = Load_TN(Savename_to_name(savename))
		else:
			self.data = savename.copy()
		#nb of nodes
		self.N = len(set(self.data[:,1]).union(set(self.data[:,2])))
		#interaction temporal network
		self.TN = []
		#data_time[i] = [t_i, n_1, n_p] with t_i the (i+1)th time appearing in data
		#n_1 the first line of occurrence of t_i and n_p-1 the last one
		self.data_time = []
		self.get_tuple_keys = None

	#compute self.data_time
	def Get_data_time(self):
		self.data_time = []
		n1 = 0; n_max = np.size(self.data,0)
		for n in range(1,n_max):
			if self.data[n,0]>self.data[n-1,0]:
				self.data_time += [[self.data[n-1,0],n1,n]]
				n1 = n
		#take care of the last line of data
		self.data_time += [[tij_data[n,0],n1,n_max]]

	#return a list of formatted tables tij, each corresponding to one day
	def Split_days(self):
		self.Get_data_time()

		#list of indices in data_time separating the days
		day_indices = [0]
		#time resolution: smallest step btw two consecutive measures
		resolution = self.data_time[1][0] - self.data_time[0][0]
		for ind,el in enumerate(self.data_time[1:],start=1):
			step = el[0] - self.data_time[ind-1][0]
			if step>50*resolution:
				day_indices += [ind-1,ind]
		day_indices.append(len(self.data_time)-1)
		#day_couples[k] = (start,end) the start and end of the day as indices of data_time
		day_couples = [(day_indices[2*k],day_indices[2*k+1]) for k in range(len(day_indices)//2)]

		#remove the days that contain too few data
		#day_info[day] = (nb of measured times in the day , nb of interactions in the day)
		day_info = {day:(day[1]-day[0],self.data_time[day[1]][2]-self.data_time[day[0]][1]) for day in day_couples}
		avg_info = ()
		for tab in zip(*day_info.values()):
			avg_info += (np.mean(tab)/10,)
		confirmed_days = [day for day,info in day_info.items() if (info>avg_info).all()]

		#last check: the maximum nb of interactions measured per time step should exceed 3 on each day
		day_info = {}
		for day in confirmed_days:
			max_nb = max([self.data_time[ind][2]-self.data_time[ind][1] for ind in range(day[0],day[1]+1)])
			day_info[day] = max_nb
		confirmed_days = [day for day,info in day_info.items() if info>3]

		list_tp = []
		for day in confirmed_days:
			tp = Motifs_tp(self.data[self.data_time[day[0]][1]:self.data_time[day[1]][2],:])
			tp.Format()
			list_tp.append(tp)
		return list_tp

	def Plot_raw_timeline(self,title):
		#plot the activity to assess the splitting
		fig,ax = Setup_Plot(r'$t$','number of interactions',fontsize=14,title=title)
		n1,n2 = zip(*self.data_time)
		ax.plot(np.array(n2)-np.array(n1),'.')

	#replace self.data by its formatted version, i.e. time begins at 0, two consecutive times
	#are separated by one and nodes are numeroted from 0 to nb of nodes-1
	def Format(self,max_T=np.inf):
		self.Get_data_time()
		last_line = min(max_T,len(self.data_time))
		self.data = self.data[:self.data_time[last_line-1][1],:]
		#remove any self-loop
		valid_lines = []
		for n in range(np.size(self.data,0)):
			if self.data[n,1]!=self.data[n,2]:
				valid_lines.append(n)
		self.data = self.data[valid_lines,:]
		#recompute data_time
		self.Get_data_time()
		#relabel the nodes
		node_to_int = {}; num = 0
		for n in range(np.size(self.data,0)):
			for k in self.data[n,1:]:
				if not k in node_to_int:
					node_to_int[k] = num
					num += 1
		for t,el in enumerate(self.data_time[:last_line]):
			self.data[el[0]:el[1],0] = t
		for n in range(np.size(self.data,0)):
			for k in range(1,3):
				self.data[n,k] = node_to_int[self.data[n,k]]

	#compute the interaction graph at aggregation level agg (sliding aggregation)
	def Get_TN(self,agg,data_time=True):
		#compute self.data_time
		if data_time:
			self.Get_data_time()
		nb_time = len(self.data_time)
		new_nb = nb_time-agg+1
		#self.TN[t] = aggregated graph of interactions on t^th time interval
		self.TN = [nx.Graph() for _ in range(new_nb)]
		#initial graph
		G = nx.Graph()
		for k in range(agg):
			for n in range(*self.data_time[k]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		for t in range(new_nb-1):
			self.TN[t].add_edges_from(G.edges)
			#remove edges from time t
			for n in range(*self.data_time[t]):
				i,j = self.data[n,1:]
				if G[i][j]['weight']==1:
					G.remove_edge(i,j)
				else:
					G[i][j]['weight'] -= 1
			#add edges from time t+agg
			for n in range(*self.data_time[t+agg]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		self.TN[new_nb-1].add_edges_from(G.edges)

	#return seq,list_interactions where list_interactions[k] = (i,j,tau) with 0<=tau<depth
	def Compute_ECTN_env(self,t,ind,depth):
		list_interactions = []
		i,j = ind; central_profile = ''
		#encode the activity profiles of the satellites: dic_s[u] = activity profile of satellite u
		dir_s = {}; rev_s = {}
		for tau in range(depth):
			if self.TN[t+tau].has_edge(i,j):
				central_profile += '1'
			else:
				central_profile += '0'
			ngh_i = set(); ngh_j = set()
			if i in self.TN[t+tau]:
				ngh_i = set(self.TN[t+tau].neighbors(i))
				for u in ngh_i:
					list_interactions.append((i,u,tau))
			if j in self.TN[t+tau]:
				ngh_j = set(self.TN[t+tau].neighbors(j))
				for u in ngh_j:
					list_interactions.append((j,u,tau))
			both_ij = ngh_i.intersection(ngh_j)
			only_i = ngh_i.difference(both_ij)
			only_j = ngh_j.difference(both_ij)
			for u in only_i:
				if u in dir_s:
					dir_s[u][tau] = '1'
					rev_s[u][tau] = '2'
				else:
					dir_s[u] = ['0']*depth
					rev_s[u] = ['0']*depth
					dir_s[u][tau] = '1'
					rev_s[u][tau] = '2'
			for u in only_j:
				if u in dir_s:
					dir_s[u][tau] = '2'
					rev_s[u][tau] = '1'
				else:
					dir_s[u] = ['0']*depth
					rev_s[u] = ['0']*depth
					dir_s[u][tau] = '2'
					rev_s[u][tau] = '1'
			for u in both_ij:
				if u in dir_s:
					dir_s[u][tau] = '3'
					rev_s[u][tau] = '3'
				else:
					dir_s[u] = ['0']*depth
					rev_s[u] = ['0']*depth
					dir_s[u][tau] = '3'
					rev_s[u][tau] = '3'
		del dir_s[i]; del dir_s[j]
		del rev_s[i]; del rev_s[j]
		dir_seq = ''.join(sorted([''.join(val) for val in dir_s.values()]))
		rev_seq = ''.join(sorted([''.join(val) for val in rev_s.values()]))
		if dir_seq<rev_seq:
			return central_profile+dir_seq,list_interactions
		return central_profile+rev_seq,list_interactions

	#compute the binary string representration of the NCTN instance starting at time t of central node v
	#and given depth
	def Compute_NCTN_string(self,t,v,depth):
		#dic_s[u][tau] = '1' if u is a satellite of v at time t+tau and '0' else
		dic_s = {}
		for tau in range(depth):
			#if v is active (i.e. has at least one satellite)
			if v in self.TN[t+tau]:
				for u in self.TN[t+tau][v]:
					if u in dic_s:
						dic_s[u][tau] = '1'
					else:
						dic_s[u] = ['0']*depth
						dic_s[u][tau] = '1'
		return ''.join(sorted([''.join(val) for val in dic_s.values()]))

	#compute the sequence of the motif starting at time t of central edge ind
	def Compute_ECTN_string(self,t,ind,depth):
		i,j = ind; central_profile = ''
		#encode the activity profiles of the satellites: dic_s[u] = activity profile of satellite u
		dir_s = {}; rev_s = {}
		for tau in range(depth):
			if self.TN[t+tau].has_edge(i,j):
				central_profile += '1'
			else:
				central_profile += '0'
			ngh_i = set(); ngh_j = set()
			if i in self.TN[t+tau]:
				ngh_i = set(self.TN[t+tau].neighbors(i))
			if j in self.TN[t+tau]:
				ngh_j = set(self.TN[t+tau].neighbors(j))
			both_ij = ngh_i.intersection(ngh_j)
			only_i = ngh_i.difference(both_ij)
			only_j = ngh_j.difference(both_ij)
			for u in only_i:
				if u in dir_s:
					dir_s[u][tau] = '1'
					rev_s[u][tau] = '2'
				else:
					dir_s[u] = ['0']*depth
					rev_s[u] = ['0']*depth
					dir_s[u][tau] = '1'
					rev_s[u][tau] = '2'
			for u in only_j:
				if u in dir_s:
					dir_s[u][tau] = '2'
					rev_s[u][tau] = '1'
				else:
					dir_s[u] = ['0']*depth
					rev_s[u] = ['0']*depth
					dir_s[u][tau] = '2'
					rev_s[u][tau] = '1'
			for u in both_ij:
				if u in dir_s:
					dir_s[u][tau] = '3'
					rev_s[u][tau] = '3'
				else:
					dir_s[u] = ['0']*depth
					rev_s[u] = ['0']*depth
					dir_s[u][tau] = '3'
					rev_s[u][tau] = '3'
		del dir_s[i]; del dir_s[j]
		del rev_s[i]; del rev_s[j]
		dir_seq = ''.join(sorted([''.join(val) for val in dir_s.values()]))
		rev_seq = ''.join(sorted([''.join(val) for val in rev_s.values()]))
		if dir_seq<rev_seq:
			return central_profile+dir_seq
		return central_profile+rev_seq

	#add NCTN instances to histo
	def Add_NCTN(self,histo,central_nodes,t,depth):
		for v in central_nodes:
			seq = self.Compute_NCTN_string(t,v,depth)
			if seq in histo:
				histo[seq] += 1
			else:
				histo[seq] = 1

	def Add_ECTN(self,histo,central_edges,t,depth):
		for ind in central_edges:
			seq = self.Compute_ECTN_string(t,ind,depth)
			Increase_dic(histo,self.get_tuple_keys(t,seq))

	def Add_interactions_ECTN(self,mode_to_interactions,central_edges,t,depth,seq_to_mode):
		for ind in central_edges:
			seq,list_interactions = self.Compute_ECTN_env(t,ind,depth)
			if seq in seq_to_mode:
				for (i,j,tau) in list_interactions:
					if j<i:
						i,j = j,i
					mode_to_interactions[seq_to_mode[seq]].add((i,j,t+tau))

	#compute dic_NCTN
	def Get_dic_NCTN(self,depth):
		#histo[seq] = nb of occurrences of ETN seq in the whole temporal network
		histo = {}
		for t in range(len(self.TN)-depth+1):
			central_nodes = set(())
			for tau in range(depth):
				central_nodes = central_nodes.union(set(self.TN[t+tau].nodes))
			self.Add_NCTN(histo,central_nodes,t,depth)
		return histo

	#compute dic_ECTN (histo) ; if transverse==True, restrict to transverse motifs
	#if timestamps==True, keep track of the occurrence time of each motif:
	#dic_ECTN[t][seq] = nb of occurrences of seq at time t
	def Get_dic_ECTN(self,depth,trunc=None,transverse=False,timestamps=False):
		if timestamps:
			self.get_tuple_keys = lambda t,seq:(seq,t)
		else:
			self.get_tuple_keys = lambda t,seq:(seq,)
		#histo[seq] = nb of occurrences of ECTN seq in the whole temporal network
		histo = {}; list_times = range(len(self.TN)-depth+1)
		#cumulate the interactions on depth time steps to keep track of the central edges
		dic_central_edges = {}
		for tau in range(depth):
			for edge in self.TN[tau].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
		self.Add_ECTN(histo,dic_central_edges.keys(),0,depth)
		for t in list_times[1:]:
			#remove the edges from time t-1
			for edge in self.TN[t-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				dic_central_edges[ind] -= 1
			#add the edges from time t+depth-1
			for edge in self.TN[t+depth-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
			edges_to_remove = {ind for ind,nb in dic_central_edges.items() if nb<=0}
			for ind in edges_to_remove:
				del dic_central_edges[ind]
			#update the EdgeTN histogram
			self.Add_ECTN(histo,dic_central_edges.keys(),t,depth)
		if transverse:
			histo = {seq:nb for seq,nb in histo.items() if '3' in seq or ('1' in seq[depth:] and '2' in seq[depth:])}
		if trunc is not None:
			most_freq = sorted(histo.keys(),key=lambda seq:histo[seq],reverse=True)[:trunc]
			return {seq:histo[seq] for seq in most_freq}
		return histo

	#seq_to_mode[mode] = mode containing the ECTN seq
	#return mode_to_interactions[mode] = set of interactions (temporal edges) contained in the motifs of
	#mode_to_seq[mode]
	def Get_mode_to_interactions_ECTN(self,depth,seq_to_mode):
		mode_to_interactions = {}
		for mode in set(seq_to_mode.values()):
			mode_to_interactions[mode] = set()
		list_times = range(len(self.TN)-depth+1)
		#cumulate the interactions on depth time steps to keep track of the central edges
		dic_central_edges = {}
		for tau in range(depth):
			for edge in self.TN[tau].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
		self.Add_interactions_ECTN(mode_to_interactions,dic_central_edges.keys(),0,depth,seq_to_mode)
		for t in list_times[1:]:
			#remove the edges from time t-1
			for edge in self.TN[t-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				dic_central_edges[ind] -= 1
			#add the edges from time t+depth-1
			for edge in self.TN[t+depth-1].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if ind in dic_central_edges:
					dic_central_edges[ind] += 1
				else:
					dic_central_edges[ind] = 1
			edges_to_remove = {ind for ind,nb in dic_central_edges.items() if nb<=0}
			for ind in edges_to_remove:
				del dic_central_edges[ind]
			#update the EdgeTN histogram
			self.Add_interactions_ECTN(mode_to_interactions,dic_central_edges.keys(),t,depth,seq_to_mode)
		return mode_to_interactions

	#return data (same format as self.data) with the temporal edges of temporal_edges removed
	def Remove_interactions_from(self,temporal_edges):
		data = []
		for t,graph in enumerate(self.TN):
			for edge in graph.edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				if (*ind,t) in temporal_edges:
					pass
				else:
					data.append([t,*ind])
		return np.array(data,dtype=int)

	#histo[seq] = probability that an edge at a starting time has seq as activity profile
	#(empty profile excluded)
	def Get_act_profile_proba_NCTN(self,dic_NCTN,depth):
		histo = {}
		histo = {seq:0 for seq in ETN.All_NCTN_profiles(depth)}
		for seq,nb in dic_NCTN.items():
			for i in range(len(seq)//depth):
				histo[seq[i*depth:(i+1)*depth]] += nb
		return Norm_dic_histo(histo)

	#return the node degree distribution, degree zero included if zero==True
	def Get_direct_deg_histo(self,zero=False):
		res = {}
		for graph in self.TN:
			for i in graph.nodes:
				n = graph.degree(i)
				if n in res:
					res[n] += 1
				else:
					res[n] = 1
			if zero:
				n = self.N-len(graph.nodes)
				if 0 in res:
					res[0] += n
				else:
					res[0] = n
		return Norm_dic_histo(res)

	#return the distribution of the degree aggregated on depth, degree zero excluded
	def Get_inst_deg_proba(self,dic_NCTN,depth):
		#histogram of the nb of satellites (aggregated degree at agg=depth)
		histo_sat = {}
		for seq,nb in dic_NCTN.items():
			nb_sat = len(seq)//depth
			if nb_sat in histo_sat:
				histo_sat[nb_sat] += nb
			else:
				histo_sat[nb_sat] = nb
		norm = sum(histo_sat.values())
		for key,val in histo_sat.items():
			histo_sat[key] = val/norm
		return histo_sat

	#compute the NCTN proba under the strong spatial ind hypothesis
	#return X,Y where X = true proba, Y = th proba
	def Get_strong_ind_NCTN(self,dic_NCTN,depth):
		#for each profile, compute the average number of given profile a NCTN contains
		profiles = All_NCTN_profiles(depth)
		avg_nb_profiles = {profile:0 for profile in profiles}
		var_nb_profiles = {profile:0 for profile in profiles}
		#nb_profile[prof] = nb of satellites with prof as activity profile in a NCTN
		nb_profile = {profile:0 for profile in profiles}
		#convert the string into the map representation
		string_to_map = {}; norm = sum(dic_NCTN.values())
		for seq,proba in dic_NCTN.items():
			for i in range(len(seq)//depth):
				nb_profile[seq[i*depth:(i+1)*depth]] += 1
			string_to_map[seq] = [nb_profile[profile] for profile in profiles]
			for profile,nb in nb_profile.items():
				avg_nb_profiles[profile] += nb*proba/norm
				var_nb_profiles[profile] += nb*(nb-1)*proba/norm
			for key in nb_profile.keys():
				nb_profile[key] = 0
		prof_to_lamb = {profile:var_nb_profiles[profile]/nb for profile,nb in avg_nb_profiles.items()}
		X = []; Y = []
		for seq,proba in dic_NCTN.items():
			X.append(proba/norm)
			th_proba = 1
			for nb,profile in zip(string_to_map[seq],profiles):
				th_proba *= prof_to_lamb[profile]**nb/math.factorial(nb)
			Y.append(th_proba)
		return X,np.asarray(Y)/sum(Y)

	#compute the NCTN proba under the weak spatial ind hypothesis
	#return X,Y where X = true proba, Y = th proba
	def Get_weak_ind_NCTN(self,dic_NCTN,depth):
		#compute the proba of each profile
		central_histo,sat_histo = self.Get_act_profile_proba_NCTN(dic_NCTN,depth)
		#compute the distribution of the degree aggregated on depth
		inst_deg_proba = self.Get_inst_deg_proba(dic_NCTN,depth)
		#deduce the theoretical proba:
		profiles = All_NCTN_profiles(depth)
		#nb_profile[prof] = nb of satellites with prof as activity profile in a NCTN
		nb_profile = {profile:0 for profile in profiles}
		#convert the string into the map representation
		string_to_map = {}
		for seq in dic_NCTN.keys():
			for i in range(len(seq)//depth):
				nb_profile[seq[i*depth:(i+1)*depth]] += 1
			string_to_map[seq] = [nb_profile[profile] for profile in profiles]
			for key in nb_profile.keys():
				nb_profile[key] = 0
		X = []; Y = []; norm = sum(dic_NCTN.values())
		for seq,proba in dic_NCTN.items():
			X.append(proba/norm)
			nb_sat = sum(string_to_map[seq])
			th_proba = math.factorial(nb_sat)*inst_deg_proba[nb_sat]
			for nb,profile in zip(string_to_map[seq],profiles):
				th_proba *= act_profile_proba[profile]**nb/math.factorial(nb)
			Y.append(th_proba)
		return X,Y

	#compute the ECTN proba under the ind hypothesis 1
	#return X,Y where X = true proba, Y = th proba
	#take care that the aggregation level is reset to agg
	def Get_ind1_ECTN(self,list_seq,depth,datapath,restrict,agg=1):
		dic_ECTN = Load_dic_ECTN(datapath,restrict)
		#compute the distribution of the nb of satellites
		Nb_sat_histo = {}
		for seq,nb in dic_ECTN.items():
			deg = len(seq)//depth-1
			if deg in Nb_sat_histo:
				Nb_sat_histo[deg] += nb
			else:
				Nb_sat_histo[deg] = nb
		Nb_sat_histo = Norm_dic_histo(Nb_sat_histo)
		#compute edge profile histo
		prof_proba = self.Get_act_profile_proba_NCTN(self.Get_dic_NCTN(depth),depth)
		#Q_0 = proba of having an empty edge profile
		self.Get_TN(depth)
		Q_0 = 1-np.mean([len(graph.edges)*2/(self.N*(self.N-1)) for graph in self.TN])
		self.Get_TN(agg)
		for key,val in prof_proba.items():
			prof_proba[key] = val*(1-Q_0)
		prof_proba['0'*depth] = Q_0
		###################################
		nb_profile = {profile:0 for profile in ETN.All_ECTN_profiles(depth)[1]}
		Y = []
		for seq in list_seq:
			nb_sat = len(seq)//depth-1
			for i in range(1,len(seq)//depth):
				new_seq = seq[i*depth:(i+1)*depth]
				nb_profile[new_seq] += 1
			th_proba = (prof_proba[seq[:depth]]/(1-Q_0))*math.factorial(nb_sat)*Nb_sat_histo[nb_sat]
			for sat_profile,nb in nb_profile.items():
				if nb>0:
					prof1,prof2 = ETN.ECTN_to_NCTN_prof(sat_profile)
					x = prof_proba[prof1]*prof_proba[prof2]/(1-Q_0**2)
					th_proba *= (x**nb)/math.factorial(nb)
			if seq!=ETN.Is_12_sym(seq,depth):
				th_proba *= 2
			Y.append(th_proba)
			for key in nb_profile.keys():
				nb_profile[key] = 0
		return Y

	def Get_ind2_ECTN(self,datapath,depth,restrict):
		dic_ECTN = Load_dic_ECTN(datapath,restrict)
		nb_states = 2
		hyp2 = Interpolate_distr(nb_states)
		hyp2.preprocess(dic_ECTN,depth)
		print('preprocess done')
		param = hyp2.fit_model()
		hyp2.set_param(param)
		print('fit done')
		#return th proba
		return hyp2.predict_for_fit()*hyp2.list_factors

	#hyp1: edges are independent: P_12 = Q_10*Q_01
	#hyp2: nodes are independent
	#hyp6: ECTN are induced from NCTN according to the maximum entropy principle (MEP)
	def Get_ind_ECTN(self,list_seq,depth,hyp_nb,restrict,datapath):
		if hyp_nb==1:
			return self.Get_ind1_ECTN(list_seq,depth,datapath,restrict)
		elif hyp_nb==2:
			return self.Get_ind2_ECTN(datapath,depth,restrict)
		elif hyp_nb==6:
			return Get_indMEP_ECTN(self.Get_dic_NCTN(depth),depth,list_seq,restrict)

	#compute the train of events for nodes
	#node_event[ind][k] = (t_0,t_f) with t_0 = starting time of the k^th event for node of identifier ind
	#and t_f = ending time
	def Node_event_train(self):
		node_event = {}
		active_nodes = set(self.TN[0].nodes)
		node_starting_time = {i:0 for i in active_nodes}
		for t in range(1,len(self.TN)):
			current_nodes = set(self.TN[t].nodes)
			#edges active at t and not at t-1
			new_nodes = current_nodes.difference(active_nodes)
			#edges active at t-1 and not at t
			finished_nodes = active_nodes.difference(current_nodes)
			for ind in finished_nodes:
				if ind in node_event:
					node_event[ind].append((node_starting_time[ind],t-1))
				else:
					node_event[ind] = [(node_starting_time[ind],t-1)]
				del node_starting_time[ind]
			for ind in new_nodes:
				node_starting_time[ind] = t
			#update the set of active nodes
			active_nodes = current_nodes
		for ind in active_nodes:
			if ind in node_event:
				node_event[ind].append((node_starting_time[ind],len(self.TN)-1))
			else:
				node_event[ind] = [(node_starting_time[ind],len(self.TN)-1)]
		return node_event

	#compute the train of events for edges
	#edge_event[ind][k] = (t_0,t_f) with t_0 = starting time of the k^th event for edge of identifier ind
	#and t_f = ending time
	def Edge_event_train(self):
		edge_event = {}
		#edge_starting_time[ind] = last starting time of the activation of the edge of identifier ind
		edge_starting_time = {}
		#active_edges = set of edges active at current time
		active_edges = set(())
		for edge in self.TN[0].edges:
			if edge[0]<edge[1]:
				ind = edge
			else:
				ind = edge[::-1]
			active_edges.add(ind)
			edge_starting_time[ind] = 0
		for t in range(1,len(self.TN)):
			#edges active at t
			current_edges = set(())
			for edge in self.TN[t].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				current_edges.add(ind)
			#edges active at t and not at t-1
			new_edges = current_edges.difference(active_edges)
			#edges active at t-1 and not at t
			finished_edges = active_edges.difference(current_edges)
			for ind in finished_edges:
				if ind in edge_event:
					edge_event[ind].append((edge_starting_time[ind],t-1))
				else:
					edge_event[ind] = [(edge_starting_time[ind],t-1)]
				del edge_starting_time[ind]
			for ind in new_edges:
				edge_starting_time[ind] = t
			#update the set of active edges
			active_edges = current_edges
		#take care of the last timestamp
		for ind in active_edges:
			if ind in edge_event:
				edge_event[ind].append((edge_starting_time[ind],len(self.TN)-1))
			else:
				edge_event[ind] = [(edge_starting_time[ind],len(self.TN)-1)]
		return edge_event

	#draw the degree profile for some nodes
	def Draw_degree_profile(self):
		#for each node extract the degree profile: deg_profile[i,t] = degree of i at time t
		deg_profile = np.zeros((self.N,len(self.TN)),dtype=int)
		for t,graph in enumerate(self.TN):
			for i in graph.nodes:
				deg_profile[i,t] = graph.degree(i)
		#plot histo for d_t+1 - d_t
		histo = {}
		for i in range(self.N):
			for t in range(np.size(deg_profile,1)-1):
				el = abs(deg_profile[i,t+1]-deg_profile[i,t])
				#el = deg_profile[i,t+2]-deg_profile[i,t]
				if el in histo:
					histo[el] += 1
				else:
					histo[el] = 1
		fig,ax = plt.subplots(constrained_layout=True)
		X,Y = zip(*histo.items())
		ax.plot(X,np.log(np.log(Y)),'.')
		#ax.plot(*zip(*histo.items()),'.')
		plt.show()
		exit()
		#plot degree profiles
		fig,ax = plt.subplots(constrained_layout=True)
		for i in rd.sample(range(self.N),5):
			ax.plot(deg_profile[i,:],label=str(i))
		ax.legend()
		plt.show()

	#estimate the autocorrelation time of node degree by plotting max(abs(d_t+tau-d_t))/max(d_t) vs tau
	def Draw_degree_profile(self):
		#for each node extract the degree profile: deg_profile[i,t] = degree of i at time t
		deg_profile = np.zeros((self.N,len(self.TN)),dtype=int)
		for t,graph in enumerate(self.TN):
			for i in graph.nodes:
				deg_profile[i,t] = graph.degree(i)
		list_tau = np.asarray(range(1,21)); deg_max = np.max(deg_profile); Y = []
		for tau in list_tau:
			M = 0
			for t in range(np.size(deg_profile,1)-tau):
				diff = np.max(abs(deg_profile[:,t+tau]-deg_profile[:,t]))
				if diff>M:
					M = diff
			Y.append(M/deg_max)
		Y = np.asarray(Y)
		fig,ax = plt.subplots(constrained_layout=True)
		#ax.plot(list_tau,np.log(1-Y),'.')
		ax.plot(list_tau,Y,'.')
		ax.plot([list_tau[0],list_tau[-1]],[1]*2,'--')
		plt.show()
		exit()
		#plot degree profiles
		fig,ax = plt.subplots(constrained_layout=True)
		for i in rd.sample(range(self.N),5):
			ax.plot(deg_profile[i,:],label=str(i))
		ax.legend()
		plt.show()

	#return res[seq] = proba of seq under the weak ind hyp, where the seq are all the NCTN
	#of proba<=min_proba
	def Get_dic_NCTN_seq_weak_ind_hyp(self,dic_NCTN,depth,min_proba,deg_max=28):
		#compute the proba of each profile
		act_profile_proba = self.Get_act_profile_proba_NCTN(dic_NCTN,depth)
		#compute the distribution of the degree aggregated on depth
		inst_deg_proba = self.Get_inst_deg_proba(dic_NCTN,depth)
		#deduce the theoretical proba:
		profiles = All_NCTN_profiles(depth)
		#nb_profile[i] = nb of satellites with profiles[i] as activity profile in a NCTN
		nb_profile = [0]*len(profiles)
		res = {}
		#browse the set of NCTN tuples
		for nb_sat,prob_deg in inst_deg_proba.items():
			print(nb_sat)
			if nb_sat<=deg_max:
				th_proba = math.factorial(nb_sat)*prob_deg
				#browse the set of non empty tuples whose sum of components is = nb_sat
				max_tuple = [nb_sat]+[0]*(len(profiles)-1)
				current_tuple = [0]*(len(profiles)-1)+[nb_sat]
				while current_tuple[:]!=max_tuple[:]:
					add_prob = 1
					for nb,profile in zip(current_tuple,profiles):
						add_prob *= act_profile_proba[profile]**nb/math.factorial(nb)
					seq_prob = th_proba*add_prob
					#consider only the NCTN with proba>=min_proba
					if seq_prob>=min_proba:
						#convert the tuple into a NCTN string
						seq = ''
						for nb,profile in zip(current_tuple,profiles):
							seq += nb*profile
						res[seq] = seq_prob
					#compute the successor of current_tuple:
					#find ind such that current_tuple[i]=0 for all i>ind
					ind = len(profiles)-1
					while current_tuple[ind]==0:
						ind -= 1
					if ind<len(profiles)-1:
						current_tuple[ind] = 0
						current_tuple[ind-1] += 1
						current_tuple[-1] = nb_sat-sum(current_tuple[:-1])
					else:
						current_tuple[-1] -= 1
						current_tuple[-2] += 1
				#handle the last tuple (max_tuple)
				add_prob = 1
				for nb,profile in zip(current_tuple,profiles):
					add_prob *= act_profile_proba[profile]**nb/math.factorial(nb)
				seq_prob = th_proba*add_prob
				#consider only the NCTN with proba>=min_proba
				if seq_prob>=min_proba:
					#convert the tuple into a NCTN string
					seq = ''
					for nb,profile in zip(current_tuple,profiles):
						seq += nb*profile
					res[seq] = seq_prob
		return res

	#dic_profile[edge] = full profile of the edge (binary string of 0 and 1)
	def Full_profile_edges(self):
		dic_profile = {}
		for t,graph in enumerate(self.TN):
			for edge in dic_profile:
				if not graph.has_edge(*edge):
					dic_profile[edge] += '0'
			for edge in graph.edges:
				if edge[0]<edge[1]:
					key = edge
				else:
					key = edge[::-1]
				if key in dic_profile:
					dic_profile[key] += '1'
				else:
					dic_profile[key] = '0'*t+'1'
		return dic_profile

	#compute the histo for edge activity profiles (empty profile excluded or included)
	def Edge_act_prof_histo(self,depth,empty=False):
		res = {}
		#full profile of edges (strings of 0 and 1)
		full_profiles = self.Full_profile_edges().values()
		for full_prof in full_profiles:
			for t in range(len(full_prof)-depth+1):
				profile = full_prof[t:t+depth]
				if profile in res:
					res[profile] += 1
				else:
					res[profile] = 1
		if not empty:
			del res['0'*depth]
		return res

	#compute the histo for ECTN satellite profiles at depth 3, with letters 1 and 2 identified
	def ECTN_sat_prof_histo(self,depth,transverse=False):
		if depth!=3:
			raise ValueError('sorry, not implemented yet for depth!=3 ^^')
		dic_ECTN = self.Get_dic_ECTN(depth,transverse=transverse)
		sat_prof = ['100','001','110','011','300','003','130','031','113','311','330','033','331','133']
		res = {profile:0 for profile in sat_prof}
		for seq,nb in dic_ECTN.items():
			for i in range(1,len(seq)//depth):
				new_seq = seq[i*depth:(i+1)*depth]
				profile = ''
				for letter in new_seq:
					if letter=='2':
						profile += '1'
					else:
						profile += letter
				if profile in res:
					res[profile] += nb
		return res

	#compute the distribution of the size of connected components of the interaction graph
	def Get_cc_size(self):
		cc_histo = {}
		for G in self.TN:
			list_cc = nx.connected_components(G)
			for cc in list_cc:
				n = len(cc)
				if n in cc_histo:
					cc_histo[n] += 1
				else:
					cc_histo[n] = 1
		return cc_histo



class Temp_net:
	"""
	"""
	def __init__(self,dataset):
		#allows to keep some info in memory to hopefully fasten computations
		self.memory = {}
		for obj in ['node','edge',('NCTN',3),('ECTN',3)]:
			self.memory[obj] = {}
			for el in ['train','time_weight']:
				self.memory[obj][el] = {}
		self.Compute_object = {'edge':{'train':self.Edge_event_train,'time_weight':self.Edge_dic_weight}}
		self.Compute_object['node'] = {'train':self.Node_event_train,'time_weight':self.Node_dic_weight}
		self.Compute_object['NCTN'] = {'train':self.ETN_event_train,'time_weight':self.ETN_histo}
		self.Compute_object['ECTN'] = {'train':self.EdgeTN_event_train,'time_weight':self.EdgeTN_histo}
		self.Get_histo = {'duration':self.Duration_histo,'interduration':self.Interduration_histo,'event_duration':self.Events_duration_histo}
		self.Get_histo['time_weight'] = self.Weight_histo
		self.Get_histo['newborn_duration'] = self.Newborn_duration_histo
		self.Get_histo['profile_from_motif'] = self.Profile_from_motif
		self.Get_histo['profile_from_events'] = self.Profile_from_events
		self.Get_histo['inst_deg'] = self.Get_inst_deg
		self.Get_histo['cc_size'] = self.Get_cc_size
		self.Get_histo['cum_edge_weight'] = self.Get_cum_edge_weight_single
		self.Get_histo['node_space_weight'] = self.Get_node_space_weight
		self.Get_histo['edge_space_weight'] = self.Get_edge_space_weight
		self.Get_val = {'motif_error':self.Get_motif_error,'entropy':self.Get_entropy}
		self.Get_val['nb_diff'] = self.Get_nb_diff
		self.Get_val['nb_tot'] = self.Get_nb_tot
		self.Get_val['ICC'] = self.Get_val_ICC
		self.Get_val['avg_cc'] = self.Get_val_avg_cc
		#dic_obs[type_obs][obs] = observable
		self.dic_obs = {'point':{},'distribution':{},'vector':{}}
		#measurement data
		self.data = dataset
		#interaction temporal network
		self.TN = {}
		#data_time[i] = [t_i, n_1, n_p] with t_i the (i+1)th time appearing in data
		#n_1 the first line of occurrence of t_i and n_p-1 the last one
		self.data_time = []
		#info contains :
		#number of nodes
		#number of timestamps
		self.info = {}

	#replace self.data by its formatted version, i.e. time begins at 0, two consecutive times
	#are separated by one and nodes are numeroted from 0 to nb of nodes-1
	#and self-loops are removed
	def Format(self,duration=np.inf):
		self.Get_data_time()
		first_line = max(0,len(self.data_time)-duration)
		self.data = self.data[self.data_time[first_line][1]:,:]
		#remove any self-loop
		valid_lines = []
		for n in range(np.size(self.data,0)):
			if self.data[n,1]!=self.data[n,2]:
				valid_lines.append(n)
		self.data = self.data[valid_lines,:]
		#recompute data_time
		self.Get_data_time()
		#relabel the nodes
		node_to_int = {}; num = 0
		for n in range(np.size(self.data,0)):
			for k in self.data[n,1:]:
				if not k in node_to_int:
					node_to_int[k] = num
					num += 1
		for t,el in enumerate(self.data_time):
			self.data[el[1]:el[2],0] = t
		for n in range(np.size(self.data,0)):
			for k in range(1,3):
				self.data[n,k] = node_to_int[self.data[n,k]]

	#compute self.data_time
	def Get_data_time(self):
		self.data_time = []
		n1 = 0; n_max = np.size(self.data,0)
		for n in range(1,n_max):
			t = self.data[n-1,0]
			if self.data[n,0]>t:
				self.data_time += [[t,n1,n]]
				n1 = n
		#take care of the last line of data
		t = self.data[-1,0]
		self.data_time += [[t,n1,n_max]]

	#extract the profile histo with given depth from the events train of obj
	def Profile_from_events(self,key_mem):
		obj,depth = key_mem
		P_to_int = All_NCTN_profiles(depth)
		P_to_int['0'*depth] = 0
		histo = {num:0 for num in P_to_int.values()}
		for val in self.memory[obj]['train'].values():
			last_time = 0
			tot_profile = ''
			for el in val:
				tot_profile += '0'*(el[0]-last_time)
				tot_profile += '1'*(el[1]-el[0]+1)
				last_time = el[1]+1
			for t in range(len(tot_profile)-depth+1):
				num = P_to_int[tot_profile[t:t+depth]]
				histo[num] += 1
		return histo

	#extract the edge profile histo from dic_ETN
	def Profile_from_NCTN(self,depth):
		P_to_int = All_NCTN_profiles(depth)
		histo = {i:0 for i in P_to_int.values()}
		for seq,nb in self.memory[('NCTN',depth)]['time_weight'].items():
			for i in range(len(seq)//depth):
				histo[P_to_int[seq[i*depth:(i+1)*depth]]] += nb
		return histo

	#extract the edge profile histo from dic_EdgeTN
	def Profile_from_ECTN(self,depth):
		Psat_to_int,Pcentral_to_int = All_ECTN_profiles(depth)
		histo = {i:0 for i in list(Psat_to_int.values())+list(Pcentral_to_int.values())}
		for seq,nb in self.memory[('ECTN',depth)]['time_weight'].items():
			histo[Pcentral_to_int[seq[:depth]]] += nb
			seq_sat = seq[depth:]
			for i in range(len(seq_sat)//depth):
				histo[Psat_to_int[seq_sat[i*depth:(i+1)*depth]]] += nb
		return histo

	#extract the edge profile histo from either dic_EdgeTN or dic_ETN
	def Profile_from_motif(self,key_mem):
		obj,depth = key_mem
		if obj=='NCTN':
			return self.Profile_from_NCTN(depth)
		elif obj=='ECTN':
			return self.Profile_from_ECTN(depth)

	#compute the total nb of ETN
	def Get_nb_tot(self,obj):
		return sum(list(self.memory[obj]['time_weight'].values()))

	#compute the nb of different ETN
	def Get_nb_diff(self,obj):
		return len(self.memory[obj]['time_weight'])

	def Clear_memory(self):
		for key1,val in self.memory.items():
			for key2 in val.keys():
				self.memory[key1][key2] = {}

	#returns the histo of the total nb of nodes active at the same time
	def Get_node_space_weight(self):
		histo = {}
		for val in self.TN.values():
			n = len(val.nodes)
			if n in histo:
				histo[n] += 1
			else:
				histo[n] = 1
		return histo

	#returns the histo of the total nb of edges active at the same time
	def Get_edge_space_weight(self):
		histo = {}
		for val in self.TN.values():
			n = len(val.edges)
			if n in histo:
				histo[n] += 1
			else:
				histo[n] = 1
		return histo

	def Get_obs(self,obs,arg):
		#break the observable into meaningful chunks: the first chunk gives the object to compute
		#and the second tells how to compute the observable using this object
		#observable standard encoding: list of objects to compute followed by the property of the objects
		#we want to sample (note that there may be no object to compute, i.e. chunk_obj may be an empty list)
		chunk_obj,chunk_prop = Obs_to_chunks(obs)
		if chunk_obj:
			if not arg:
				key_mem = chunk_obj[0]
			else:
				key_mem = (chunk_obj[0],*arg)
			if not key_mem in self.memory:
				self.memory[key_mem] = {'train':{},'time_weight':{}}
				self.memory[key_mem][chunk_obj[1]] = self.Compute_object[chunk_obj[0]][chunk_obj[1]](*arg)
			elif not self.memory[key_mem][chunk_obj[1]]:
				self.memory[key_mem][chunk_obj[1]] = self.Compute_object[chunk_obj[0]][chunk_obj[1]](*arg)
			tot_arg = tuple([chunk_obj[0]])+arg
		else:
			tot_arg = arg
		if not tot_arg:
			if chunk_prop in self.Get_histo:
				return self.Get_histo[chunk_prop]()
			elif chunk_prop in self.Get_val:
				return self.Get_val[chunk_prop]()
		elif len(tot_arg)==1:
			key_mem = tot_arg[0]
		else:
			key_mem = tot_arg
		if chunk_prop in self.Get_histo:
			return self.Get_histo[chunk_prop](key_mem)
		elif chunk_prop in self.Get_val:
			return self.Get_val[chunk_prop](key_mem)

	def Get_val_ICC(self):
		return np.mean([nx.average_clustering(G) for G in self.TN.values()])

	#compute the average nb of cc per time step
	def Get_val_avg_cc(self):
		return np.mean([len(list(nx.connected_components(G))) for G in self.TN.values()])

	def Get_entropy(self,obj):
		list_proba = np.asarray(list(self.memory[obj]['time_weight'].values()),dtype=float)
		norm = np.sum(list_proba)
		list_proba = list_proba/norm
		return np.sum(-list_proba*np.log10(list_proba))

	#return the error made on ETN proba by assuming independence of activity profiles
	#as well as independence btw nb of satellites and activity profiles
	def Get_motif_error(self,key_mem):
		obj,depth = key_mem
		if obj=='NCTN':
			abundancy,list_proba = self.NCTN_xpth_probas(depth)
		elif obj=='ECTN':
			abundancy,list_proba = self.ECTN_xpth_probas(depth)
		return np.sqrt(np.sum((abundancy-list_proba)**2)/len(list_proba))

	def Get_cum_edge_weight_single(self,cum):
		nb_blocks = len(self.TN)//cum
		#histo[w] = nb of occurrences such that an edge is active w times on a time interval of length cum
		histo = {}
		for t in range(nb_blocks):
			cum_net = nx.Graph()
			for tau in range(t*cum,(t+1)*cum):
				for edge in self.TN[tau].edges:
					if cum_net.has_edge(*edge):
						cum_net[edge[0]][edge[1]]['weight'] += 1
					else:
						cum_net.add_edge(*edge,weight=1)
			for edge in cum_net.edges:
				w = cum_net[edge[0]][edge[1]]['weight']
				if w in histo:
					histo[w] += 1
				else:
					histo[w] = 1
		return histo

	def NCTN_histo_sat(self,depth):
		histo_sat = {}
		for seq,nb in self.memory[('NCTN',depth)]['time_weight'].items():
			nb_sat = len(seq)//depth
			if nb_sat in histo_sat:
				histo_sat[nb_sat] += nb
			else:
				histo_sat[nb_sat] = nb
		return histo_sat

	def ECTN_histo_sat(self,depth):
		histo_sat = {}
		for seq,nb in self.memory[('ECTN',depth)]['time_weight'].items():
			nb_sat = len(seq)//depth-1
			if nb_sat in histo_sat:
				histo_sat[nb_sat] += nb
			else:
				histo_sat[nb_sat] = nb
		return histo_sat

	#sliding time shuffling (STS) with range b
	#followed by sliding time aggregation of level agg
	#modifies self.TN as well as self.data_time but data is preserved
	def Sliding_transfo(self,b,agg):
		#compute self.data_time
		self.Get_data_time()
		nb_time = len(self.data_time)
		#sliding time shuffling
		list_times = list(range(nb_time))
		for k in range(nb_time-b+1):
			list_times[k:k+b] = rd.sample(list_times[k:k+b],b)
		new_data_time = [[list_times[el[0]],*el[1:]] for el in self.data_time]
		self.data_time = sorted(new_data_time,key=lambda el:el[0])
		#sliding time aggregation
		new_nb = nb_time-agg+1
		#self.TN[t] = aggregated graph of interactions on t^th time interval
		self.TN = {t:nx.Graph() for t in range(new_nb)}
		#initial graph
		G = nx.Graph()
		for k in range(agg):
			for n in range(*self.data_time[k][1:]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		for t in range(new_nb-1):
			self.TN[t].add_edges_from(G.edges)
			#remove edges from time t
			for n in range(*self.data_time[t][1:]):
				i,j = self.data[n,1:]
				if G[i][j]['weight']==1:
					G.remove_edge(i,j)
				else:
					G[i][j]['weight'] -= 1
			#add edges from time t+agg
			for n in range(*self.data_time[t+agg][1:]):
				i,j = self.data[n,1:]
				if G.has_edge(i,j):
					G[i][j]['weight'] += 1
				else:
					G.add_edge(i,j,weight=1)
		self.TN[new_nb-1].add_edges_from(G.edges)
		self.info['T'] = new_nb

	#local time shuffling (TS) of self.TN with time window b
	#modifies self.TN as well as self.data_time but data is preserved
	def Local_TS(self,b,agg=1):
		#compute self.data_time
		self.Get_data_time()
		#compute the new self.data_time and deduce self.TN at agg level 1
		#determine the times permutation
		list_times = []; nb_blocks = len(self.data_time)//b
		for k in range(nb_blocks):
			list_times += rd.sample(range(k*b,(k+1)*b),b)
		last_block = len(self.data_time)-b*nb_blocks
		if last_block>0:
			list_times += rd.sample(range(b*nb_blocks,len(self.data_time)),last_block)
		new_data_time = [[list_times[el[0]],*el[1:]] for el in self.data_time]
		self.data_time = sorted(new_data_time,key=lambda el:el[0])
		self.Get_TN(agg)

	#minimum intialization to get the temporal network at aggregation level agg
	def Init(self,agg=None):
		self.Get_data_time()
		self.info['N'] = len(set(self.data[:,1]).union(set(self.data[:,2])))
		if type(agg)==int:
			self.Get_TN(agg)

	#return the cumulated interaction graph at level cum
	def Get_cum_TN(self,cum):
		#split the timeline into blocks of size cum
		new_nb = len(self.TN)//cum
		cum_TN = {t:nx.Graph() for t in range(new_nb)}
		for t in range(new_nb):
			for k in range(cum):
				n1,n2 = self.data_time[t*cum+k][1:]
				for n in range(n1,n2):
					i,j = self.data[n,1:]
					if cum_TN[t].has_edge(i,j):
						cum_TN[t][i][j]['weight'] += 1
					else:
						cum_TN[t].add_edge(i,j,weight=1)
		return cum_TN

	#compute the interaction graph at aggregation level agg
	def Get_TN(self,agg):
		new_nb = len(self.data_time)//agg
		#self.TN[t] = aggregated graph of interactions on t^th time interval
		self.TN = {t:nx.Graph() for t in range(new_nb)}
		for t in range(new_nb):
			for k in range(agg):
				n1,n2 = self.data_time[t*agg+k][1:]
				for n in range(n1,n2):
					self.TN[t].add_edge(*self.data[n,1:])
		#take care of the last aggregation window
		if new_nb*agg<len(self.data_time):
			self.TN[new_nb] = nx.Graph()
			for k in range(len(self.data_time)-agg*new_nb):
				n1,n2 = self.data_time[agg*new_nb+k][1:]
				for n in range(n1,n2):
					self.TN[new_nb].add_edge(*self.data[n,1:])
		self.info['T'] = len(self.TN)

	def Edge_to_ind(self,i,j):
		if i>j:
			i,j = j,i
		return i*(2*self.info['N']-i-1)//2 + j-i-1

	#assumes self.Get_TN has been run
	#compute the histogram for instantaneous node degree
	def Get_inst_deg(self):
		histo = {}
		for val in self.TN.values():
			#count the nb of nodes with degree zero (these are the inactive nodes)
			nb = self.info['N']-len(val.nodes)
			if nb:
				histo[0] = nb
			for i in val.nodes:
				deg = val.degree(i)
				if deg in histo:
					histo[deg] += 1
				else:
					histo[deg] = 1
		return histo

	#compute the train of events for nodes
	def Node_event_train(self):
		node_event = {}
		active_nodes = set(self.TN[0].nodes)
		node_starting_time = {i:0 for i in active_nodes}
		for t in range(1,len(self.TN)):
			current_nodes = set(self.TN[t].nodes)
			#edges active at t and not at t-1
			new_nodes = current_nodes.difference(active_nodes)
			#edges active at t-1 and not at t
			finished_nodes = active_nodes.difference(current_nodes)
			for ind in finished_nodes:
				if ind in node_event:
					node_event[ind].append((node_starting_time[ind],t-1))
				else:
					node_event[ind] = [(node_starting_time[ind],t-1)]
				del node_starting_time[ind]
			for ind in new_nodes:
				node_starting_time[ind] = t
			#update the set of active nodes
			active_nodes = current_nodes
		for ind in active_nodes:
			if ind in node_event:
				node_event[ind].append((node_starting_time[ind],self.info['T']-1))
			else:
				node_event[ind] = [(node_starting_time[ind],self.info['T']-1)]
		return node_event

	#compute the train of events for edges
	#edge_event[ind][k] = (t_0,t_f) with t_0 = starting time of the k^th event for edge of identifier ind
	#and t_f = ending time
	def Edge_event_train(self):
		edge_event = {}
		#edge_starting_time[ind] = last starting time of the activation of the edge of identifier ind
		edge_starting_time = {}
		#active_edges = set of edges active at current time
		active_edges = set(())
		for edge in self.TN[0].edges:
			if edge[0]<edge[1]:
				ind = edge
			else:
				ind = edge[::-1]
			active_edges.add(ind)
			edge_starting_time[ind] = 0
		for t in range(1,len(self.TN)):
			#edges active at t
			current_edges = set(())
			for edge in self.TN[t].edges:
				if edge[0]<edge[1]:
					ind = edge
				else:
					ind = edge[::-1]
				current_edges.add(ind)
			#edges active at t and not at t-1
			new_edges = current_edges.difference(active_edges)
			#edges active at t-1 and not at t
			finished_edges = active_edges.difference(current_edges)
			for ind in finished_edges:
				if ind in edge_event:
					edge_event[ind].append((edge_starting_time[ind],t-1))
				else:
					edge_event[ind] = [(edge_starting_time[ind],t-1)]
				del edge_starting_time[ind]
			for ind in new_edges:
				edge_starting_time[ind] = t
			#update the set of active edges
			active_edges = current_edges
		#take care of the last timestamp
		for ind in active_edges:
			if ind in edge_event:
				edge_event[ind].append((edge_starting_time[ind],self.info['T']-1))
			else:
				edge_event[ind] = [(edge_starting_time[ind],self.info['T']-1)]
		return edge_event

	#compute the observables of evaluation
	def Evaluation_obs(self):
		self.eval_obs['vector']['ETN3'] = self.ETN_vector(5,3)[0]

	#compute the distribution of the size of connected components of the interaction graph
	def Get_cc_size(self):
		cc_histo = {}
		for G in self.TN.values():
			list_cc = nx.connected_components(G)
			for cc in list_cc:
				n = len(cc)
				if n in cc_histo:
					cc_histo[n] += 1
				else:
					cc_histo[n] = 1
		return cc_histo

	#compute the train of events for EdgeTN
	def EdgeTN_event_train(self,depth):
		ETN_event = {}
		#active_ETN = set of ETN active at current time
		active_ETN = self.Get_current_EdgeTN(0,depth)
		#ETN_starting_time[seq] = last starting time of the activation of the ETN of signature seq
		ETN_starting_time = {seq:0 for seq in active_ETN}
		for t in range(1,len(self.TN)-depth+1):
			#ETN active at t
			current_ETN = self.Get_current_EdgeTN(t,depth)
			#ETN active at t and not at t-1
			new_ETN = current_ETN.difference(active_ETN)
			#ETN active at t-1 and not at t
			finished_ETN = active_ETN.difference(current_ETN)
			for seq in finished_ETN:
				if seq in ETN_event:
					ETN_event[seq].append((ETN_starting_time[seq],t-1))
				else:
					ETN_event[seq] = [(ETN_starting_time[seq],t-1)]
				del ETN_starting_time[seq]
			for seq in new_ETN:
				ETN_starting_time[seq] = t
			#update the set of active ETN
			active_ETN = current_ETN
		#take care of the last timestamp
		for seq in active_ETN:
			if seq in ETN_event:
				ETN_event[seq].append((ETN_starting_time[seq],self.info['T']-1))
			else:
				ETN_event[seq] = [(ETN_starting_time[seq],self.info['T']-1)]
		return ETN_event

	#compute the train of events for ETN
	def ETN_event_train(self,depth):
		ETN_event = {}
		#active_ETN = set of ETN active at current time
		active_ETN = self.Get_current_ETN(0,depth)
		#ETN_starting_time[seq] = last starting time of the activation of the ETN of signature seq
		ETN_starting_time = {seq:0 for seq in active_ETN}
		for t in range(1,len(self.TN)-depth+1):
			#ETN active at t
			current_ETN = self.Get_current_ETN(t,depth)
			#ETN active at t and not at t-1
			new_ETN = current_ETN.difference(active_ETN)
			#ETN active at t-1 and not at t
			finished_ETN = active_ETN.difference(current_ETN)
			for seq in finished_ETN:
				if seq in ETN_event:
					ETN_event[seq].append((ETN_starting_time[seq],t-1))
				else:
					ETN_event[seq] = [(ETN_starting_time[seq],t-1)]
				del ETN_starting_time[seq]
			for seq in new_ETN:
				ETN_starting_time[seq] = t
			#update the set of active ETN
			active_ETN = current_ETN
		#take care of the last timestamp
		for seq in active_ETN:
			if seq in ETN_event:
				ETN_event[seq].append((ETN_starting_time[seq],self.info['T']-1))
			else:
				ETN_event[seq] = [(ETN_starting_time[seq],self.info['T']-1)]
		return ETN_event

	#compute the ETN vector for various aggregation levels and fixed depth
	#assume that the current aggregation level is 1
	def ETN_vector(self,agg_max,depth):
		giant_vec = {agg:{} for agg in range(1,agg_max+1)}
		ETN_agg1 = self.ETN_histo(depth)
		#retain only the 20 most frequent ETN
		most_freq = sorted(ETN_agg1.keys(),key=lambda seq:ETN_agg1[seq],reverse=True)[:20]
		giant_vec[1] = {seq:ETN_agg1[seq] for seq in most_freq}
		for agg in range(2,agg_max+1):
			self.Get_TN(agg)
			dic = self.ETN_histo(depth)
			#retain only the 20 most frequent ETN
			most_freq = sorted(dic.keys(),key=lambda seq:dic[seq],reverse=True)[:20]
			giant_vec[agg] = {seq:dic[seq] for seq in most_freq}
		return giant_vec,ETN_agg1
