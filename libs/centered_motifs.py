import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from Librairies.utils import increase_dic,cosine_similarity,truncate_histo

def ordinal(i):
	"""convert the integer `i` into its ordinal counterpart
	useful for figure labels
	"""
	if i == 0:
		return r"$1^{st}$"
	elif i == 1:
		return r"$2^{nd}$"
	elif i == 2:
		return r"$3^{rd}$"
	return str(i + 1) + r"$^{th}$"

#about profiles of centered objects and their satellites
#################################################################################
def All_NCTN_profiles(depth):
	profiles = []; profile = ['0']*depth
	for _ in range(1, 2**depth):
		ind = depth - 1
		while profile[ind] == '1':
			profile[ind] = '0'
			ind -= 1
		profile[ind] = '1'
		profiles.append(''.join(profile))
	return profiles

def All_ECTN_profiles(depth):
	satellites_prof = []; profile = ['0']*depth
	for _ in range(1,4**depth):
		ind = depth-1
		while profile[ind]=='3':
			profile[ind] = '0'
			ind -= 1
		profile[ind] = str(int(profile[ind])+1)
		satellites_prof.append(''.join(profile))
	return All_NCTN_profiles(depth), satellites_prof

def swap_conv_ECTN(prof):
	"""exchange the '1' and '2' inside the string ECTN prof ('conv' stand for convention)"""
	res = ''
	for letter in prof:
		if letter=='1':
			res += '2'
		elif letter=='2':
			res += '1'
		else:
			res += letter
	return res

#map representation
#################################################################################
def string_to_map_ECTN(depth, profiles, seq):
	"""convert the string representation of a ECTN into its map representation"""
	#nb_profile[prof] = nb of satellites with prof as activity profile in a ECTN
	nb_profile = {profile: 0 for profile in profiles}
	for i in range(1, len(seq)//depth):
		nb_profile[seq[i*depth: (i+1)*depth]] += 1
	return (seq[:depth], (nb_profile[profile] for profile in profiles))

def string_to_map_NCTN(depth,profiles,seq):
	"""convert the string representation of a NCTN into its map representation"""
	#nb_profile[prof] = nb of satellites with prof as activity profile in a NCTN
	nb_profile = {profile:0 for profile in profiles}
	for i in range(len(seq)//depth):
		nb_profile[seq[i*depth:(i+1)*depth]] += 1
	return np.asarray([nb_profile[profile] for profile in profiles],dtype=int)

#compute the sub-NCTN contained in ECTN
#################################################################################
def ECTN_to_NCTN_prof(seq):
	"""return the two NCTN profiles deduced from the ECTN satellite profile seq"""
	prof1 = ''; prof2 = ''
	for letter in seq:
		if letter=='1':
			prof1 += '1'; prof2 += '0'
		elif letter=='2':
			prof1 += '0'; prof2 += '1'
		elif letter=='3':
			prof1 += '1'; prof2 += '1'
		else:
			prof1 += '0'; prof2 += '0'
	return prof1,prof2

def get_sub_NCTN(seq,depth):
	"""return seq1,seq2 the string representations of the 2 NCTN included in the ECTN of string representation seq"""
	seq1 = [seq[:depth]]; seq2 = [seq[:depth]]
	for i in range(1,len(seq)//depth):
		prof1,prof2 = ECTN_to_NCTN_prof(seq[i*depth:(i+1)*depth])
		if prof1!='0'*depth:
			seq1.append(prof1)
		if prof2!='0'*depth:
			seq2.append(prof2)
	return ''.join(sorted(seq1)),''.join(sorted(seq2))

#sample centered motifs in a temporal network (list of graphs)
#############################################################################################################
def compute_NCTN_string(TN, t, v, depth):
	"""compute the binary string representration of the NCTN instance starting at time t of central node v and given depth"""
	#dic_s[u][tau] = '1' if u is a satellite of v at time t+tau and '0' else
	dic_s = {}
	for tau in range(depth):
		#if v is active (i.e. has at least one satellite)
		if v in TN[t+tau]:
			for u in TN[t+tau][v]:
				if u in dic_s:
					dic_s[u][tau] = '1'
				else:
					dic_s[u] = ['0']*depth
					dic_s[u][tau] = '1'
	return ''.join(sorted([''.join(val) for val in dic_s.values()]))

def compute_ECTN_string(TN,t,ind,depth):
	"""compute the quaternary string representation of the ECTN instance starting at time t of central edge ind and given depth"""
	i,j = ind; central_profile = ''
	#encode the activity profiles of the satellites: dic_s[u] = activity profile of satellite u
	dir_s = {}; rev_s = {}
	for tau in range(depth):
		if TN[t+tau].has_edge(i,j):
			central_profile += '1'
		else:
			central_profile += '0'
		ngh_i = set(); ngh_j = set()
		if i in TN[t+tau]:
			ngh_i = set(TN[t+tau].neighbors(i))
		if j in TN[t+tau]:
			ngh_j = set(TN[t+tau].neighbors(j))
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
def add_NCTN(TN,histo,central_nodes,t,depth):
	for v in central_nodes:
		increase_dic(histo, (compute_NCTN_string(TN,t,v,depth),))

def add_ECTN(TN,histo,central_edges,t,depth):
	for ind in central_edges:
		increase_dic(histo, (compute_ECTN_string(TN,t,ind,depth),))

def get_NCTN_to_weight(TN,depth,trunc=None):
	"""return histo[seq] = space-time weight of the NCTN of string representation seq and depth depth in the TN TN (list of graphs)"""
	# histo[seq] = nb of occurrences of ETN seq in the whole temporal network
	histo = {}
	for t in range(len(TN)-depth+1):
		central_nodes = set()
		for tau in range(depth):
			central_nodes = central_nodes.union(set(TN[t+tau].nodes()))
		add_NCTN(TN, histo, central_nodes, t, depth)
	if trunc is not None:
		return truncate_histo(histo, trunc)
	return histo

#return histo[seq] = space-time weight of the ECTN of string representation seq and depth depth in the TN TN (list of graphs)
def get_ECTN_to_weight(TN,depth,trunc=None):
	#histo[seq] = nb of occurrences of ECTN seq in the whole temporal network
	histo = {}; list_times = range(len(TN)-depth+1)
	#cumulate the interactions on depth time steps to keep track of the central edges
	dic_central_edges = {}
	for tau in range(depth):
		for edge in TN[tau].edges:
			if edge[0]<edge[1]:
				ind = edge
			else:
				ind = edge[::-1]
			if ind in dic_central_edges:
				dic_central_edges[ind] += 1
			else:
				dic_central_edges[ind] = 1
	add_ECTN(TN,histo,dic_central_edges.keys(),0,depth)
	for t in list_times[1:]:
		#remove the edges from time t-1
		for edge in TN[t-1].edges:
			if edge[0]<edge[1]:
				ind = edge
			else:
				ind = edge[::-1]
			dic_central_edges[ind] -= 1
		#add the edges from time t+depth-1
		for edge in TN[t+depth-1].edges:
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
		#update ECTN_to_weight (=histo)
		add_ECTN(TN,histo,dic_central_edges.keys(),t,depth)
	if trunc is not None:
		return truncate_histo(histo,trunc)
	return histo

#cosine similarities btw histo of motifs (as returned by get_NCTN_to_weight or get_ECTN_to_weight)
#################################################################################
#compute the NCTN or ECTN autosimilarity (depends on choice kwarg) of temp_net (instance of Temp_net.Temp_net) for motifs of depth depth vs the aggregation level
def compute_CTN_autosim(temp_net,agg_max=50,step=2,choice="NCTN",depth=3,trunc=None):
	temp_net.get_TN()
	nb = agg_max//step + 1
	X = [k*step+1 for k in range(nb)]
	Y = [1]
	if choice=="NCTN":
		get_CTN = lambda TN: get_NCTN_to_weight(TN,depth,trunc=trunc)
	elif choice=="ECTN":
		get_CTN = lambda TN: get_ECTN_to_weight(TN,depth,trunc=trunc)
	else:
		raise ValueError('the value of choice should be either NCTN or ECTN')
	ori_NCTN = get_CTN(temp_net.TN)
	for agg in X[1:]:
		print('aggregation level '+str(agg)+' begins')
		temp_net.get_TN(agg=agg,data_time=False)
		Y.append(cosine_similarity(ori_NCTN,get_CTN(temp_net.TN)))
	return X,Y

#compute the NCTN or ECTN similarity (depends on choice kwarg) btw temp_net and other_net (instances of Temp_net.Temp_net) for motifs of depth depth vs the aggregation level
def compute_CTN_sim(temp_net,other_net,agg_max=50,step=2,choice="NCTN",depth=3,trunc=None):
	nb = agg_max//step + 1
	X = [k*step+1 for k in range(nb)]
	Y = []
	if choice=="NCTN":
		get_CTN = lambda TN: get_NCTN_to_weight(TN,depth,trunc=trunc)
	elif choice=="ECTN":
		get_CTN = lambda TN: get_ECTN_to_weight(TN,depth,trunc=trunc)
	else:
		raise ValueError('the value of choice should be either NCTN or ECTN')
	for agg in X:
		print('aggregation level '+str(agg)+' begins')
		temp_net.get_TN(agg=agg,data_time=False)
		other_net.get_TN(agg=agg,data_time=False)
		Y.append(cosine_similarity(get_CTN(temp_net.TN),get_CTN(other_net.TN)))
	return X,Y

#display motifs (diagram representation)
#################################################################################
#convert the string representation of a ECTN into its diagram representation
def string_to_diag_ECTN(seq,depth):
	res = nx.Graph()
	nb_satellites = len(seq)//depth-1
	#insert the central edge in the graph
	for i in range(depth-1):
		res.add_edge(i,i+1)
	for i in range(depth,2*depth-1):
		res.add_edge(i,i+1)
	#indicates when the central edge is active
	seq_central = seq[:depth]
	for tau in range(depth):
		if seq_central[tau]=='1':
			res.add_edge(tau,tau+depth)
	#sequence of activity profiles for satellites
	seq_sat = seq[depth:]
	#id of the last labelled node
	id_last = 2*depth+1
	#lateral[i] = duplicata of node i ordered by order of apparition in time
	#we have to draw one edge btw two consecutive duplicata
	lateral = {i:[] for i in range(nb_satellites)}
	for i in range(nb_satellites):
		for j in range(i*depth,(i+1)*depth):
			if seq_sat[j]=='1':
				#draw one vertical edge with the first node of the central edge
				res.add_edge(id_last,j-i*depth)
				lateral[i].append(id_last)
				id_last += 1
			elif seq_sat[j]=='2':
				#draw one vertical edge with the second node of the central edge
				res.add_edge(id_last,j-(i-1)*depth)
				lateral[i].append(id_last)
				id_last += 1
			elif seq_sat[j]=='3':
				#draw one vertical edge with each node of the central edge
				res.add_edge(id_last,j-i*depth)
				res.add_edge(id_last,j-(i-1)*depth)
				lateral[i].append(id_last)
				id_last += 1
	#draw the lateral edges
	for val in lateral.values():
		for i in range(len(val)-1):
			res.add_edge(val[i],val[i+1])
	return res

#convert the string representation of a NCTN into its diagram representation
def string_to_diag_NCTN(seq,depth):
	res = nx.Graph(); nb = len(seq)
	nb_nodes = nb//depth
	#insert the central node in the graph
	for i in range(depth-1):
		res.add_edge(i,i+1)
	#id of the last labelled node
	id_last = depth+1
	#lateral[i] = duplicata of node i ordered by order of apparition in time
	#we have to draw one edge btw two consecutive duplicata
	lateral = {i:[] for i in range(nb_nodes)}
	for i in range(nb_nodes):
		for j in range(i*depth,(i+1)*depth):
			if seq[j]=='1':
				res.add_edge(id_last,j-i*depth) #draw a vertical edge
				lateral[i].append(id_last)
				id_last += 1
	#draw the lateral edges
	for val in lateral.values():
		for i in range(len(val)-1):
			res.add_edge(val[i],val[i+1])
	return res

#add the diagram representation of the NCTN of string seq to ax
def add_to_ax_ECTN(seq,depth,ax):
	res = string_to_diag_ECTN(seq,depth)
	#visualize the graph res
	color_map = []
	for node in res:
		if node<2*depth:
			color_map.append('red')
		else:
			color_map.append('green')
	#we want to pin the central edge along a horizontal time-ordered axis
	fixed_positions = {i:(i,0.5) for i in range(depth)}
	for i in range(depth):
		fixed_positions[i+depth] = (i,-0.5)
	fixed_nodes = fixed_positions.keys()
	pos = nx.spring_layout(res,pos=fixed_positions,fixed=fixed_nodes)
	nx.draw_networkx(res,pos=pos,arrows=True,arrowstyle='-',node_color=color_map,with_labels=False,ax=ax)

#add the diagram representation of the NCTN of string seq to ax
def add_to_ax_NCTN(seq,depth,ax):
	res = string_to_diag_NCTN(seq,depth)
	#visualize the graph res
	color_map = []
	for node in res:
		if node<depth:
			color_map.append('red')
		else:
			color_map.append('green')
	#we want to pin the central node along a horizontal time-ordered axis
	fixed_positions = {i:(i,0) for i in range(depth)}
	fixed_nodes = fixed_positions.keys()
	pos = nx.spring_layout(res,pos=fixed_positions,fixed=fixed_nodes)
	nx.draw_networkx(res,pos=pos,arrows=True,arrowstyle='-',node_color=color_map,with_labels=False,ax=ax)

#draw the diagram representation of the CTN of string seq
def draw_CTN(seq,depth,title,choice="NCTN"):
	_,ax = plt.subplots(constrained_layout=True)
	fontsize = 16
	ax.set_axis_off()
	ax.set_title(title,fontsize=fontsize)
	if choice=="NCTN":
		add_to_ax_NCTN(seq,depth,ax)
	elif choice=="ECTN":
		add_to_ax_ECTN(seq,depth,ax)
	else:
		raise ValueError('the value of choice should be either NCTN or ECTN')

def draw_10_CTN(CTN_to_weight, depth, starting_rank=0, normalize=True, choice="NCTN"):
	"""
	visualize the ten most frequent (starting from starting_rank) CTN from dic_ECTN in the histo CTN_to_weight
	if normalize, then display the CTN probabilities (if not display the nb of occurrences)
	"""
	def helper(nb,norm):
		freq = nb/norm
		x = math.floor(math.log10(nb/norm))
		return str(np.round(freq*10**(-x),3)) + 'e'+str(x)

	if normalize:
		norm = sum(CTN_to_weight.values())
		get_freq_or_nb = lambda nb: helper(nb, norm)
	else:
		get_freq_or_nb = lambda nb: str(nb)
	if choice=="NCTN":
		add_to_ax_CTN = lambda seq, ax: add_to_ax_NCTN(seq, depth, ax)
	elif choice=="ECTN":
		add_to_ax_CTN = lambda seq, ax: add_to_ax_ECTN(seq, depth, ax)
	else:
		raise ValueError('the value of choice should be either NCTN or ECTN')
	
	motifs = list(truncate_histo(CTN_to_weight, 10, starting_rank=starting_rank).keys())
	fig, ax = plt.subplots(2, 5, figsize=(12, 6), constrained_layout=True)
	fontsize = 16
	for k in range(2):
		for l in range(5):
			seq = motifs[5*k+l]
			addtitle = get_freq_or_nb(CTN_to_weight[seq])
			ax[k,l].set_axis_off()
			ax[k,l].set_title(ordinal(5*k+l+starting_rank)+'\nfreq: '+addtitle,fontsize=fontsize)
			add_to_ax_CTN(seq, ax[k,l])
	return fig, ax
