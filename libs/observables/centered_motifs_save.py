import numpy as np
import math
import networkx as nx
from libs.settings import Load_vector

def All_NCTN_profiles(depth):
	profiles = []; profile = ['0']*depth
	for _ in range(1,2**depth):
		ind = depth-1
		while profile[ind]=='1':
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
	return All_NCTN_profiles(depth),satellites_prof

#convert the string representation of a ECTN into its map representation
def String_to_map_ECTN(depth,profiles,seq):
	#nb_profile[prof] = nb of satellites with prof as activity profile in a ECTN
	nb_profile = {profile:0 for profile in profiles}
	for i in range(1,len(seq)//depth):
		nb_profile[seq[i*depth:(i+1)*depth]] += 1
	return (seq[:depth],(nb_profile[profile] for profile in profiles))

#convert the string representation of a NCTN into its map representation
def String_to_map_NCTN(depth,profiles,seq):
	#nb_profile[prof] = nb of satellites with prof as activity profile in a NCTN
	nb_profile = {profile:0 for profile in profiles}
	for i in range(len(seq)//depth):
		nb_profile[seq[i*depth:(i+1)*depth]] += 1
	return np.asarray([nb_profile[profile] for profile in profiles],dtype=int)

#return seq1,seq2 the two NCTN profiles included in the ECTN seq
def Sub_NCTN(seq,depth):
	seq1 = [seq[:depth]]; seq2 = [seq[:depth]]
	for i in range(1,len(seq)//depth):
		prof1 = ''; prof2 = ''
		for letter in seq[i*depth:(i+1)*depth]:
			if letter=='1':
				prof1 += '1'
				prof2 += '0'
			elif letter=='2':
				prof1 += '0'
				prof2 += '1'
			elif letter=='3':
				prof1 += '1'
				prof2 += '1'
			else:
				prof1 += '0'
				prof2 += '0'
		if prof1!='0'*depth:
			seq1.append(prof1)
		if prof2!='0'*depth:
			seq2.append(prof2)
	return ''.join(sorted(seq1)),''.join(sorted(seq2))

#return the cosine similarity btw two ETN histograms of only one aggregation level
def ETN_sim_single(obs1,obs2):
	norm1 = sum([val**2 for val in obs1.values()])
	norm2 = sum([val**2 for val in obs2.values()])
	s = 0
	for key in set(obs1.keys()).intersection(set(obs2.keys())):
		s += obs1[key]*obs2[key]
	return s/math.sqrt(norm1*norm2)

#compute the ETN autosimilarity of name for motifs of depth 3 vs the aggregation level
def Compute_ETN_autosim(name,b_max=50,step=2):
	if type(name)==tuple:
		savename = name[0]+str(name[1])+name[2]
	else:
		savename = name
	net = tp.Temp_net(tp.Load_TN(name))
	net.Init(1)
	nb = b_max//step + 1
	X = [k*step+1 for k in range(nb)]
	Y = [1]; ori_ETN = net.ETN_histo(3)
	for b in X[1:]:
		print(str(b)+' begins')
		net.Get_TN(b)
		Y.append(ETN_sim_single(ori_ETN,net.ETN_histo(3)))
	np.savetxt('codata/Compute_ETN_autosim_'+savename+'.txt',np.array([X,Y]))

def Ordinal(i):
	if i==0:
		return r"$1^{st}$"
	elif i==1:
		return r"$2^{nd}$"
	elif i==2:
		return r"$3^{rd}$"
	return str(i+1)+r"$^{th}$"

#convert the EdgeTN signature seq into an EdgeTN multigraph
#the central edge is represented by the nodes 0, 1,... to 2*depth-1
def EdgeSeq_to_net(seq,depth):
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

#convert the ETN signature seq into an ETN network
#the central node is represented by the nodes 0, 1,... to depth-1
def Seq_to_net(seq,depth):
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

#given a motif, return all the possible motifs resulting from its shift in time (either forward or backward)
#assuming this motif is the only one to occur
def Shifted(seq,depth):
	net = Seq_to_net(seq,depth)
	#motifs shifted forward
	forward = []
	#motifs shifted backward
	backward = []
	seq_ = list(seq)
	for t in range(depth):
		new_seq = []
		nb_nodes = len(seq_)//depth
		for i in range(nb_nodes):
			sign = seq[i*depth+1:(i+1)*depth] + ['0']
			if sign!=['0']*3:
				new_seq.append(sign)
		new_seq.sort()
		pass

#return all the motifs of depth-1 caused by the given motif
def SubETN(seq,depth):
	res = []
	pass
	return res

#!!!not optimized yet!!!
#returns 1 if the time conjugate of the ETN is one of its shifted motifs
def Conj_is_shifted(seq,depth):
	return int(Time_conjugate(seq,depth) in Shifted(seq,depth))

#take a motif in input and
#return 1 if it is symmetric under time-reversal
#and 0 else
def Is_time_sym(seq,depth):
	nb_nodes = len(seq)//depth
	new_seq = [seq[i*depth:(i+1)*depth][::-1] for i in range(nb_nodes)]
	new_seq.sort()
	new_seq = ''.join(new_seq)
	return new_seq==seq

#take a motif in input and return its conjugate under time reversal
def Time_conjugate(seq,depth):
	nb_nodes = len(seq)//depth
	new_seq = [seq[i*depth:(i+1)*depth][::-1] for i in range(nb_nodes)]
	new_seq.sort()
	new_seq = ''.join(new_seq)
	return new_seq

#take an ECTN under its string form and return the string obtained by exchange of the two central nodes
def Is_12_sym(seq,depth):
	new_seq = []
	for i in range(1,len(seq)//depth):
		prof = ''
		for letter in seq[i*depth:(i+1)*depth]:
			if letter=='1':
				prof += '2'
			elif letter=='2':
				prof += '1'
			else:
				prof += letter
		new_seq.append(prof)
	return seq[:depth]+''.join(sorted(new_seq))

#take an ECTN satellite profile and return its image under 1/2 symmetry
def Swap_12_in_profile(profile):
	prof = ''
	for letter in profile:
		if letter=='1':
			prof += '2'
		elif letter=='2':
			prof += '1'
		else:
			prof += letter
	return prof

#take an ECTN under its string form and return its image under time reversal (also in string form)
def TR_ECTNimage(seq,depth):
	central_profile = seq[:depth][::-1]
	list_seq = [seq[i*depth:(i+1)*depth][::-1] for i in range(1,len(seq)//depth)]
	#extract the NCTN strings (central profile excluded) of the two nodes
	seq1 = []; seq2 = []
	for profile in list_seq:
		sat1 = ''; sat2 = ''
		for letter in profile:
			if letter=='1':
				sat1 += '1'
				sat2 += '0'
			elif letter=='2':
				sat1 += '0'
				sat2 += '1'
			elif letter=='3':
				sat1 += '1'
				sat2 += '1'
			else:
				sat1 += letter
				sat2 += letter
		if sat1!='0'*depth:
			seq1.append(sat1)
		if sat2!='0'*depth:
			seq2.append(sat2)
	seq1 = ''.join(sorted(seq1)); seq2 = ''.join(sorted(seq2))
	#in the ECTN string, the '1' correspond to interactions with the smallest central node
	#so if seq1>seq2, we exchange the '1' and the '2' in the ECTN string
	if seq1>seq2:
		for k,profile in enumerate(list_seq):
			new_prof = ''
			for letter in profile:
				if letter=='1':
					new_prof += '2'
				elif letter=='2':
					new_prof += '1'
				else:
					new_prof += letter
			list_seq[k] = new_prof
	return central_profile+''.join(sorted(list_seq))

#return the nb of satellites contributing to at least one complete triangle
def Get_tri_nb_ECTN(seq,depth):
	tri = 0
	for i in range(len(seq)//depth-1):
		ok = False
		for letter1,letter2 in zip(seq[:depth],seq[(i+1)*depth:(i+2)*depth]):
			if letter1=='1' and letter2=='3':
				ok = True
		tri += int(ok)
	return tri

#return the two NCTN profiles deduced from the ECTN sat prof seq
def ECTN_to_NCTN_prof(seq):
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

#exchange the '1' and '2' inside the string prof
def Swap_conv_ECTN(prof):
	res = ''
	for letter in prof:
		if letter=='1':
			res += '2'
		elif letter=='2':
			res += '1'
		else:
			res += letter
	return res

#visualize the ten most frequent NCTN from dic_NCTN of depth
#and save the figure at savepath
def Draw_ten_freq_NCTN(dic_NCTN,depth,savepath):
	motifs = sorted(dic_NCTN.keys(),key=lambda seq:dic_NCTN[seq],reverse=True)[:10]
	fig,ax = plt.subplots(2,5,figsize=(12,6),constrained_layout=True)
	norm = sum(dic_NCTN.values())
	fontsize = 16
	for k in range(2):
		for l in range(5):
			seq = motifs[5*k+l]
			freq = dic_NCTN[seq]/norm
			order = math.floor(math.log10(freq))
			addtitle = str(np.round(freq*10**(-order),3))+'e'+str(order)
			ax[k,l].set_axis_off()
			ax[k,l].set_title(Ordinal(5*k+l)+'\nfreq: '+addtitle,fontsize=fontsize)
			Plot_NCTN(seq,depth,ax[k,l])
	plt.savefig(savepath)
	plt.close()

#visualize an edge centered motif
#starting from its signature (string containing some 0, 1, 2 or 3)
def Plot_ECTN(seq,depth,ax):
	res = EdgeSeq_to_net(seq,depth)
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

#visualize an egocentric motif
#starting from its signature (string containing some 0 and/or 1)
def Plot_NCTN(seq,depth,ax):
	res = Seq_to_net(seq,depth)
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

def Draw_single_ECTN(seq,depth,title):
	fig,ax = plt.subplots(constrained_layout=True)
	fontsize = 16
	ax.set_axis_off()
	ax.set_title(title,fontsize=fontsize)
	Plot_ECTN(seq,depth,ax)

#visualize the ten most frequent ECTN from dic_ECTN of depth
#and save the figure at savepath
def Draw_ten_freq_ECTN(dic_ECTN,depth,savepath,starting_rank=0,normalize=True):
	motifs = sorted(dic_ECTN.keys(),key=lambda seq:dic_ECTN[seq],reverse=True)[starting_rank:starting_rank+10]
	fig,ax = plt.subplots(2,5,figsize=(12,6),constrained_layout=True)
	if normalize:
		norm = sum(dic_ECTN.values())
	else:
		norm = 1
	fontsize = 16
	for k in range(2):
		for l in range(5):
			seq = motifs[5*k+l]
			freq = dic_ECTN[seq]/norm
			order = math.floor(math.log10(freq))
			addtitle = str(np.round(freq*10**(-order),3))+'e'+str(order)
			ax[k,l].set_axis_off()
			ax[k,l].set_title(Ordinal(5*k+l)+'\nfreq: '+addtitle,fontsize=fontsize)
			Plot_ECTN(seq,depth,ax[k,l])
	plt.savefig(savepath)
	plt.close()
