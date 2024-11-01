import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)
import Librairies.Temp_net as tp
from Librairies.settings import Setup_Plot,Cosim,Get_versions,Load_instance_param,Get_savename,Raw_to_binned,Load_TN_ADM,Savename_to_name
from Librairies.atn import ADM_class,Min_EW
import Librairies.ETN as etn_lib
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
'''
import numpy as np
import zipfile
import io
from math import ceil,floor,sqrt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable,get_cmap
from matplotlib.colors import Normalize
from scipy import interpolate
import networkx as nx
import community.community_louvain as louvain
import random as rd
from tsmoothie.smoother import *
from tsmoothie.utils_func import sim_randomwalk
from sklearn.metrics import adjusted_rand_score
#turn off warnings from polynomial fitting
import warnings
warnings.simplefilter('ignore', np.RankWarning)
print('begin')

#returns 'scalar'/'distr' if obs is a scalar/distribution observable
def Get_nature(obs):
	scalar_signature = {'ICC','nb','error','entropy','sim','avg','frac'}
	for seq in scalar_signature:
		if seq in obs:
			return 'scalar'
	return 'distr'

#res[i] = integer name of the folder corresponding to the (obs,arg) pair obs_with_arg[i]
#record[arg] = name of the folder corresponding to arg (one record per observable)
def Check_folder_flow(obs_with_arg):
	res = [0]*len(obs_with_arg)
	for i,(obs,arg) in enumerate(obs_with_arg):
		if not os.path.isdir('codata/'+obs):
			os.mkdir('codata/'+obs)
			res[i] = 0
			os.mkdir('codata/'+obs+'/0')
			np.savetxt('codata/'+obs+'/record.txt',[str(arg),'0'],fmt='%s')
		else:
			tab = np.loadtxt('codata/'+obs+'/record.txt',dtype=str)
			if tab.ndim==1:
				record = {tab[0]:tab[1]}
			else:
				record = {tab[0,i]:tab[1,i] for i in range(np.size(tab,1))}
			key = str(arg)
			if key in record:
				res[i] = int(record[key])
			else:
				res[i] = len(record)
				record[key] = str(res[i])
				os.mkdir('codata/'+obs+'/'+str(res[i]))
			np.savetxt('codata/'+obs+'/record.txt',np.array(list(zip(*record.items())),dtype=str),fmt='%s')
	return res

def Check_dir(obs):
	num = Find_folder(*obs)
	if not os.path.isdir('figures/'+obs[0]):
		os.mkdir('figures/'+obs[0])
	if not os.path.isdir('figures/'+obs[0]+'/'+str(num)):
		os.mkdir('figures/'+obs[0]+'/'+str(num))

#return X,np.log10(Y) after normalization of the histo (X,Y)
def Raw_to_exp(histo):
	X,Y = zip(*histo.items())
	Y = np.asarray(Y,dtype=float)
	#normalization
	norm = np.sum(Y)
	Y[:] /= norm
	#transfo
	return X,np.log10(Y)

#return the xlabel and ylabel associated to obs in Plot_distr_nb_flow
#as well as the function taking as input the raw histogram and returning the curve to be plotted
#in Plot_distr_nb_flow
def Obs_to_labels(name_obs):
	if 'duration' in name_obs:
		xlabel = r'$\log_{10}(\Delta t)$'; ylabel = r'$\log_{10}(P)$'
		func = Raw_to_binned
	elif 'inst_deg' in name_obs:
		xlabel = r'$k$'; ylabel = r'$\log_{10}(P)$'
		func = Raw_to_exp
	else:
		xlabel = r'$\log_{10}(n)$'; ylabel = r'$\log_{10}(P)$'
		func = Raw_to_binned
	return xlabel,ylabel,func

def Rewrite_listobs(list_obs):
	res = []
	for obs in list_obs:
		if type(obs)!=tuple:
			res.append((obs,()))
		elif type(obs[1])!=tuple:
			res.append((obs[0],tuple([obs[1]])))
		else:
			res.append(obs)
	return res

def Rewrite_obs(obs):
	if type(obs)!=tuple:
		return (obs,())
	elif type(obs[1])!=tuple:
		return (obs[0],tuple([obs[1]]))
	else:
		return obs

#collect all missing values
def Get_missing_obs(list_name,obs_with_arg,list_agg):
	list_obs = Rewrite_listobs(obs_with_arg)
	res = {name:{obs:[] for obs in obs_with_arg} for name in list_name}
	scalar_obs = []; distr_obs = []
	for i,obs in enumerate(list_obs):
		if Get_nature(obs[0])=='scalar':
			scalar_obs.append(i)
		else:
			distr_obs.append(i)
	for name in list_name:
		savename = Get_savename(name)
		for i in scalar_obs:
			obs = list_obs[i]; raw_obs = obs_with_arg[i]
			num = Find_folder(*obs)
			for agg in list_agg:
				try:
					data = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'.txt')
				except:
					res[name][raw_obs].append(agg)
		for i in distr_obs:
			obs = list_obs[i]; raw_obs = obs_with_arg[i]
			num = Find_folder(*obs)
			for agg in list_agg:
				try:
					data = np.loadtxt('codata/'+obs[0]+'/'+str(num)+'/Flow_'+savename+'_n'+str(agg)+'_b'+str(1)+'.txt')
				except:
					res[name][raw_obs].append(agg)
	#remove empty parts
	key_to_remove = set()
	for name,el in res.items():
		res[name] = {obs:val for obs,val in el.items() if val}
		if not res[name]:
			key_to_remove.add(name)
	for name in key_to_remove:
		del res[name]
	return res

#generate and save figures from obs realizations
def Plot_obs(obs_with_arg,list_name):
	standard_obs = Rewrite_listobs(obs_with_arg)
	scalar_obs = []; distr_obs = []
	for obs in standard_obs:
		if Get_nature(obs[0])=='scalar':
			scalar_obs.append(obs)
		else:
			distr_obs.append(obs)
		Check_dir(obs)
	for name in list_name:
		print(Get_savename(name)+' begins')
		print('\tscalar obs begins')
		for obs in scalar_obs:
			print('\t\t'+obs[0]+' begins')
			Plot_scalar_nb_flow(name,obs,list_b,list_agg)
		print('\tdistr obs begins')
		for obs in distr_obs:
			print('\t\t'+obs[0]+' begins')
			Plot_distr_nb_flow(name,obs,list_b,list_agg)

#do the analysis for the missing values
def Complete_missing_obs(list_name,list_obs,list_agg,list_b):
	missing_obs = Get_missing_obs(list_name,list_obs,list_agg)
	for name,val in missing_obs.items():
		list_agg = set()
		for obs,el in val.items():
			list_agg = list_agg.union(set(el))
		Internalobs_nb_flow(name,list(val.keys()),list_b,list(list_agg))

#visualize the ten most frequent NCTN
def Visu_NCTN(dic_ETN,depth,savefig=''):
	#sort the motifs by decreasing abundancy
	list_seq = sorted(dic_ETN.keys(),key=lambda seq:dic_ETN[seq],reverse=True)
	fig,ax = plt.subplots(2,5,constrained_layout=True,figsize=(12,6))
	fontsize = 16
	for i in range(2):
		for j in range(5):
			num = 5*i+j
			ax[i,j].set_axis_off()
			ax[i,j].set_title(etn_lib.Ordinal(num),fontsize=fontsize)
			etn_lib.Plot_ETN(list_seq[num],depth,ax[i,j])
	plt.savefig('test_GOL/visu_NCTN'+savefig+'.png')
	plt.close()

#return the list of XP data large enough for their flow to be computed
def Get_valid_XP():
	list_XP = []; valid_XP = []; removed_XP = []
	for i in range(16,20):
		list_XP.append('conf'+str(i))
	for i in range(1,4):
		list_XP.append('highschool'+str(i))
	for i in range(1,3):
		list_XP.append('work'+str(i))
	list_XP += ['utah','french','baboon','hospital','malawi']
	for name in list_XP:
		net = tp.Temp_net(tp.Load_TN(name))
		net.Get_data_time()
		if len(net.data_time)>500:
			valid_XP.append(name)
		else:
			removed_XP.append(name)
	return valid_XP,removed_XP

#load human evaluation of the sign sequences for different observables and datasets
#human_word[obs][name][n] = word vs b of obs for TN name
def Load_human():
	with open('data_NN/word_vs_b.txt','r') as file:
		list_paragraphs = []; human_word = {}; list_name = []; list_agg = []
		#read the whole file and break it into paragraphs, which are list of lines
		paragraph = []
		while True:
			line = file.readline()
			if not line:
				break
			if line=='\n':
				list_paragraphs.append(paragraph)
				paragraph = []
			else:
				#remove the '\n' at the end of line
				paragraph.append(line[:-1])
		if paragraph:
			list_paragraphs.append(paragraph)
		#build list_name from the first paragraph
		for line in list_paragraphs[0]:
			#a white space separates the identifier from the name
			ind = 0
			while line[ind]!=' ':
				ind += 1
			list_name.append(line[ind+1:])
		#second paragraph indicates the list of aggregation levels and observables investigated
		list_obs = []
		for line in list_paragraphs[1][1:]:
			name_obs,arg = line.split(' ')
			if '2' in arg:
				obs = (name_obs[2:-2],tuple([2]))
			elif '3' in arg:
				obs = (name_obs[2:-2],tuple([3]))
			else:
				obs = (name_obs[2:-2],())
			list_obs.append(obs)
		agg = ''
		for letter in list_paragraphs[1][0]:
			if letter==' ':
				list_agg.append(int(agg))
				agg = ''
			else:
				agg += letter
		if agg:
			list_agg.append(int(agg))
		#collect words
		for obs,paragraph in zip(list_obs,list_paragraphs[2:]):
			human_word[obs] = {}
			for name,line in zip(list_name,paragraph[1:]):
				list_word = line.split('|')
				human_word[obs][name] = {}
				for n,word in zip(list_agg,list_word):
					human_word[obs][name][n] = word
		return human_word,list_name,list_obs

#generate a test set for NN evaluation
#x_test and y_test are numpy arrays of type "float32"
#their nb of rows is equal to the number of samples in the test set
def Generate_test_set():
	letter_to_int = {'-':0,'0':1,'+':2}
	#load human prediction
	human_word,list_name,list_obs = Load_human()
	#build the x_test and the two y_test
	data_to_symb = Data_to_symb()
	data_to_symb.archive = zipfile.ZipFile('myarchive_average_all_XP.zip','r')
	nb_pts = len(list_obs)*len(list_name)*len(data_to_symb.tot_agg)
	x_test = np.zeros((nb_pts,30))#,dtype='float32')
	y_test_first_letter = np.zeros((nb_pts,3))
	y_test_word_size = np.zeros((nb_pts,5))
	ind = 0
	for obs in list_obs:
		Load_b_flow,list_real = data_to_symb.Get_load_b_flow(obs)
		for name in list_name:
			for agg in data_to_symb.tot_agg:
				#y_test
				word = human_word[obs][name][agg]
				y_test_word_size[ind,len(word)-1] = 1.0
				y_test_first_letter[ind,letter_to_int[word[0]]] = 1.0
				#x_test
				if list_real is not None:
					Y = np.zeros(data_to_symb.n)
					for i in list_real:
						Y += Load_b_flow(agg,name,'real'+str(i))
				else:
					Y = Load_b_flow(agg,name)
				m,M = np.min(Y),np.max(Y)
				if m<M:
					x_test[ind,:] = (Y-m)/(M-m)
				else:
					x_test[ind,:] = 0.5
				ind += 1
	#save the test set
	np.savetxt('data_NN/x_test.txt',x_test)
	np.savetxt('data_NN/y_test_first_letter.txt',y_test_first_letter)
	np.savetxt('data_NN/y_test_word_size.txt',y_test_word_size)

#modify the curve of f by pre-composing it by squeeze (x is modified but y is untouched)
#or post-composing
#squeeze is an increasing automorphism of [0,1]
#preserve the word of f
#alpha<1: left part of f-curve is contracted, right part is extended ; alpha>1: opposite
def squeeze1(x):
	for el in x:
		if el<0:
			print(el)
	return np.power(x,0.2+rd.random()*2.8)
#squeeze or extend the central part of the curve
#vec[0] must be in [0,1] and vec[1] in [1,np.inf]
def squeeze2(x):
	vec = [0.3+rd.random()*0.4,7+rd.random()*8]
	u = 1/(1+np.exp(-(x-vec[0])*vec[1]))
	u0 = 1/(1+np.exp(vec[0]*vec[1]))
	u1 = 1/(1+np.exp(-(1-vec[0])*vec[1]))
	return (u-u0)/(u1-u0)

def squeeze3(x):
	vec = [0.4+rd.random()*0.2,5+rd.random()*5]
	u = np.sinh((x-vec[0])*vec[1])
	u0 = -np.sinh(vec[0]*vec[1])
	u1 = np.sinh((1-vec[0])*vec[1])
	return (u-u0)/(u1-u0)


#get the ratios btw min and max height/slope
def Get_ratios(x,y):
	domains = []
	keep = True
	raw_sign = np.sign(y[1:]-y[:-1])
	dom_sign = raw_sign[0]; pos = 0
	for i,sign in enumerate(raw_sign):
		if sign!=dom_sign:
			domains.append((pos,i))
			pos = i; dom_sign = sign
	domains.append((pos,len(raw_sign)))
	min_height = abs(y[domains[0][-1]]-y[domains[0][0]]); max_height = min_height
	min_slope = min_height/x[domains[0][-1]]-x[domains[0][0]]; max_slope = min_slope
	for (i,j) in domains[1:]:
		height = abs(y[i]-y[j])
		if height<min_height:
			min_height = height
		elif height>max_height:
			max_height = height
		slope = height/(x[j]-x[i])
		if slope<min_slope:
			min_slope = slope
		elif slope>max_slope:
			max_slope = slope
	return max_height/min_height,max_slope/min_slope

#place extrema of a curve described by first_letter and size
#the minimum height of an extremum has to be not negligible wrt curve diameter
def Place_points(first_sign,size,height_ratio=5,slope_ratio=5):
	nb_pts = size + 1
	res = np.zeros((2,nb_pts))
	sign = first_sign #1(-1) if first_letter=='+'('-')
	res[0,1] = 1
	res[1,1] = sign
	#maximum and minimum gaps btw two consecutive extrema
	max_height = 1; min_height = 1
	#maximum and minimum slopes btw two consecutive extrema
	max_slope = 1; min_slope = 1
	for k in range(2,nb_pts):
		sign *= -1
		#decide of the height or y-gap
		a = max_height/height_ratio; b = min_height*height_ratio
		height = a + (b-a)*rd.random()
		max_height = max(height,max_height)
		min_height = min(height,min_height)
		res[1,k] = res[1,k-1] + sign*height
		#decide of the x-gap or slope
		a = max_slope/slope_ratio; b = min_slope*slope_ratio
		slope = a + (b-a)*rd.random()
		max_slope = max(slope,max_slope)
		min_slope = min(slope,min_slope)
		res[0,k] = res[0,k-1] + height/slope
	for k in range(2):
		m,M = np.min(res[k,:]),np.max(res[k,:])
		res[k,:] = (res[k,:]-m)/(M-m)
	return res

class Func:
	"""docstring for Func"""
	def __init__(self):
		self.param = 1
	def Refresh(self):
		self.param = 1 + 9*rd.random()
	def Get_val(self,x,typ):
		if typ=='log':
			return np.log(1+self.param*x)
		elif typ=='exp':
			return np.exp(x*self.param)
		elif typ=='expinv':
			return np.exp(-x*self.param)
		elif typ=='inv':
			return 1/(1+x*self.param)
		elif typ=='tanh':
			return np.tanh(x*self.param)
		elif typ=='raw':
			return x

#add intermediate points to obtain a smooth curve
def Dress_points(curve,size,nb_data):
	xs = np.linspace(0,1,nb_data)
	keep = True; func = Func(); max_nb = 20; nb = 0; max_tot = 100
	while keep:
		func.Refresh()
		typ = rd.choice(['raw','tanh','exp','inv','expinv','log'])
		if size<5:
			pol = np.polyfit(func.Get_val(curve[0,:],typ),curve[1,:],size)
			ys = np.polyval(pol,func.Get_val(xs,typ))
		if size==5 or nb>max_nb:
			f = interpolate.interp1d(func.Get_val(curve[0,:],typ),curve[1,:],kind='cubic')
			ys = f(func.Get_val(xs,typ))
		#check the height and slope ratios criterion is fulfilled
		height_ratio,slope_ratio = Get_ratios(xs,ys)
		if height_ratio<=7 and slope_ratio<=7:
			keep = False
		nb += 1
		if nb>max_tot:
			return None
	m,M = np.min(ys),np.max(ys)
	return (ys-m)/(M-m)

'''
slope_ratio = 5; height_ratio = 5
first_sign = 1; size = 5
for _ in range(5):
	tab = Place_points(first_sign,size,height_ratio=height_ratio,slope_ratio=slope_ratio)
	fig,ax = plt.subplots()
	ax.plot(tab[0,:],tab[1,:],'--')
	ax.plot(*Dress_points(tab,size))
plt.show()
x_train = np.loadtxt('data_NN/x_train.txt')
y_train_word_size = np.loadtxt('data_NN/y_train_word_size.txt')
y_train_first_letter = np.loadtxt('data_NN/y_train_first_letter.txt')
for ind in [30,60,99]:
	fig,ax = plt.subplots()
	ax.plot(x_train[ind,:])
	size = 0; val = y_train_word_size[ind,size]
	while val<0.5:
		size += 1; val = y_train_word_size[ind,size]
	k = 0; val = y_train_first_letter[ind,k]
	while val<0.5:
		k += 1; val = y_train_first_letter[ind,k]
	if k==0:
		first_letter = '-'
	elif k==1:
		first_letter = '0'
	else:
		first_letter = '+'
	ax.set_title('size: '+str(size+1)+'\nfirst_letter: '+first_letter)
plt.show()
'''

#generate and save a training set
#nb_pts = nb of generated samples
#nb_pts//5 samples per word size ; one half with first letter '+'
#except for word_size = 1 where we have one third with first letter '+' and one third with '0'
#the reference function is a sinus, then we transform it with functions squeezei and Noise
def Generate_train_set_old(nb_pts):
	x_train = np.zeros((nb_pts,30))
	y_train_first_letter = np.zeros((nb_pts,3))
	y_train_word_size = np.zeros((nb_pts,5))
	#first generate examples for words with no '0' in it
	for k in range(5):
		print(k)
		for l in range(2):
			print(l)
			for i in range(nb_pts//11):
				ind = k*2*nb_pts//11 + l*nb_pts//11 + i
				y_train_first_letter[ind,2*l] = 1.0
				y_train_word_size[ind,k] = 1.0
				#create the deformed curve
				keep = True
				while keep:
					curve = Dress_points(Place_points(2*l-1,k+1),k+1)
					if curve is not None:
						keep = False
				#add noise
				curve += 0.05*np.random.normal(size=30,loc=0,scale=1)
				#rescale the curve btw 0 and 1
				m,M = np.min(curve),np.max(curve)
				x_train[ind,:] = (curve-m)/(M-m)
	#second generate examples for word '0'
	for i in range(ind+1,nb_pts):
		y_train_first_letter[i,1] = 1.0
		y_train_word_size[i,0] = 1.0
		if rd.random()<0.9:
			curve = np.random.normal(size=30,loc=0,scale=1)
			m,M = np.min(curve),np.max(curve)
		else:
			curve = np.zeros(30) + 0.5
			m = 0; M = 1
		x_train[i,:] = (curve-m)/(M-m)
	#shuffle the training set to avoid bias in neural network training
	new_pos = list(range(nb_pts))
	rd.shuffle(new_pos)
	x_train = np.array([x_train[pos,:] for pos in new_pos])
	y_train_first_letter = np.array([y_train_first_letter[pos,:] for pos in new_pos])
	y_train_word_size = np.array([y_train_word_size[pos,:] for pos in new_pos])
	#save the training set
	np.savetxt('data_NN/x_train.txt',x_train)
	np.savetxt('data_NN/y_train_first_letter.txt',y_train_first_letter)
	np.savetxt('data_NN/y_train_word_size.txt',y_train_word_size)

def Generate_train_set_denoiser(nb_pts):
	x_train = np.zeros((nb_pts,30))
	#first generate examples for words with no '0' in it
	for k in range(5):
		print(k)
		for l in range(2):
			print(l)
			for i in range(nb_pts//11):
				ind = k*2*nb_pts//11 + l*nb_pts//11 + i
				#create the deformed curve
				keep = True
				while keep:
					curve = Dress_points(Place_points(2*l-1,k+1),k+1,rd.randint(11,30))
					if curve is not None:
						keep = False
				#rescale the curve btw 0 and 1 and complete the curve to have 30 data points
				nb_data = len(curve)
				curve = np.concatenate((curve,np.asarray([curve[-1]]*(30-nb_data))))
				m,M = np.min(curve),np.max(curve)
				x_train[ind,:] = (curve-m)/(M-m)
	#save the training set
	np.savetxt('data_NN/x_train_denoiser.txt',x_train)

#maximum word size = 3; noise type: Gaussian + salt and pepper
def Generate_train_set(nb_pts,max_size):
	x_train = np.zeros((nb_pts,30))
	y_train_first_letter = np.zeros((nb_pts,3))
	y_train_word_size = np.zeros((nb_pts,max_size))
	#first generate examples for words with no '0' in it
	for k in range(max_size):
		print(k)
		for l in range(2):
			print(l)
			for i in range(nb_pts//(2*max_size+1)):
				ind = k*2*nb_pts//(2*max_size+1) + l*nb_pts//(2*max_size+1) + i
				y_train_first_letter[ind,2*l] = 1.0
				y_train_word_size[ind,k] = 1.0
				#create the deformed curve
				keep = True
				while keep:
					curve = Dress_points(Place_points(2*l-1,k+1),k+1,rd.randint(11,30))
					if curve is not None:
						keep = False
				nb_data = len(curve)
				#add noise
				#Gaussian noise
				curve += 0.05*np.random.normal(size=nb_data,loc=0,scale=1)
				#salt and pepper noise
				m,M = np.min(curve),np.max(curve)
				if rd.random()<0.05:
					pos = rd.randint(0,nb_data-1)
					if rd.random()<0.5:
						curve[pos] = M
					else:
						curve[pos] = m
				#rescale the curve btw 0 and 1 and complete the curve to have 30 data points
				curve = np.concatenate((curve,np.asarray([curve[-1]]*(30-nb_data))))
				x_train[ind,:] = (curve-m)/(M-m)
	#second generate examples for word '0'
	for i in range(ind+1,nb_pts):
		y_train_first_letter[i,1] = 1.0
		y_train_word_size[i,0] = 1.0
		if rd.random()<0.9:
			#Gaussian noise
			nb_data = rd.randint(11,30)
			curve = np.random.normal(size=nb_data,loc=0,scale=1)
			#salt and pepper noise
			m,M = np.min(curve),np.max(curve)
			if rd.random()<0.05:
				pos = rd.randint(0,nb_data-1)
				if rd.random()<0.5:
					curve[pos] = M
				else:
					curve[pos] = m
			#complete the curve until we get 30 data points
			curve = np.concatenate((curve,np.asarray([curve[-1]]*(30-nb_data))))
		else:
			curve = np.zeros(30) + 0.5
			m = 0; M = 1
		x_train[i,:] = (curve-m)/(M-m)
	#shuffle the training set to avoid bias in neural network training
	new_pos = list(range(nb_pts))
	rd.shuffle(new_pos)
	x_train = np.array([x_train[pos,:] for pos in new_pos])
	y_train_first_letter = np.array([y_train_first_letter[pos,:] for pos in new_pos])
	y_train_word_size = np.array([y_train_word_size[pos,:] for pos in new_pos])
	#save the training set
	np.savetxt('data_NN/x_train.txt',x_train)
	np.savetxt('data_NN/y_train_first_letter.txt',y_train_first_letter)
	np.savetxt('data_NN/y_train_word_size.txt',y_train_word_size)

def Get_list_obs():
	list_obs = []
	for k in [2,3]:
		for prefix in ['N','E']:
			for suffix in ['nb_diff','nb_tot','motif_error','sim','sim_trunc']:
				list_obs.append((prefix+'CTN0'+suffix,(k,)))
	list_obs.append(('ICC',()))
	for scalar in ['avg','frac']:
		list_obs.append(('edge0time_weight0'+scalar,()))
	for obj in ['node','edge']:
		for prefix in ['','event_','inter']:
			for scalar in ['avg','frac']:
				list_obs.append((obj+'0'+prefix+'duration0'+scalar,()))
	return list_obs

#removed data set: 'brownD1'
def Get_list_name():
	list_name = []
	#list_XP
	for i in range(16,20):
		list_name.append('conf'+str(i))
	for i in range(1,4):
		list_name.append('highschool'+str(i))
	for i in range(1,3):
		list_name.append('work'+str(i))
	list_name += ['utah','french','baboon','hospital','malawi']
	#list_raha
	list_name += ['ABP2pi','ABPpi4','Vicsek2pi','Vicsekpi4','brownD01','brownD001']
	#list_model
	list_name += ['ADM9conf16','ADM18conf16','min_EW1','min_EW2','min_EW3','min_ADM1','min_ADM2']
	return list_name

#return the sizes (nb of nodes, nb of timesteps, nb of temporal edges) of all the data sets
#analyzed by the flow method
def Get_sizes_TN():
	list_savename = Get_list_name()
	res = []
	for savename in list_savename:
		print('measuring '+savename)
		#TN measurement data
		data = tp.Load_TN(Savename_to_name(savename))
		#nb of timesteps
		nb_time = data[-1,0] + 1
		#nb of temporal edges
		nb_edges = np.size(data,0)
		#nb of nodes
		nb_nodes = len(set(data[:,1]).union(set(data[:,2])))
		res.append([savename,nb_nodes,nb_time,nb_edges])
	#save sizes
	path = os.path.join(PROJECT_ROOT,'conferences/Netsci2023/TN_sizes.txt')
	np.savetxt(path,np.asarray(res,dtype=str),fmt='%s',delimiter='\t')

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

#return the list of all randomized TN (savename format)
def Get_tot_TN_randomized():
	return [savename+'_randomized1' for savename in Get_tot_TN_not_randomized()]

#compute the list of all TN actually present in archive_name.zip among the TN in tot_TN (savename format)
def Get_computed_TN(list_obs,archive_name,tot_TN):
	#check what models are not present in archiveee.zip
	list_agg = [1,2,3,4,5,10,20,30,40,50,100]
	b_min = 1; b_max = 300; step = 10; nb = (b_max-b_min)//step + 1
	list_b = [k*step+b_min for k in range(nb)]
	archive = zipfile.ZipFile(archive_name+'.zip','r')
	available_TN = set(tot_TN)
	for obs in list_obs:
		print(obs)
		surpath = ''
		if '0sim' in obs[0]:
			surpath = 'vsb'
		if '0avg' in obs[0]:
			distr_obs = (obs[0][:-4],obs[1])
			surpath = '_b1'
		elif '0frac' in obs[0]:
			distr_obs = (obs[0][:-5],obs[1])
			surpath = '_b1'
		else:
			distr_obs = obs
		tab = np.loadtxt(io.BytesIO(archive.read('codata/'+distr_obs[0]+'/record.txt')),dtype=str)
		if tab.ndim==1:
			record = {tab[0]:tab[1]}
		else:
			record = {tab[0,i]:tab[1,i] for i in range(np.size(tab,1))}
		folder_num = int(record[str(distr_obs[1])])
		dic_missing = {}
		for savename in tot_TN:
			dic_missing[savename] = []
			for agg in list_agg:
				path = 'codata/'+distr_obs[0]+'/'+str(folder_num)+'/Flow_'+savename+'_n'+str(agg)+surpath+'real'
				try:
					for i in range(10):
						np.loadtxt(io.BytesIO(archive.read(path+str(i)+'.txt')))
				except:
					dic_missing[savename].append(agg)
			if not dic_missing[savename]:
				del dic_missing[savename]
			else:
				dic_missing[savename].sort()
				if savename in available_TN:
					available_TN.remove(savename)
		for key,val in dic_missing.items():
			print('\t'+key,val)
	#list of available TN
	return list(available_TN)

#first neural network:
#a denoiser taking as input a curve of 30 points (eventually flat on the last 19 pts in case of n-flows)
#and returning its denoised original version. The denoising is successful if the raw sign sequence of the
#output equals the target sign sequence.
#Then what happens when applying again and again the denoiser?
#second neural network (if the first one is not enough):
#take the output of the first NN and return the target sign sequence
class NN_b_word:
	"""
	n_input = nb of points in the curve to analyze
	the NN has two outputs:
	 - the first letter of the b-word, coded on three neurons (+,-,0)
	 - the size of the b-word, coded on max_size neurons
	we assume that a word of size greater than 1 cannot contain the letter 0
	"""
	def __init__(self,model_name):
		#load the trained model if possible
		try:
			self.model = keras.models.load_model("data_NN/"+model_name)
		except:
			self.model = None
		#convert a one-hot encoded letter to its string writing
		self.ind_to_letter = {0:'-',1:'0',2:'+'}

	def Train_NN(self,n_input,max_size):
		#initialize the neural network
		inputs = keras.Input(shape=(n_input,))
		x = layers.Dense(60,activation="relu")(inputs)
		word_size = layers.Dense(max_size,activation="softmax",name="word_size")(x)
		first_letter = layers.Dense(3,activation="softmax",name="first_letter")(x)
		self.model = keras.Model(inputs=inputs,outputs=[word_size,first_letter],name="b_word")
		self.model.summary()
		#compile the model
		self.model.compile(
		optimizer=keras.optimizers.RMSprop(1e-3),
		loss=[
		keras.losses.CategoricalCrossentropy(from_logits=False),
		keras.losses.CategoricalCrossentropy(from_logits=False),
		],
		loss_weights=[1.0,1.0],
		metrics=[keras.metrics.CategoricalAccuracy()],
		)
		#load the training set
		x_train = np.loadtxt('data_NN/x_train.txt')
		y_train_word_size = np.loadtxt('data_NN/y_train_word_size.txt')
		y_train_first_letter = np.loadtxt('data_NN/y_train_first_letter.txt')
		#fit the model
		self.model.fit(
		x_train,
		{"word_size":y_train_word_size,"first_letter":y_train_first_letter},
		epochs=2,
		batch_size=1,
		validation_split=0.2,
		)
		#save the model
		self.model.save("data_NN/model")

	def Train_denoiser(self):
		#initialize the neural network
		inputs = keras.Input(shape=(30,))
		x = layers.Dense(20,activation="relu")(inputs)
		y = layers.Dense(10,activation="relu")(x)
		z = layers.Dense(20,activation="relu")(y)
		denoised = layers.Dense(30,activation="relu")(z)
		self.model = keras.Model(inputs=inputs,outputs=denoised)
		self.model.summary()
		#compile the model
		self.model.compile(
		optimizer='adam',
		loss=keras.losses.MeanSquaredError(),
		)
		#load the training set
		x_train = np.loadtxt('data_NN/x_train_denoiser.txt')
		noise_factor = 0.2
		x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape) 
		x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
		#fit the model
		self.model.fit(x_train_noisy,x_train,
		epochs=5,
		batch_size=1,
		shuffle=True,
		validation_split=0.2,
		)
		#save the model
		self.model.save("data_NN/denoiser")

	#evaluate the NN on human appreciation of empirical flows
	def Evaluate_NN(self):
		#load the test set
		x_test = np.loadtxt('data_NN/x_test.txt')
		y_test_word_size = np.loadtxt('data_NN/y_test_word_size.txt')
		y_test_first_letter = np.loadtxt('data_NN/y_test_first_letter.txt')
		#evaluate the model
		y_word_size,y_first_letter = self.model.predict(x_test,verbose=2)
		equal_word = np.sum(y_first_letter*y_test_first_letter>0.5,axis=1)
		equal_letter = np.sum(y_word_size*y_test_word_size>0.5,axis=1)
		test_accuracy = np.sum(equal_word*equal_letter>0.5)/len(x_test)
		print("Test accuracy:",test_accuracy*100,'%')

	#get sentence prediction
	#obs and name are in standard format
	def Predict_NN_vsb(self,obs,name):
		x_pred = np.zeros((len(self.tot_agg),len(self.list_b)))
		for ind,n in enumerate(self.tot_agg):
			Y = np.zeros(len(self.list_b))
			for i in range(10):
				Y += self.obs_to_load_vsb[obs](obs,n,'real'+str(i),name)
			#rescale Y
			m,M = np.min(Y),np.max(Y)
			if abs(M-m)<1e-10:
				Y[:] = 0.5
			else:
				Y = (Y-m)/(M-m)
			#add Y to the NN input
			x_pred[ind,:] = Y
		#compute NN predictions
		y_word_size,y_first_letter = self.model.predict(x_pred,verbose=2)
		#convert them into the b-sentence
		list_word = ['first_word']
		for ind in range(len(self.tot_agg)):
			#identify the first letter
			k = 0; M = y_first_letter[ind,k]
			for i in range(3):
				test = y_first_letter[ind,i]
				if M<test:
					M = test
					k = i
			first_letter = self.ind_to_letter[k]
			if first_letter=='0':
				word = '0'
			else:
				#identify the word size
				size = 0; M = y_word_size[ind,size]
				for i in range(5):
					test = y_word_size[ind,i]
					if M<test:
						M = test
						size = i
				size += 1
				#build the word
				word = ''
				for _ in range(size):
					word += self.ind_to_letter[k]
					k = 2-k
			if word!=list_word[-1]:
				list_word.append(word)
		sentence = ''
		for word in list_word[1:]:
			sentence += word+'|'
		return sentence[:-1]

	#get sentence prediction
	#obs and name are in standard format
	def Predict_NN_vsb_denoiser(self,obs,name,nb_iter):
		dic_pred = {}; dic_word = {}
		sign_to_digit = {0:'0',-1:'-',1:'+'}
		for n in self.tot_agg:
			Y = np.zeros(len(self.list_b))
			for i in range(10):
				Y += self.obs_to_load_vsb[obs](obs,n,'real'+str(i),name)
			#rescale Y
			m,M = np.min(Y),np.max(Y)
			if abs(M-m)<1e-10:
				dic_word[n] = '0'
			else:
				Y = (Y-m)/(M-m)
				#if Y is monotonous avoid using the NN
				raw_sign = np.sign(Y[1:]-Y[:-1])
				if (raw_sign<0).all():
					dic_word[n] = '-'
				elif (raw_sign>0).all():
					dic_word[n] = '+'
				else:
					#add Y to the NN input
					dic_pred[n] = Y
		#compute NN predictions if necessary
		if dic_pred:
			agg_pred,x_pred = zip(*dic_pred.items())
			x_pred = np.asarray(x_pred)
			#replace by a while loop on raw sign convergence
			for _ in range(nb_iter):
				x_pred = self.model.predict(x_pred,verbose=2)
			#convert NN predictions into words
			for agg,curve in zip(agg_pred,list(x_pred)):
				raw_sign = np.sign(curve[1:]-curve[:-1])
				string = ['2']
				for sign in raw_sign:
					letter = sign_to_digit[sign]
					if letter!='0' and letter!=string[-1]:
						string.append(letter)
				if len(string)==1:
					dic_word[agg] = '0'
				else:
					dic_word[agg] = ''.join(string[1:])
		#aggregate words into the b-sentence
		list_word = ['first_word']
		for agg in self.tot_agg:
			word = dic_word[agg]
			if word!=list_word[-1]:
				list_word.append(word)
		sentence = ''
		for word in list_word[1:]:
			sentence += word+'|'
		return sentence[:-1]

class NN_convertor:
	"""docstring for NN_convertor"""
	def __init__(self):
		self.model = None
		#convert a one-hot encoded letter to its string writing
		self.ind_to_letter = {0:'-',1:'0',2:'+'}
		self.n_input = 30
		self.max_size = 3

	def Train_NN(self):
		#initialize the neural network
		inputs = keras.Input(shape=(self.n_input,))
		x = layers.Dense(60,activation="relu")(inputs)
		y = layers.Dense(40,activation="relu")(x)
		z = layers.Dense(20,activation="relu")(y)
		word_size = layers.Dense(self.max_size,activation="softmax",name="word_size")(z)
		first_letter = layers.Dense(3,activation="softmax",name="first_letter")(z)
		self.model = keras.Model(inputs=inputs,outputs=[word_size,first_letter],name="b_word")
		self.model.summary()
		#compile the model
		self.model.compile(
		optimizer=keras.optimizers.RMSprop(1e-3),
		loss=[
		keras.losses.CategoricalCrossentropy(from_logits=False),
		keras.losses.CategoricalCrossentropy(from_logits=False),
		],
		loss_weights=[1.0,1.0],
		metrics=[keras.metrics.CategoricalAccuracy()],
		)
		#load the training set
		x_train = np.loadtxt('data_NN/x_train.txt')
		y_train_word_size = np.loadtxt('data_NN/y_train_word_size.txt')
		y_train_first_letter = np.loadtxt('data_NN/y_train_first_letter.txt')
		#fit the model
		self.model.fit(
		x_train,
		{"word_size":y_train_word_size,"first_letter":y_train_first_letter},
		epochs=1,
		batch_size=1,
		validation_split=0.2,
		shuffle=True,
		)
		#save the model
		self.model.save("data_NN/NN_convertor")

	def Prepare(self,nb_pts=30):
		#load the trained neural network
		self.model = keras.models.load_model("data_NN/NN_convertor")

	def Curve_to_word(self,x_pred):
		nb_data = np.size(x_pred,1)
		if nb_data<self.n_input:
			vec = x_pred[:,-1]
			x_pred = np.concatenate((x_pred,np.tile(vec,(self.n_input-nb_data,1)).T),axis=1)
		y_word_size,y_first_letter = self.model.predict(x_pred,verbose=2)
		list_word = []
		for ind in range(np.size(x_pred,0)):
			#identify the first letter
			k = 0; M = y_first_letter[ind,k]
			for i in range(3):
				test = y_first_letter[ind,i]
				if M<test:
					M = test
					k = i
			first_letter = self.ind_to_letter[k]
			if first_letter=='0':
				word = '0'
			else:
				#identify the word size
				size = 0; M = y_word_size[ind,size]
				for i in range(self.max_size):
					test = y_word_size[ind,i]
					if M<test:
						M = test
						size = i
				size += 1
				#build the word
				word = ''
				for _ in range(size):
					word += self.ind_to_letter[k]
					k = 2-k
			list_word.append(word)
		return list_word

#display nb points of the training set as well as their labels
def Display_train_set():
	ind_to_letter = {0:'-',1:'0',2:'+'}
	#load the training set
	x_train = np.loadtxt('data_NN/x_train.txt')
	y_train_word_size = np.loadtxt('data_NN/y_train_word_size.txt')
	y_train_first_letter = np.loadtxt('data_NN/y_train_first_letter.txt')
	#choose nb points from the set:
	#compute the true word for every point
	ind_to_word = {}
	for ind in range(np.size(x_train,0)):
		k = np.argmax(y_train_first_letter[ind,:])
		first_letter = ind_to_letter[k]
		if first_letter=='0':
			word = '0'
		else:
			size = 1 + np.argmax(y_train_word_size[ind,:])
			word = ''
			for _ in range(size):
				word += ind_to_letter[k]
				k = 2-k
		ind_to_word[ind] = word
	#group the points by words
	word_to_ind = {}
	for ind,word in ind_to_word.items():
		if word in word_to_ind:
			word_to_ind[word].append(ind)
		else:
			word_to_ind[word] = [ind]
	#choose one example from each word
	list_word = ['0','-','+','-+','+-','-+-','+-+']
	list_ind = [rd.choice(word_to_ind[word]) for word in list_word]
	#display the curves
	list_color = ['blue','orange','purple','green','red','black']; fontsize = 15
	fig,ax = plt.subplots(constrained_layout=True)
	ax.tick_params(axis='both',labelsize=fontsize)
	ax.plot(x_train[list_ind[0],:],'--',label='0')
	ax.legend(fontsize=fontsize)
	plt.savefig('data_NN/figures/0_size.png')
	plt.close()
	num = 1
	while num<len(list_word):
		fig,ax = plt.subplots(constrained_layout=True)
		ax.tick_params(axis='both',labelsize=fontsize)
		for k in range(2):
			ind = list_ind[num+k]; word = list_word[num+k]
			ax.plot(x_train[ind,:],'--',label=word,color=list_color[k])
		ax.legend(fontsize=fontsize)
		plt.savefig('data_NN/figures/'+str(len(word))+'_size.png')
		plt.close()
		num += 2

class Symb_conv:
	"""docstring for Symb_conv"""
	def __init__(self):
		self.archive = None
		b_min = 1; b_max = 300; step = 10; nb = (b_max-b_min)//step + 1
		self.list_b = [k*step+b_min for k in range(nb)]
		self.tot_agg = [1,2,3,4,5,10,20,30,40,50,100]
		#maps obs to their folder
		self.obs_to_folder = {}
		#maps obs to the function loading their realizations vs b
		self.obs_to_load_vsb = {}
		#maps obs to the function loading their realizations vs n
		self.obs_to_load_vsn = {}
		#method to convert curves into words
		self.method = None

	#res = integer name of the folder corresponding to the (obs,arg) pair
	#record[arg] = name of the folder corresponding to arg (one record per observable)
	def Find_folder_local(self,obs,arg):
		tab = np.loadtxt(io.BytesIO(self.archive.read('codata/'+obs+'/record.txt')),dtype=str)
		if tab.ndim==1:
			record = {tab[0]:tab[1]}
		else:
			record = {tab[0,i]:tab[1,i] for i in range(np.size(tab,1))}
		return record[str(arg)]

	def Load_scalar_new(self,obs,savename,agg,surname=''):
		path = 'codata/'+obs[0]+'/'+self.obs_to_folder[obs]+'/Flow_'+savename+'_n'+str(agg)+surname+'.txt'
		return np.loadtxt(io.BytesIO(self.archive.read(path)))

	def Load_scalar_vsn(self,obs,savename):
		res = np.zeros((len(self.list_b),len(self.tot_agg)))
		path = 'codata/'+obs[0]+'/'+self.obs_to_folder[obs]+'/Flow_'+savename+'_n'
		for j,agg in enumerate(self.tot_agg):
			new_path = path+str(agg)+'real'
			for ind in range(10):
				for i,val in enumerate(np.loadtxt(io.BytesIO(self.archive.read(new_path+str(ind)+'.txt')))):
					res[i,j] += val
		return res

	def Load_etn_sim(self,obs,savename,agg,type_sim='vsb',surname=''):
		path = 'codata/'+obs[0]+'/'+self.obs_to_folder[obs]+'/Flow_'+savename+'_n'+str(agg)+type_sim+surname+'.txt'
		return np.loadtxt(io.BytesIO(self.archive.read(path)))[1,:]

	def Load_etn_sim_vsn(self,obs,savename,type_sim='vsb'):
		res = np.zeros((len(self.list_b),len(self.tot_agg)))
		path = 'codata/'+obs[0]+'/'+self.obs_to_folder[obs]+'/Flow_'+savename+'_n'
		for j,agg in enumerate(self.tot_agg):
			new_path = path+str(agg)+type_sim+'real'
			for ind in range(10):
				for i,val in enumerate(np.loadtxt(io.BytesIO(self.archive.read(new_path+str(ind)+'.txt')))[1,:]):
					res[i,j] += val
		return res

	def Load_scalarized(self,num,obs,savename,agg,func,surname=''):
		res = np.zeros(len(self.list_b))
		distr_obs = obs[0][:-num]
		path = 'codata/'+distr_obs+'/'+self.obs_to_folder[(distr_obs,obs[1])]+'/Flow_'+savename+'_n'+str(agg)+'_b'
		for i,b in enumerate(self.list_b):
			tab = np.loadtxt(io.BytesIO(self.archive.read(path+str(b)+surname+'.txt')))
			if tab.ndim==1:
				if tab[0]<0.5:
					res[i] = -1
				else:
					res[i] = tab[0]
			else:
				res[i] = func(tab)
		return res

	def Load_scalarized_vsn(self,num,obs,savename,func):
		res = np.zeros((len(self.list_b),len(self.tot_agg)))
		distr_obs = obs[0][:-num]
		path = 'codata/'+distr_obs+'/'+self.obs_to_folder[(distr_obs,obs[1])]+'/Flow_'+savename+'_n'
		for i,b in enumerate(self.list_b):
			for j,agg in enumerate(self.tot_agg):
				new_path = path+str(agg)+'_b'+str(b)+'real'
				for ind in range(10):
					tab = np.loadtxt(io.BytesIO(self.archive.read(new_path+str(ind)+'.txt')))
					if tab.ndim==1:
						if tab[0]<0.5:
							res[i,j] += -1
						else:
							res[i,j] += tab[0]
					else:
						res[i,j] += func(tab)
		return res

	#maps every obs (standard format) in list_obs to the folder containing its realizations
	#and every obs to the function loading these realizations
	def Get_obs_folder_load(self,list_obs,archive_name='myarchive6'):
		self.archive = zipfile.ZipFile(archive_name+'.zip','r')
		for obs in list_obs:
			if '0avg' in obs[0]:
				distr_obs = (obs[0][:-4],obs[1])
				self.obs_to_folder[distr_obs] = self.Find_folder_local(*distr_obs)
				self.obs_to_load_vsb[obs] = lambda obss,n,surname,savename:self.Load_scalarized(4,obss,savename,n,lambda tab:np.sum(tab[0,:]*tab[1,:])/np.sum(tab[1,:]),surname=surname)
				self.obs_to_load_vsn[obs] = lambda obss,savename:self.Load_scalarized_vsn(4,obss,savename,lambda tab:np.sum(tab[0,:]*tab[1,:])/np.sum(tab[1,:]))
			elif '0frac' in obs[0]:
				distr_obs = (obs[0][:-5],obs[1])
				self.obs_to_folder[distr_obs] = self.Find_folder_local(*distr_obs)
				self.obs_to_load_vsb[obs] = lambda obss,n,surname,savename:self.Load_scalarized(5,obss,savename,n,lambda tab:np.sum(tab[0,:]**2*tab[1,:])/np.sum(tab[0,:]*tab[1,:]),surname=surname)
				self.obs_to_load_vsn[obs] = lambda obss,savename:self.Load_scalarized_vsn(5,obss,savename,lambda tab:np.sum(tab[0,:]**2*tab[1,:])/np.sum(tab[0,:]*tab[1,:]))
			else:
				self.obs_to_folder[obs] = self.Find_folder_local(*obs)
				if not '0sim' in obs[0]:
					self.obs_to_load_vsb[obs] = lambda obss,n,surname,savename:self.Load_scalar_new(obss,savename,n,surname=surname)
					self.obs_to_load_vsn[obs] = lambda obss,savename:self.Load_scalar_vsn(obss,savename)
				else:
					obs_vsb = (obs[0],(obs[1][0],'vsb'))
					obs_vsn = (obs[0],(obs[1][0],'vsn'))
					self.obs_to_folder[obs_vsb] = self.obs_to_folder[obs]
					self.obs_to_folder[obs_vsn] = self.obs_to_folder[obs]
					self.obs_to_load_vsb[obs_vsb] = lambda obss,n,surname,savename:self.Load_etn_sim(obss,savename,n,type_sim='vsb',surname=surname)
					self.obs_to_load_vsb[obs_vsn] = lambda obss,n,surname,savename:self.Load_etn_sim(obss,savename,n,type_sim='vsn',surname=surname)
					self.obs_to_load_vsn[obs_vsb] = lambda obss,savename:self.Load_etn_sim_vsn(obss,savename,type_sim='vsb')
					self.obs_to_load_vsn[obs_vsn] = lambda obss,savename:self.Load_etn_sim_vsn(obss,savename,type_sim='vsn')
	#sets the method to convert curves into words
	def Set_conv_method(self,method):
		if method=='hand':
			self.method = Hand_convertor()
		elif method=='denoiser':
			self.method = Denoiser()
		elif method=='NN':
			self.method = NN_convertor()

	#display the flows vs b for each value of n on the same figure for the specified data obs,savename
	def Display_flows_vsb(self,obs,savename,figpath):
		list_marker = ['s','^','<','>','*','v','1','2','3','4','P','p']
		list_color = ['blue','green','red','black','brown','purple','gray','orange','pink','cyan']
		#load the data and display the results
		fig,ax = Setup_Plot(r'$b$','',fontsize=14)
		for ind,agg in enumerate(self.tot_agg):
			marker = list_marker[ind%len(list_marker)]
			color = list_color[ind%len(list_color)]
			Y = np.zeros(len(self.list_b))
			#average over ten realizations of STS
			for i in range(10):
				Y += self.obs_to_load_vsb[obs](obs,agg,'real'+str(i),savename)
			#display the flow
			ax.plot(self.list_b,Y,marker,color=color,markersize=7,label=r'$n = $'+str(agg))
			ax.legend(fontsize=12,ncol=min(4,len(self.tot_agg)),bbox_to_anchor=(0.5,1.23),loc="upper center")
		plt.savefig(figpath+"Flow_vsb.png")
		plt.close()

	#display the flows vs n for each value of b on the same figure for the specified data obs,savename
	def Display_flows_vsn(self,obs,savename,figpath):
		#all_flows[i,:] = flow vs n for b = self.list_b[i]
		all_flows = self.obs_to_load_vsn[obs](obs,savename)
		fig,ax = Setup_Plot(r'$n$','',fontsize=14)
		ax.set_facecolor('gray'); cmap = get_cmap('Spectral')
		listb_norm = [(b-self.list_b[0])/(self.list_b[-1]-self.list_b[0]) for b in self.list_b]
		for i,b_norm in enumerate(listb_norm):
			ax.plot(self.tot_agg,all_flows[i,:],color=cmap(b_norm))
		im_ratio = fig.get_size_inches()[0]/fig.get_size_inches()[1]
		fig.colorbar(ScalarMappable(norm=Normalize(vmin=self.list_b[0],vmax=self.list_b[-1]),cmap=cmap),ax=ax,fraction=0.05*im_ratio)
		plt.savefig(figpath+"Flow_vsn.png")
		plt.close()

	#return predicted b-sentence for the couple (obs,name)
	def Predict_vsb(self,obs,name):
		x_pred = np.zeros((len(self.tot_agg),len(self.list_b)))
		for ind,n in enumerate(self.tot_agg):
			Y = np.zeros(len(self.list_b))
			for i in range(10):
				Y += self.obs_to_load_vsb[obs](obs,n,'real'+str(i),name)
			#rescale Y
			m,M = np.min(Y),np.max(Y)
			if abs(M-m)<1e-10:
				Y[:] = 0.5
			else:
				Y = (Y-m)/(M-m)
			#add Y to the input
			x_pred[ind,:] = Y
		#compute method output (list of words)
		list_word = self.method.Curve_to_word(x_pred)
		#aggregate words into the b-sentence
		sentence = list_word[0]
		for ind,word in enumerate(list_word[1:]):
			if word!=list_word[ind]:
				sentence += '|'+word
		return sentence

	#return predicted n-sentence for the couple (obs,name)
	def Predict_vsn(self,obs,name):
		x_pred = self.obs_to_load_vsn[obs](obs,name)
		#rescale each curve (line of x_pred)
		for ind in range(np.size(x_pred,0)):
			m,M = np.min(x_pred[ind,:]),np.max(x_pred[ind,:])
			if abs(M-m)<1e-10:
				x_pred[ind,:] = 0.5
			else:
				x_pred[ind,:] = (x_pred[ind,:]-m)/(M-m)
		#compute method output (list of words)
		list_word = self.method.Curve_to_word(x_pred)
		#aggregate words into the b-sentence
		sentence = list_word[0]
		for ind,word in enumerate(list_word[1:]):
			if word!=list_word[ind]:
				sentence += '|'+word
		return sentence	
	
	#return predicted b-sentences for all registered observables
	def Predict_all_vsb(self,list_name):
		self.method.Prepare(nb_pts=30)
		dic_sentence = {}
		for obs in self.obs_to_load_vsb.keys():
			print(obs,'\n')
			dic_sentence[obs] = {}
			for name in list_name:
				dic_sentence[obs][name] = self.Predict_vsb(obs,name)
				print('\t'+name,dic_sentence[obs][name])
		return dic_sentence

	#return predicted n-sentences for all registered observables
	def Predict_all_vsn(self,list_name):
		self.method.Prepare(nb_pts=30)
		dic_sentence = {}
		for obs in self.obs_to_load_vsn.keys():
			print(obs,'\n')
			dic_sentence[obs] = {}
			for name in list_name:
				dic_sentence[obs][name] = self.Predict_vsn(obs,name)
				print('\t'+name,dic_sentence[obs][name])
		return dic_sentence

	#saving format: paragraphs headed by observable name, then each line
	#contains the sentence for one given data set
	#the correspondance btw line nb and data set is given at the file head
	def Save_sentence(self,dic_sentence,name_file,type_sentence,path='data_NN/sentence/'):
		if type_sentence=='b':
			end_file = 'b_word'
			data_dic = self.obs_to_load_vsb
		elif type_sentence=='n':
			end_file = 'n_word'
			data_dic = self.obs_to_load_vsn
		if name_file:
			if name_file[0]!='_':
				end_file += '_'
		with open(path+end_file+name_file+'.txt','w') as file:
			list_name = list(dic_sentence[list(data_dic.keys())[0]].keys())
			#the TN will appear in this order as sentences
			for name in list_name:
				file.write(name+'\n')
			file.write('\n')
			for obs in data_dic.keys():
				file.write(str(obs)+'\n')
				for name in list_name:
					file.write(dic_sentence[obs][name]+'\n')
				file.write('\n')

class Hand_convertor:
	"""
	"""
	def __init__(self):
		self.Y_data = []
		self.list_std = []
		self.digit_to_letter = {0:'0',-1:'-',1:'+'}
		self.deg_max = 10
		self.pol_mat = {deg:[] for deg in range(self.deg_max+1)}
		self.tot_pol_mat = {}
		self.nb_pts = None
		self.X = None
		self.int_scalar_obs = []
		self.int_distr_obs = []
		self.ext_obs = []

	def Prepare(self,nb_pts=3):
		self.nb_pts = nb_pts
		self.X = np.linspace(0,1,self.nb_pts)
		#compute useful matrices for polynomial interpolation
		self.Get_dic_pol_mat()

	def Get_optistring_local(self,Y):
		sign_der = np.sign(Y[1:]-Y[:-1])
		if (sign_der<0).all():
			return '-'*len(sign_der)
		elif (sign_der>0).all():
			return '+'*len(sign_der)
		elif (sign_der==0).all():
			return '0'*len(sign_der)
		m,M = np.min(Y),np.max(Y)
		if (M-m)/M<0.005:
			return '0'*len(sign_der)
		if (M-m)<=max(abs(Y[1:]-Y[:-1]))*1.05:
			return '0'*len(sign_der)
		#normalise Y
		Y = (Y-m)/(M-m)
		der_tab = Y[1:]-Y[:-1]
		#interpolate the curve with a polynom defined on [0,1] using cross-validation
		pol_fit = self.Get_pol_fit_local(Y)
		#if the polynom is constant then the string is full of zeros
		if len(pol_fit)==1:
			return '0'*len(sign_der)
		#self.Y_data = np.copy(Y)
		string = self.Pol_to_string_local(pol_fit,Y[:3])
		new_string = list(string)
		#eliminate flat intermediate domains
		domains = self.Find_domains_local(string)
		for ind,(start,end,letter) in enumerate(domains[1:-1]):
			diff = abs(Y[end+1]-Y[start])
			if diff<0.05:
				#absorb by the domain most compatible with its letter
				if abs(Y[end+1]-Y[domains[ind][0]])>abs(Y[domains[ind+2][1]+1]-Y[start]):
					#left absorption (by the left domain)
					new_string[start:end+1] = domains[ind][2]*(end-start+1)
				else:
					#right absorption
					new_string[start:end+1] = domains[ind+2][2]*(end-start+1)
		return ''.join(new_string)

	#compute useful matrices for polynomial interpolation
	def Get_dic_pol_mat(self):
		for pos in range(self.nb_pts):
			XX = np.concatenate((self.X[:pos],self.X[pos+1:]))
			dic_mat = self.Get_newdic_mat(XX)
			for deg,val in dic_mat.items():
				self.pol_mat[deg].append(val)
		self.tot_pol_mat = self.Get_newdic_mat(self.X)

	def Find_domains_local(self,string):
		domains = []; letter = string[0]; start = 0; end = 0
		while end<len(string):
			if string[end]==letter:
				end += 1
			else:
				domains.append((start,end-1,letter))
				start = end
				letter = string[start]
		domains.append((start,end-1,string[start]))
		return domains

	#the algo has a bias due to flat parts of the curve: it still sees oscillations
	#we impose a limited vertical resolution: two points are seen at the same vertical level if the gap
	#btw them is below a given percentage of the maximum or average diameter of the data along the y axis
	def Pol_to_string_local(self,pol_fit,Y_start):
		YY = np.dot(self.tot_pol_mat[len(pol_fit)-1][0].T,pol_fit)
		m = np.min(YY); M = np.max(YY); der_tab = YY[1:]-YY[:-1]
		if (M-m)/M<0.005:
			return '0'*len(der_tab)
		if (M-m)<=max(abs(der_tab))*1.05:
			return '0'*len(der_tab)
		res = [self.digit_to_letter[el] for el in np.sign(der_tab)]
		first_letter = self.digit_to_letter[np.sign(Y_start[1]-Y_start[0])]
		if res[0]!=first_letter:
			#we do not allow the polynom to change sign btw the first two points if we have a data curve
			#strictly monotonous at the beginning
			first_sign = np.sign(Y_start[1:]-Y_start[:2])
			if (first_sign>0).all() or (first_sign<0).all():
				res[0] = first_letter
		#absorb the non-signficant zeros
		last_letter = res[0]; pos = 0; keep = True
		while keep:
			letter = res[pos]
			if letter=='0':
				res[pos] = last_letter
			else:
				last_letter = letter
			pos += 1
			if pos==len(res):
				keep = False
		#the sign cannot change at the last point
		if res[-1]!=res[-2]:
			res[-1] = res[-2]
		#check no domain is flat: consider the first domain is correct, then test whether the remaining
		#part of the curve can be considerered as flat, i.e. (M_remainder-m_remainder)<<(M_tot-m_tot)
		#where M,m are maximum and minimum of the polynomial fit (not data)
		domains = self.Find_domains_local(res)
		M_tot = M; m_tot = m
		keep = True; ind = 0
		while keep:
			start,end,letter = domains[ind]
			m_remainder = np.min(YY[end+1:]); M_remainder = np.max(YY[end+1:])
			if (M_remainder-m_remainder)/(M_tot-m_tot)<0.05:
				return ''.join(res[:end+1])+letter*(len(res)-end-1)
			else:
				ind += 1
				if ind==len(domains):
					keep = False
		return ''.join(res)

	def Get_newdic_mat(self,X):
		M = np.ones((len(X),self.deg_max+1))
		for j in range(self.deg_max-1,-1,-1):
			M[:,j] = M[:,j+1]*X
		return {deg:(M[:,self.deg_max-deg:].T,np.linalg.inv(np.dot(M[:,self.deg_max-deg:].T,M[:,self.deg_max-deg:]))) for deg in range(self.deg_max+1)}

	#use cross validation instead of MDL to determine the best degree for the fitting polynom
	def Get_pol_fit_local(self,Y):
		dic_errors = {deg:self.Cross_val_local(Y,deg) for deg in range(self.deg_max)}
		best_deg = min(dic_errors.keys(),key=lambda deg:dic_errors[deg])
		return np.dot(self.tot_pol_mat[best_deg][1],np.dot(self.tot_pol_mat[best_deg][0],Y))

	def Cross_val_local(self,Y,deg):
		errors = []
		for pos in range(self.nb_pts):
			YY = np.concatenate((Y[:pos],Y[pos+1:]))
			pol_fit = np.dot(self.pol_mat[deg][pos][1],np.dot(self.pol_mat[deg][pos][0],YY))
			errors.append((np.sum(pol_fit*self.tot_pol_mat[deg][0][:,pos])-Y[pos])**2)
		return np.mean(errors)

	def String_to_word(self,string):
		word = ['2']
		for letter in string:
			if letter!='0' and letter!=word[-1]:
				word.append(letter)
		if len(word)==1:
			return '0'
		else:
			return ''.join(word[1:])

	#convert a curve into a word
	def Get_str(self,Y):
		#first check for monotonicity
		sign_der = np.sign(Y[1:]-Y[:-1])
		if (sign_der<0).all():
			return '-'
		elif (sign_der>0).all():
			return '+'
		elif (sign_der==0).all():
			return '0'
		m,M = min(Y),max(Y)
		if (M-m)/M<0.005:
			return '0'
		if (M-m)<=max(abs(Y[1:]-Y[:-1]))*1.05:
			return '0'
		#add the mirror of Y to avoid overfitting at the last data point
		#Y = np.concatenate((Y,Y[::-1]))
		der_tab = Y[1:]-Y[:-1]
		#interpolate the curve with a polynom defined on [0,1] using cross-validation
		pol_fit = self.Get_pol_fit_local(Y)
		#if the polynom is constant then the string is full of zeros
		if len(pol_fit)==1:
			return '0'
		#determine the polynom shell resulting from confidence on the fitting curve
		smoother = PolynomialSmoother(degree=len(pol_fit)-1)
		smoother.smooth(Y)
		self.Y_data = np.copy(Y)
		#generate intervals
		low_ci,up_ci = smoother.get_intervals('confidence_interval',confidence=0.05)
		#we add Gaussian noise to the data with std equal to half the CI width
		#then we choose as description the most frequent one
		self.list_std = (up_ci.flatten()-low_ci.flatten())/2
		original = self.Pol_to_string_local(pol_fit,Y[:3])
		nb_test = 100; dic_sub = {original:1}
		#Y_pol = np.polyval(pol_fit,np.linspace(0,1,len(Y)))
		for _ in range(nb_test):
			string = self.Get_optistring_local(Y + np.random.default_rng().normal(loc=0.0,scale=self.list_std).flatten())
			if string in dic_sub:
				dic_sub[string] += 1
			else:
				dic_sub[string] = 1
		#display the abundancies of obtained strings
		sorted_key = sorted(dic_sub.keys(),key=lambda s:dic_sub[s],reverse=True)
		max_pos = 0; keep = True; max_nb = dic_sub[sorted_key[0]]
		while keep:
			new_nb = dic_sub[sorted_key[max_pos]]
			if new_nb!=max_nb:
				keep = False
			else:
				max_pos += 1
				if max_pos==len(dic_sub):
					keep = False
		string = min(sorted_key[:max_pos],key=lambda s:len(self.String_to_word(s)))
		#check the last domain: pick the simplest string among the second most frequents
		#if it coincides with string but on the last domain, then it is returned as description
		ind = max_pos; keep = ind<len(dic_sub)
		if not keep:
			return self.String_to_word(string)
		nb = dic_sub[sorted_key[ind]]
		while keep:
			new_nb = dic_sub[sorted_key[ind]]
			if new_nb==nb:
				ind += 1
			else:
				keep = False
			if ind==len(dic_sub):
				keep = False
		second_string = min(sorted_key[max_pos:ind],key=lambda s:len(self.String_to_word(s)))
		#compare string and second_string
		letter1 = self.String_to_word(string); letter2 = self.String_to_word(second_string)
		if letter2==letter1[:-1]:
			chosen_string = second_string
		else:
			chosen_string = string
		return self.String_to_word(chosen_string)

	def Curve_to_word(self,x_pred):
		return [self.Get_str(Y) for Y in list(x_pred)]

#convert a obs written in tuple format to a string
def Obs_to_nameobs(obs):
	x = ''
	for el in obs[1]:
		if type(el)==str:
			x += el
		else:
			x += str(el)
	return obs[0] + x

'''
Generate_train_set(20000,3)
print('training set generated')
mymodel = NN_convertor()
mymodel.Train_NN()
exit()
'''

'''
mymodel = NN_b_word('denoiser')
list_obs = Get_list_obs()
list_name = Get_list_name()
mymodel.Get_obs_folder_load(list_obs)
#save the sentences
mymodel.Save_sentence(dic_sentence,'b_words_denoiser')
exit()
list_nb_iter = [5,10,20,30,60]; avg_size = []
for nb_iter in list_nb_iter:
	print("nb_iter:",nb_iter)
	dic_sentence = mymodel.Predict_all_vsb(list_name,choice='denoiser',nb_iter=nb_iter)
	list_size = []
	for obs in dic_sentence.keys():
		for name in list_name:
			list_size += [len(word) for word in dic_sentence[obs][name].split('|')]
	avg_size.append(np.mean(list_size))
fig,ax = plt.subplots(constrained_layout=True)
ax.plot(list_nb_iter,avg_size,'.')
ax.set_xlabel('nb of denoising iterations')
ax.set_ylabel('average word size')
ax.set_title('procedure: raw sign of the denoiser output')
plt.show()
exit()
'''

#return the edit distance btw the strings s and t
def String_ED(s,t):
	n = len(s); p = len(t)
	#M[i,j] = edit distance btw s[:i] and t[:j]
	M = np.zeros((n+1,p+1),dtype=int)
	for i in range(1,n+1):
		M[i,0] = i
	for j in range(1,p+1):
		M[0,j] = j
	for i in range(1,n+1):
		for j in range(1,p+1):
			M[i,j] = min([
				M[i-1,j]+1, #deletion
				M[i,j-1]+1, #insertion
				M[i-1,j-1]+int(s[i-1]!=t[j-1]) #substitution
				])
	return M[n,p]

#compute the Jaccard index of similarity btw two communities, that are lists of sets of nodes
def Jaccard_sim(comm1,comm2):
	res = 0
	for _ in range(50):
		rd.shuffle(comm1)
		rd.shuffle(comm2)
		val = 0
		for set1,set2 in zip(comm1,comm2):
			val += len(set1.intersection(set2))/len(set1.union(set2))
		val /= max(len(comm1),len(comm2))
		if res<val:
			res = val
	return res

#return communities in G
def Custom_louvain(G,x):
	TN_to_comm = louvain.best_partition(G,resolution=x,randomize=True)
	comm_to_TN = {}
	for key,val in TN_to_comm.items():
		if val in comm_to_TN:
			comm_to_TN[val].add(key)
		else:
			comm_to_TN[val] = {key}
	return comm_to_TN

class Diagram:
	"""docstring for Diagram"""
	def __init__(self):
		self.list_obs = []
		self.list_name = []
		self.dic_sentence = {}
		self.list_color = ['blue','orange','purple','green','red','black']
		self.diag = {}
		self.figpath = 'figures//new_flows/'

	#add a sentence to self.dic_sentence for each TN and observable
	def Add_sentence(self,name_file):
		with open(name_file+'.txt','r') as file:
			list_paragraphs = []
			#read the whole file and break it into paragraphs, i.e. lists of lines
			paragraph = []
			while True:
				line = file.readline()
				if not line:
					break
				if line=='\n':
					list_paragraphs.append(paragraph)
					paragraph = []
				else:
					#remove the '\n' at the end of line
					paragraph.append(line[:-1])
			if paragraph:
				list_paragraphs.append(paragraph)
			#build list_name from the first paragraph
			self.list_name = [line for line in list_paragraphs[0]]
			#collect words and observables
			for paragraph in list_paragraphs[1:]:
				#add the observable to list_obs
				name_obs,arg = paragraph[0].split("', ")
				obs_0 = name_obs[2:]
				elem = arg[1:-2].split(',')
				list_arg = []
				for el in elem:
					if el in ['2','3']:
						list_arg.append(int(el))
					elif el:
						list_arg.append(el[2:-1])
				obs_1 = tuple(list_arg)
				obs = (obs_0,obs_1)
				#add the sentence to dic_sentence
				if obs in self.dic_sentence:
					for name,line in zip(self.list_name,paragraph[1:]):
						self.dic_sentence[obs][name] = (self.dic_sentence[obs][name],line)
				else:
					self.dic_sentence[obs] = {name:line for name,line in zip(self.list_name,paragraph[1:])}

	#load the list of observables, data sets and corresponding sentences vs n and b
	def Load_sentence(self,conv_method='NN',path='data_NN/sentence/',end_name=''):
		self.dic_sentence = {}
		for type_sentence in ['n','b']:
			self.Add_sentence(path+type_sentence+'_word_'+conv_method+end_name)
		self.list_obs = list(self.dic_sentence.keys())

	#compute the Edit Distance Matrix (EDM) for every pair of data sets and every observable
	#mat_dist[i,j,k] = ED btw D1(i) and D2(j) wrt obs(k)
	def Get_EDM(self):
		nb_name = len(self.list_name)
		nb_obs = len(self.list_obs)
		mat_dist = np.zeros((nb_name,nb_name,nb_obs))
		for i,name1 in enumerate(self.list_name):
			for j,name2 in enumerate(self.list_name[i+1:]):
				for k,obs in enumerate(self.list_obs):
					for ind in range(2):
						mat_dist[i,i+1+j,k] += String_ED(self.dic_sentence[obs][name1][ind],self.dic_sentence[obs][name2][ind])
		for k in range(nb_obs):
			for i in range(nb_name):
				for j in range(i+1,nb_name):
					mat_dist[j,i,k] = mat_dist[i,j,k]
		return mat_dist

	#build the diagram: a dictionary with obs and couples of sentences (vs n, vs b) as keys
	#and sets of data sets as values
	def Get_diag(self):
		self.diag = {}
		for obs in self.list_obs:
			self.diag[obs] = {}
			for name,label in self.dic_sentence[obs].items():
				if label in self.diag[obs]:
					self.diag[obs][label].add(name)
				else:
					self.diag[obs][label] = {name}

	#display the diagram corresponding to obs
	def Display_diag(self,obs):
		fig,ax = plt.subplots(1,1)
		ax.set_axis_off()
		height = len(self.diag[obs])
		ax.set_xlim(-1,1)
		y = 0; width = 1.4; gap = 0.4
		y_gap = 3*width
		ax.set_ylim(-y_gap*height,2)
		for label,val in self.diag[obs].items():
			ax.annotate(label,(-1,y),fontsize=14,ha='center')
			#we place the TN names into a rectangle of maximum width 2 names
			nb_name = len(val)
			depth = ceil(nb_name/2)
			k,l = 0,0 #position in the rectangle
			if nb_name==1:
				for name in val:
					ax.annotate(name,(1-gap,y),fontsize=12,ha='center')
			else:
				for name in val:
					ax.annotate(name,(1-(depth-1-l)*gap,y+(1-2*k)*width),fontsize=11,ha='center')
					k += 1
					if k==2:
						l += 1
						k = 0
			ax.plot([-1,1],[y+0.9-y_gap/2]*2,'--',color='blue')
			y -= y_gap
		namefig = obs[0]
		for el in obs[1]:
			if type(el)==str:
				namefig += '_'+el
			else:
				namefig += '_'+str(el)
		plt.savefig('figures/new_flows/Diag/'+namefig+'.png')
		plt.close()

	#given a 3 dimensional array of pairwise distances btw its keys,
	#save the similarity graph in .gexf format
	def Compute_metric_sim(self,mat_dist,end_name=''):
		nb_name = len(self.list_name)
		nb_obs = len(self.list_obs)
		'''
		#compute the upper bound for the edit distance: it is the size of the longest sentence
		upper_bound = {k:0 for k in range(nb_obs)}
		for k,obs in enumerate(self.list_obs):
			bound_ind = [0]*2
			for name in self.list_name:
				for ind in range(2):
					bound = len(self.dic_sentence[obs][name][ind])
					if bound_ind[ind]<bound:
						bound_ind[ind] = bound
			upper_bound[k] = sum(bound_ind)
		'''
		#convert the distance matrix into a similarity matrix
		sim_mat = np.zeros((nb_name,nb_name,nb_obs))
		for k in range(nb_obs):
			M = np.max(mat_dist[:,:,k]) #upper_bound[k]
			if M==0:
				M = 1
			sim_mat[:,:,k] = 1-mat_dist[:,:,k]/M
		#average over observables
		#avg_sim_mat = np.sum(sim_mat,axis=2)/nb_obs
		avg_sim_mat = np.ones((nb_name,nb_name))
		for k in range(nb_obs):
			avg_sim_mat[:,:] *= sim_mat[:,:,k]
		avg_sim_mat = avg_sim_mat**(1/nb_obs)
		#convert avg_sim_mat into a weighted network to handle with Gephi
		G = nx.Graph()
		for i in range(nb_name):
			for j in range(i+1,nb_name):
				weight = avg_sim_mat[i,j]
				if weight>0:
					G.add_edge(self.list_name[i],self.list_name[j],weight=weight)
		#format compatible with gephi
		nx.write_gexf(G,self.figpath+'EDM/metric_simnet'+end_name+'.gexf')
		print('metric sim graph saved at '+self.figpath+'EDM/metric_simnet'+end_name+'.gexf')

	#compute the repartition functions associated to each class in classes
	#classes is a dictionary, keys are integers (class number) and values are sets
	#rep_dic[num][obs][label] = nb of TN from class num with label for obs / nb of TN in class
	def Get_repartition(self,classes):
		rep_dic = {}
		for num,val in classes.items():
			rep_dic[num] = {}
			for obs in self.list_obs:
				rep_dic[num][obs] = {}
				for label,sets in self.diag[obs].items():
					rep_dic[num][obs][label] = len(val.intersection(sets))/len(val)
		return rep_dic

	#display the histogram of the repartition function of the total class (equal to all TN)
	def Display_rep_tot(self):
		fig,ax = plt.subplots(constrained_layout=True)
		fontsize = 14
		dic_nb = {}
		for obs,dic1 in self.diag.items():
			for label,sets in dic1.items():
				n = len(sets)
				if n in dic_nb:
					dic_nb[n] += 1
				else:
					dic_nb[n] = 1
		X,Y = zip(*dic_nb.items())
		Y = np.log10(Y)
		ax.plot(X,Y,'.')
		ax.set_xlabel('group size',fontsize=fontsize-2)
		ax.set_ylabel('number of\noccurrences (log)',fontsize=fontsize-2)
		#plt.savefig(self.figpath+'Classes/'+name_obs+'.png')
		plt.show()

	pass
	#for the total class, compute the K-matrix, then display its eigenvectors
	def K_matrix(self):
		pass

	#build canonic classes:
	# - 0: conferences
	# - 1: schools
	# - 2: pedestrian models
	# - 3: other data sets
	def Get_canonic_classes(self):
		classes = {k:set() for k in range(4)}
		num_to_name = {0:'conferences',1:'schools',2:'pedestrian models',3:'other data sets'}
		for name in self.list_name:
			if 'conf' in name:
				classes[0].add(name)
			elif 'school' in name or name=='utah' or name=='french':
				classes[1].add(name)
			elif 'ABP' in name or 'Vicsek' in name or 'brownD' in name:
				classes[2].add(name)
			else:
				classes[3].add(name)
		return classes,num_to_name

	#build classes by pairwise analysis:
	# - 0: highschools
	# - 1: conferences
	# - 2: models
	def Get_comp_classes(self):
		classes = {}
		classes[0] = {'min_EW2','min_EW3','ADM9conf16','ADM18conf16','min_ADM1','min_ADM2','conf16','french','utah'}
		classes[1] = {'highschool1','highschool2','highschool3','conf17','conf18','conf19','ABPpi4','malawi','baboon','hospital','work1','work2'}
		classes[2] = {'min_EW1','Vicsekpi4','Vicsek2pi','brownD01','brownD001','ABP2pi'}
		num_to_name = {0:'ADM_models',1:'empirical',2:'pedestrian_models'}
		num_to_color = {0:'orange',1:'purple',2:'green'}
		return classes,num_to_name,num_to_color

	#return a randomized version of Get_comp_classes
	def Randomized_comp_classes(self):
		shuffled_name = rd.sample(self.list_name,len(self.list_name))
		res = {}
		res[0] = set(shuffled_name[:9])
		res[1] = set(shuffled_name[9:21])
		res[2] = set(shuffled_name[21:])
		return res

	#return the diameter and heterogeneity averaged over random classes
	#return also the average overlap?
	def Randomized_diam_hetero(self):
		nb_avg = 200
		#initialize the diameter and heterogeneity histo
		dic_diam = {}; dic_hetero = {}
		for num in range(3):
			dic_diam[num] = {obs:0 for obs in self.list_obs}
			dic_hetero[num] = {obs:0 for obs in self.list_obs}
		for ind in range(nb_avg):
			print(nb_avg-ind)
			classes = self.Randomized_comp_classes()
			rep_dic = self.Get_repartition(classes)
			new_diam = self.Get_diameter(classes,rep_dic)
			new_hetero = self.Get_hetero(classes,rep_dic)
			for num in classes.keys():
				for obs in self.list_obs:
					dic_diam[num][obs] += new_diam[num][obs]
					dic_hetero[num][obs] += new_hetero[num][obs]
		for num in classes.keys():
			for obs in self.list_obs:
				dic_diam[num][obs] /= nb_avg
				dic_hetero[num][obs] /= nb_avg
		return dic_diam,dic_hetero

	#compute the diameter of each class in classes
	#dic_diam[num][obs] = diameter of class num for obs
	def Get_diameter(self,classes,rep_dic):
		dic_diam = {}
		for num,val in classes.items():
			dic_diam[num] = {}
			for obs in self.list_obs:
				res = 0
				for label1 in self.diag[obs].keys():
					for label2 in self.diag[obs].keys():
						res += String_ED(label1,label2)*rep_dic[num][obs][label1]*rep_dic[num][obs][label2]
				dic_diam[num][obs] = res
		return dic_diam

	#compute the heterogeneity of each class in classes
	#dic_hetero[num][obs] = heterogeneity of class num for obs
	def Get_hetero(self,classes,rep_dic):
		dic_hetero = {}
		for num in rep_dic.keys():
			dic_hetero[num] = {}
			for obs in self.list_obs:
				res = 0
				for proba in rep_dic[num][obs].values():
					if proba>0:
						res -= proba*np.log2(proba)
				dic_hetero[num][obs] = res
		return dic_hetero

	#display the histogram (over observables) of heterogeneity or diameter for each class
	def OldDisplay_histo(self,dic_obs,num_to_name,name_obs):
		nb_class = len(num_to_name)
		n = floor(sqrt(nb_class)); p = n
		eps = True
		while n*p<nb_class:
			if eps:
				p += 1
				eps = False
			else:
				n += 1
				eps = True
		fig,ax = plt.subplots(n,p,constrained_layout=True)
		ind = 0; fontsize = 14
		for (num,dic),axis in zip(dic_obs.items(),ax.flatten()):
			axis.hist(dic.values(),bins='auto',color=self.list_color[ind])
			axis.set_title(num_to_name[num],fontsize=fontsize)
			ind += 1
		if n==1:
			for l in range(p):
				ax[l].set_xlabel(name_obs,fontsize=fontsize-2)
			ax[0].set_ylabel('number of\noccurrences',fontsize=fontsize-2)
		else:
			for l in range(p):
				ax[n-1,l].set_xlabel(name_obs,fontsize=fontsize-2)
			for l in range(n):
				ax[l,0].set_ylabel('number of\noccurrences',fontsize=fontsize-2)
		plt.savefig(self.figpath+'Classes/'+name_obs+'.png')
		plt.show()
		plt.close()

	#display the histogram (over observables) of heterogeneity or diameter for each class
	def Display_histo(self,dic_obs,num_to_name,name_obs,num_to_color=None,x_extend=None):
		fontsize = 16
		if num_to_color is None:
			num_to_color = {key:'blue' for key in num_to_name.keys()}
		for num,dic in dic_obs.items():
			fig,ax = plt.subplots(1,1,constrained_layout=True)
			ax.hist(dic.values(),bins='auto',color=num_to_color[num])
			#ax.set_xlabel(name_obs,fontsize=fontsize)
			#ax.set_ylabel('number of\noccurrences',fontsize=fontsize)
			if x_extend is not None:
				ax.set_xlim(0,x_extend)
			ax.tick_params(axis='both',labelsize=fontsize)
			plt.savefig(os.path.join(PROJECT_ROOT,'conferences/Netsci2023/'+name_obs+'_'+num_to_name[num]+'.png'))
			plt.close()

	#compute the pairwise overlap in classes
	#dic_overlap[num1][num2][obs] = overlap btw classes num1 and num2 wrt obs
	def Get_overlap(self,classes,rep_dic):
		dic_overlap = {}
		list_num = list(classes.keys())
		for i,num1 in enumerate(list_num):
			dic_overlap[num1] = {}
			for num2 in list_num[i+1:]:
				dic_overlap[num1][num2] = {}
				for obs in self.list_obs:
					res = 0
					for label in self.diag[obs].keys():
						res += rep_dic[num1][obs][label]*rep_dic[num2][obs][label]
					dic_overlap[num1][num2][obs] = res
		return dic_overlap

	#compute a network of classes, with weight equal to the overlap across all observables
	#then save it in .gexf format to visualize it with Gephi
	def Display_overlap(self,dic_overlap,num_to_name):
		G = nx.Graph(); num_to_name = {k:'class '+str(k) for k in range(3)}
		for num1,dic1 in dic_overlap.items():
			for num2,dic2 in dic1.items():
				G.add_edge(num_to_name[num1],num_to_name[num2],weight=np.mean(list(dic2.values())))
		for edge in [('class 0','class 1'),('class 0','class 2'),('class 1','class 2')]:
			print(G[edge[0]][edge[1]]['weight'])
		nx.write_gexf(G,self.figpath+'Classes/overlap.gexf')
		print('overlap graph saved at '+self.figpath+'Classes/overlap.gexf')

	#determine the correct value for the resolution in the Louvain algorithm run on the network G
	def Fine_tune_Louvain(self,G):
		#add isolated nodes to G
		for name in self.list_name:
			if not name in G.nodes:
				G.add_node(name)
		resolution = np.linspace(0.4,1,40); nb_avg = 100
		rand_sim = np.zeros(len(resolution))
		TN_to_comm1 = {name:rd.randint(0,3) for name in self.list_name}
		for ind in range(nb_avg):
			print(nb_avg-ind)
			for num,x in enumerate(resolution):
				TN_to_comm = louvain.best_partition(G,resolution=x,randomize=True)
				comm1 = [TN_to_comm1[name] for name in self.list_name]
				comm2 = [TN_to_comm[name] for name in self.list_name]
				rand_sim[num] += adjusted_rand_score(comm1,comm2)
				TN_to_comm1 = {key:val for key,val in TN_to_comm.items()}
		rand_sim /= nb_avg
		fig,ax = plt.subplots(constrained_layout=True)
		ax.set_title('Rand score btw communities\nobtained at two consecutive resolution levels')
		ax.set_xlabel('resolution')
		ax.set_ylabel('Rand similarity')
		ax.plot(resolution[:-1],rand_sim[1:],'.')
		'''
		best_pos = 0; M = glob_mod[0]
		for pos,mod in enumerate(glob_mod):
			if M<mod:
				best_pos = pos
				M = mod
		best_res = resolution[best_pos]
		print('best resolution:',best_res)
		ax.plot([best_res]*2,[0,M],'--',color='red')
		'''

	#compute a graph of TN, with weight equal to the proba of sharing the same label over observables
	#then save it in .gexf format to visualize with Gephi
	def Compute_raw_sim(self,end_name=''):
		G = nx.Graph(); name_to_int = {name:i for i,name in enumerate(self.list_name)}
		for obs,val1 in self.diag.items():
			for label,val2 in val1.items():
				for name1 in val2:
					num1 = name_to_int[name1]
					for name2 in val2:
						num2 = name_to_int[name2]
						if num1<num2:
							if G.has_edge(name1,name2):
								G[name1][name2]['weight'] += 1
							else:
								G.add_edge(name1,name2,weight=1)
		nx.write_gexf(G,self.figpath+'Diag/raw_simnet'+end_name+'.gexf')
		print('raw sim graph saved at '+self.figpath+'Diag/raw_simnet'+end_name+'.gexf')

	#same as Compute_raw_sim but with a specified subset of observables
	def Partial_raw_sim(self,chosen_obs,name_to_int):
		G = nx.Graph(); missing_nodes = set(self.list_name)
		for obs in chosen_obs:
			for label,val2 in self.diag[obs].items():
				for name1 in val2:
					num1 = name_to_int[name1]
					for name2 in val2:
						num2 = name_to_int[name2]
						if num1<num2:
							missing_nodes.discard(name1)
							missing_nodes.discard(name2)
							if G.has_edge(name1,name2):
								G[name1][name2]['weight'] += 1
							else:
								G.add_edge(name1,name2,weight=1)
		#add the missing (isolated) nodes
		for node in missing_nodes:
			G.add_node(node)
		return G

	#compute the metric similarity graph btw TN for a specified subset of observables
	def Partial_metric_sim(self,chosen_obs,name_to_int):
		#compute the distance matrix
		nb_name = len(self.list_name)
		nb_obs = len(chosen_obs)
		mat_dist = np.zeros((nb_name,nb_name,nb_obs))
		for i,name1 in enumerate(self.list_name):
			for j,name2 in enumerate(self.list_name[i+1:]):
				for k,obs in enumerate(chosen_obs):
					for ind in range(2):
						mat_dist[i,i+1+j,k] += String_ED(self.dic_sentence[obs][name1][ind],self.dic_sentence[obs][name2][ind])
		#deduce the similarity matrix
		sim_mat = np.zeros((nb_name,nb_name,nb_obs))
		for k in range(nb_obs):
			M = np.max(mat_dist[:,:,k])
			if M==0:
				M = 1
			for i in range(nb_name-1):
				for j in range(i+1,nb_name):
					sim_mat[i,j,k] = 1-mat_dist[i,j,k]/M
		#geometric average over observables and compute the desired metric similarity graph
		G = nx.Graph(); missing_nodes = set(self.list_name)
		for i in range(nb_name):
			for j in range(i+1,nb_name):
				weight = 1
				for k in range(nb_obs):
					weight *= sim_mat[i,j,k]
				weight = weight**(1/nb_obs)
				if weight>0:
					G.add_edge(self.list_name[i],self.list_name[j],weight=weight)
					missing_nodes.discard(self.list_name[i])
					missing_nodes.discard(self.list_name[j])
		#add the missing (isolated) nodes
		for node in missing_nodes:
			G.add_node(node)
		return G

	#choose the resolution parameter using a stability criterion for detected communities
	def Choose_resolution(self,choice='raw',end_name=''):
		if choice=='raw':
			G_path = self.figpath+'Diag/raw_simnet'+end_name+'.gexf'
		elif choice=='metric':
			G_path = self.figpath+'EDM/metric_simnet'+end_name+'.gexf'
		self.Fine_tune_Louvain(nx.read_gexf(G_path))
		plt.savefig(os.path.join(PROJECT_ROOT,'conferences/Netsci2023/'+choice+'choice_resolution_'+end_name+'.png'))

	#load and display the sim matrix associated to the raw sim network
	def Display_simat(self,choice='raw',end_name=''):
		nb_name = len(self.list_name)
		sorted_sim = np.eye(nb_name)
		if choice=='raw':
			G_path = self.figpath+'Diag/raw_simnet'+end_name+'.gexf'
			rescale = len(self.list_obs)
		elif choice=='metric':
			G_path = self.figpath+'EDM/metric_simnet'+end_name+'.gexf'
			rescale = 1
		G = nx.read_gexf(G_path)
		#perform community detection based on modularity
		comm_to_TN = Custom_louvain(G,1)
		#order data sets
		list_comm = []
		for key in sorted(comm_to_TN.keys(),key=lambda el:len(comm_to_TN[el])):
			list_comm += comm_to_TN[key]
		missing_TN = [name for name in self.list_name if name not in list_comm]
		list_comm += missing_TN
		#we say that two TN belong to the same class iif the proba of sharing the same label is >=0.5
		for i in range(nb_name):
			for j in range(i+1,nb_name):
				if G.has_edge(list_comm[i],list_comm[j]):
					sorted_sim[i,j] = G[list_comm[i]][list_comm[j]]['weight']/rescale
					sorted_sim[j,i] = sorted_sim[i,j]

		fig,ax = plt.subplots(constrained_layout=True,figsize=(7,7))
		fontsize = 12
		img = ax.imshow(sorted_sim,cmap='gnuplot2')
		im_ratio = sorted_sim.shape[0]/sorted_sim.shape[1]
		fig.colorbar(img,ax=ax,fraction=0.05*im_ratio)
		#tick_labels = list_comm
		tick_loc = np.asarray(list(range(nb_name)))
		ax.set_xticks(tick_loc)
		ax.set_xticklabels(list_comm,rotation=90,fontsize=fontsize)
		ax.set_yticks(tick_loc)
		ax.set_yticklabels(list_comm,fontsize=fontsize)
		plt.savefig(os.path.join(PROJECT_ROOT,'conferences/Netsci2023/'+choice+'_simat'+end_name+'.png'))
		#plt.show()

	#compute communities of TN for the chosen similarity measure for a increasing number of observables
	#drawn at random ; then plot the Rand score similarity vs this number (average over nb_avg realizations)
	def Comm_vs_obs(self,choice='raw'):
		if choice=='raw':
			func = self.Partial_raw_sim
		elif choice=='metric':
			func = self.Partial_metric_sim
		nb_avg = 100; list_nb = np.arange(1,len(self.list_obs)+1,1)
		rand_sim = np.zeros(len(list_nb)); TN_to_comm1 = {name:rd.randint(0,3) for name in self.list_name}
		name_to_int = {name:i for i,name in enumerate(self.list_name)}
		for ind in range(nb_avg):
			print(nb_avg-ind)
			for i,nb_obs in enumerate(list_nb):
				#draw observables at random
				chosen_obs = rd.sample(self.list_obs,nb_obs)
				#compute the proximity network then communities
				TN_to_comm = louvain.best_partition(func(chosen_obs,name_to_int),resolution=1,randomize=True)
				comm1 = [TN_to_comm1[name] for name in self.list_name]
				comm2 = [TN_to_comm[name] for name in self.list_name]
				rand_sim[i] += adjusted_rand_score(comm1,comm2)
				TN_to_comm1 = {key:val for key,val in TN_to_comm.items()}
		rand_sim /= nb_avg
		#plot results
		fontsize = 16
		fig,ax = plt.subplots(constrained_layout=True)
		ax.set_xlabel('number of observables',fontsize=fontsize)
		ax.set_ylabel('Jaccard similarity',fontsize=fontsize)
		ax.plot([1,43],[1]*2,'--',color='red')
		ax.tick_params(axis='both',labelsize=fontsize)
		ax.plot(list_nb[:-1],rand_sim[1:],'.')
		plt.savefig(os.path.join(PROJECT_ROOT,'conferences/Netsci2023/'+choice+'_vsobs.png'))
		#plt.show()

def Flows_to_sentences(archive_name,list_obs=None,list_name=None):
	if list_obs is None:
		list_obs = Get_list_obs()
	if list_name is None:
		list_name = Get_list_name()
	cla = Symb_conv()
	cla.Get_obs_folder_load(list_obs,archive_name=archive_name)
	cla.Set_conv_method('NN')
	dic_sentence = cla.Predict_all_vsb(list_name)
	cla.Save_sentence(dic_sentence,'NN'+archive_name,'b')
	dic_sentence = cla.Predict_all_vsn(list_name)
	cla.Save_sentence(dic_sentence,'NN'+archive_name,'n')

#if no data has been computed yet, run first
#Get_fig_all(example_diag=True,basics=True,classes=False)
#then analyze the sim graphs with gephi, deduce the detected and random classes, modify the code accordingly
#then run
#Get_fig_all(example_diag=False,basics=False,classes=True)
def Get_fig_all(example_diag=True,basics=True,classes=False):
	diag = Diagram()
	diag.Load_sentence(conv_method='NN',end_name='arch6')
	diag.Get_diag()
	if basics:
		#compute and save the raw sim graph btw TN
		diag.Compute_raw_sim()
		#compute and save the metric sim graph btw TN
		mat_dist = diag.Get_EDM()
		diag.Compute_metric_sim(mat_dist)
		#display the sim matrices btw TN and choose the resolution parameter
		for choice in ['raw','metric']:
			diag.Choose_resolution(choice=choice)
			diag.Display_simat(choice=choice)
		#see how the detected classes of TN depend on the number of observables we consider
		for choice in ['raw','metric']:
			diag.Comm_vs_obs(choice=choice)
	#print a particular diagram as an example
	if example_diag:
		obs = ('ECTN0sim_trunc', (2, 'vsn'))
		for label,val in diag.diag[obs].items():
			print(label)
			for name in val:
				print('\t'+name)
	if classes:
		#compute the histograms of diameter and heterogeneity for random classes
		dic_diam,dic_hetero = diag.Randomized_diam_hetero()
		#display the histograms
		num_to_name = {k:'class'+str(k) for k in range(3)}
		for dic_obs,name_obs,x_extend in zip([dic_diam,dic_hetero],['diameter','heterogeneity'],[2,4]):
			diag.Display_histo(dic_obs,num_to_name,name_obs,x_extend=x_extend)
		#compute the same histograms for the detected classes
		classes,num_to_name,num_to_color = diag.Get_comp_classes()
		rep_dic = diag.Get_repartition(classes)
		dic_diam = diag.Get_diameter(classes,rep_dic)
		dic_hetero = diag.Get_hetero(classes,rep_dic)
		#display the histograms
		for dic_obs,name_obs,x_extend in zip([dic_diam,dic_hetero],['diameter','heterogeneity'],[2,4]):
			diag.Display_histo(dic_obs,num_to_name,name_obs,num_to_color=num_to_color,x_extend=x_extend)
		#compute the overlap similarity graph btw detected classes
		dic_overlap = diag.Get_overlap(classes,rep_dic)
		diag.Display_overlap(dic_overlap,num_to_name)

#return the average size of sentences in a given class
def Get_label_size():
	diag = Diagram()
	diag.Load_sentence(conv_method='NN',end_name='arch6')
	classes,num_to_name,num_to_color = diag.Get_comp_classes()
	for key,val in classes.items():
		print(key)
		res = 0
		for obs in diag.list_obs:
			for name in val:
				label = diag.dic_sentence[obs][name]
				for sentence in label:
					res += len(sentence)
		print('avg sentence size: '+str(res/(2*len(val)*len(diag.list_obs))))

def Load_sorted_sim_labels(choice,end_name):
	figpath = 'figures/rando/'
	list_comm = np.loadtxt(figpath+choice+"_listcomm"+end_name+".txt",dtype=str)
	new_labels = []
	for label in list_comm:
		if "_randomized1" in label:
			new_labels.append("rd1")
		elif "_randomized2" in label:
			new_labels.append("rd2")
		else:
			if "ADM" in label or "EW" in label or "ABP" in label or "brown" in label or "Vicsek" in label:
				new_labels.append("model")
			else:
				new_labels.append("xp")
	return new_labels,np.loadtxt(figpath+choice+"_simat"+end_name+".txt")

def Load_xp_rdlabels(choice,end_name):
	figpath = 'figures/rando/'
	list_comm = np.loadtxt(figpath+choice+"_listcomm"+end_name+".txt",dtype=str)
	label_to_ind = {}
	for i,label in enumerate(list_comm):
		if "ADM" in label or "EW" in label or "ABP" in label or "brown" in label or "Vicsek" in label or "highschool1" in label or "highschool3" in label or "work1" in label:
			pass
		else:
			label_to_ind[label] = i
	xp_labels = []; rd1_labels = []; rd2_labels = []
	for label,i in label_to_ind.items():
		if not "randomized" in label:
			xp_labels.append((label,i))
	for label in xp_labels:
		lab = label[0]+"_randomized1"
		rd1_labels.append((lab,label_to_ind[lab]))
		lab = label[0]+"_randomized2"
		rd2_labels.append((lab,label_to_ind[lab]))
	sorted_sim = np.loadtxt(figpath+choice+"_simat"+end_name+".txt")
	labels = xp_labels+rd1_labels+rd2_labels
	nb = len(labels)
	rd_sim = np.zeros((nb,nb))
	for k in range(nb):
		i = labels[k][1]
		for l in range(nb):
			j = labels[l][1]
			rd_sim[k,l] = sorted_sim[i,j]
	nb = len(xp_labels)
	return ["xp"]*nb+["rd1"]*nb+["rd2"]*nb,rd_sim

def Plot_simat(tick_labels,new_sim,savefig):
	nb = len(tick_labels)
	for k in range(nb):
		new_sim[k,k] = 0
	#plot the simat
	fig,ax = plt.subplots(constrained_layout=True,figsize=(6,6))
	fontsize = 14
	img = ax.imshow(new_sim,cmap='gnuplot2')
	im_ratio = new_sim.shape[0]/new_sim.shape[1]
	clb = fig.colorbar(img,ax=ax,fraction=0.05*im_ratio)
	clb.ax.tick_params(labelsize=fontsize)
	tick_loc = np.asarray(list(range(nb)))
	ax.set_xticks(tick_loc)
	ax.set_xticklabels(tick_labels,rotation=90,fontsize=fontsize)
	ax.set_yticks(tick_loc)
	ax.set_yticklabels(tick_labels,fontsize=fontsize)
	plt.savefig(os.path.join(PROJECT_ROOT,'conferences/Netsci2023/'+savefig+'.png'))
	plt.show()

#restrict the similarity matrix sorted_sim to randomized TN only and separate the different randomizations
def Simat_only_rd(new_labels,sorted_sim,savefig):
	#extract the randomized TN
	rd_labels = [(label,i) for i,label in enumerate(new_labels) if 'rd' in label]; nb = len(rd_labels)
	rd_sim = np.eye(nb)
	for k in range(nb):
		i = rd_labels[k][1]
		for l in range(nb):
			j = rd_labels[l][1]
			rd_sim[k,l] = sorted_sim[i,j]
	#permute lines of rd_sim so that the different randomizations are separated
	new_sim = np.eye(nb)
	rd1_labels = []; rd2_labels = []
	for i,el in enumerate(rd_labels):
		if el[0]=="rd1":
			rd1_labels.append((el[0],i))
		else:
			rd2_labels.append((el[0],i))
	new_rd_labels = rd1_labels+rd2_labels
	for k in range(nb):
		i = new_rd_labels[k][1]
		for l in range(nb):
			j = new_rd_labels[l][1]
			new_sim[k,l] = rd_sim[i,j]
		new_sim[k,k] = 0
	#plot the simat
	fig,ax = plt.subplots(constrained_layout=True,figsize=(8,8))
	fontsize = 12
	img = ax.imshow(new_sim,cmap='gnuplot2')
	im_ratio = new_sim.shape[0]/new_sim.shape[1]
	fig.colorbar(img,ax=ax,fraction=0.05*im_ratio)
	tick_labels = [el[0] for el in new_rd_labels]
	tick_loc = np.asarray(list(range(nb)))
	ax.set_xticks(tick_loc)
	ax.set_xticklabels(tick_labels,rotation=90,fontsize=fontsize)
	ax.set_yticks(tick_loc)
	ax.set_yticklabels(tick_labels,fontsize=fontsize)
	plt.savefig(os.path.join(PROJECT_ROOT,'conferences/Netsci2023/'+savefig+'only_rd.png'))
	plt.show()

def Extract_rd(labels,sorted_sim):
	#extract the randomized TN
	rd_labels = [(label,i) for i,label in enumerate(labels) if 'rd' in label]; nb = len(rd_labels)
	rd_sim = np.eye(nb)
	for k in range(nb):
		i = rd_labels[k][1]
		for l in range(nb):
			j = rd_labels[l][1]
			rd_sim[k,l] = sorted_sim[i,j]
	return [el[0] for el in rd_labels],rd_sim

#Get_sizes_TN(); exit()
#Get_fig_all(example_diag=False,basics=False,classes=True)

#restrict the similarity matrix sorted_sim to randomized and xp TN
def Simat_xp_rd(labels,sorted_sim,savefig):
	rd_labels = [(label,i) for i,label in enumerate(labels) if 'rd' in label or 'xp' in label]; nb = len(rd_labels)
	rd_sim = np.eye(nb)
	for k in range(nb):
		i = rd_labels[k][1]
		for l in range(nb):
			j = rd_labels[l][1]
			rd_sim[k,l] = sorted_sim[i,j]
	#permute lines of rd_sim so that the xp and randomized TN are separated
	new_sim = np.zeros((nb,nb))
	xp_labels = []; rd1_labels = []; rd2_labels = []
	for i,el in enumerate(rd_labels):
		if el[0]=="rd1":
			rd1_labels.append((el[0],i))
		elif el[0]=='rd2':
			rd2_labels.append((el[0],i))
		else:
			xp_labels.append((el[0],i))
	new_labels = xp_labels+rd1_labels+rd2_labels
	for k in range(nb):
		i = new_labels[k][1]
		for l in range(nb):
			j = new_labels[l][1]
			new_sim[k,l] = rd_sim[i,j]
		new_sim[k,k] = 0
	#plot the simat
	fig,ax = plt.subplots(constrained_layout=True,figsize=(8,8))
	fontsize = 13
	img = ax.imshow(new_sim,cmap='gnuplot2')
	im_ratio = new_sim.shape[0]/new_sim.shape[1]
	fig.colorbar(img,ax=ax,fraction=0.05*im_ratio)
	tick_labels = [el[0] for el in new_labels]
	tick_loc = np.asarray(list(range(nb)))
	ax.set_xticks(tick_loc)
	ax.set_xticklabels(tick_labels,rotation=90,fontsize=fontsize)
	ax.set_yticks(tick_loc)
	ax.set_yticklabels(tick_labels,fontsize=fontsize)
	plt.savefig(os.path.join(PROJECT_ROOT,'conferences/Netsci2023/'+savefig+'xp_rd.png'))
	plt.show()

choice = 'raw'; end_name = '_randomized12'
labels,sorted_sim = Extract_rd(*Load_xp_rdlabels(choice,end_name))
#Simat_only_rd(labels,sorted_sim,choice+'_simat'+end_name); exit()
#Plot_simat(labels,sorted_sim,choice+'_simat'+end_name+'xp_rd')only_rd
Plot_simat(labels,sorted_sim,choice+'_simat'+end_name+'only_rd')
exit()

#display the flows vs n and vs b
archive_name = "arch10"; figpath = "figures/Mathieu_fig/"
#('NCTN0sim',(2,'vsb')), ('ECTN0motif_error',(3,)), ('ICC', ()), ('edge0interduration0frac', ()), ('NCTN0motif_error',(3,))
list_obs = Get_list_obs(); cla = Symb_conv()
cla.Get_obs_folder_load(list_obs,archive_name=archive_name)
savename = 'ABPpi4'
for obs in [('NCTN0motif_error',(3,))]:
	print(obs)
	name_obs = Obs_to_nameobs(obs)
	cla.Display_flows_vsb(obs,savename,figpath+name_obs+"_"+savename)
	cla.Display_flows_vsn(obs,savename,figpath+name_obs+"_"+savename)
exit()

#analyze the randomized TN
archive_name = "arch10"
list_obs = Get_list_obs()
available_TN = Get_computed_TN(list_obs,archive_name,Get_tot_TN_randomized())
print('available_TN:\n',available_TN); exit()
end_name = '_randomized1'
not_rd_TN = [savename[:-len(end_name)] for savename in available_TN]
#compute sentences
Flows_to_sentences(archive_name,list_obs=list_obs,list_name=available_TN+not_rd_TN)
#analyze the flows
diag = Diagram()
diag.Load_sentence(conv_method='NN',end_name=archive_name)
diag.Get_diag()
diag.Compute_raw_sim(end_name=end_name)
mat_dist = diag.Get_EDM()
diag.Compute_metric_sim(mat_dist,end_name=end_name)
#display the sim matrices btw TN and choose the resolution parameter
for choice in ['raw','metric']:
	diag.Choose_resolution(choice=choice,end_name=end_name)
	diag.Display_simat(choice=choice,end_name=end_name)
