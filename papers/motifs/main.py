#study reversibility and causality in temporal networks
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)
from libs import ETN
from libs.settings import Raw_to_binned,Setup_Plot,Get_versions,Get_savename,Savename_to_name,Draw_simat,Cosim
import libs.Temp_net as tp
from libs import atn
from libs.settings import LIST_MARKER,LIST_COLOR,Cosim_triple

import numpy as np
import random as rd
import math
import matplotlib.pyplot as plt
import networkx as nx
from scipy.special import binom
from scipy.optimize import least_squares
from scipy.stats import pearsonr
import multiprocessing as mp

class Interpolate_distr:
	"""
	find the f_{e}(c), \Omega_{e}(n) and p_{e,k} which minimize the gap btw a function
	on tuples of integers (c,n_{1},...,n_{d}) and the function
	g(c,\vec{n}) = \sum_{e}f_{e}(c)\Omega_{e}(n)\prod_{k=1}^{d}\frac{p_{e,k}^{n_{k}}}{n_{k}!}
	where the nb of hidden states $e$ is given by nb_states and we restrict to values of n below n_bound
	hence the nb of parameters is nb_states*(n_bound + d + nb_c)
	index location of parameters in self.param:
	 - \Omega_{e}(n): e*n_bound + n
	 - p_{e,k}: nb_states*n_bound + e*d + k
	 - f_{e}(c): nb_states*(n_bound+d) + e*nb_c + c
	"""
	def __init__(self,nb_states):
		self.nb_states = nb_states
		self.n_bound = 0
		self.D = 0
		self.nb_c = 0
		self.param = None
		self.train_input = []
		self.train_target = None
		self.param_omega = None
		self.param_proba = None
		self.param_f = None
		self.list_factors = None

	#preprocess the data from histo before fitting
	#compute n_bound, n_sum and diverse factorials to speed up computations
	#train_input = list of ECTN in map representation (xp distribution)
	#train_input[k] = (c,n1,n2) where n1 and n2 are image under symmetry 1/2
	#train_target[k] = (proba that should be returned by the model for n_vec = train_input[k]) /
	# (binomial coefficient)
	#list_factors[k] = binomial coefficient by which we should multiply train_target[k] or
	#self.predict_for_fit()[k] to recover motifs proba
	def preprocess(self,dic_ECTN,depth):
		nb_train = len(dic_ECTN)
		self.train_input = []
		self.train_target = np.zeros(nb_train)
		self.list_factors = np.zeros(nb_train)
		profiles = ETN.All_ECTN_profiles(depth)[1]
		n_bound = 0
		#profiles[12sym[k]] = image of profiles[k] under 1/2 symmetry
		prof_to_ind = {profile:i for i,profile in enumerate(profiles)}
		sym_12 = [prof_to_ind[ETN.Swap_12_in_profile(profile)] for profile in profiles]
		powers_tab = 2**np.arange(depth)

		#compute train_input and train_target
		for k,(seq,nb) in enumerate(dic_ECTN.items()):
			central_prof,n_vec1 = ETN.String_to_map_ECTN(depth,profiles,seq)
			n_vec1 = tuple(n_vec1)
			n_vec2 = (n_vec1[new_ind] for new_ind in sym_12)
			c = np.sum(np.array(list(central_prof),dtype=int)*powers_tab)-1
			self.train_input.append((c,n_vec1,tuple(n_vec2)))
			n_sum = sum(n_vec1)
			if n_bound<n_sum:
				n_bound = n_sum
			factor = math.factorial(n_sum)
			for n in n_vec1:
				factor /= math.factorial(n)
			self.train_target[k] = nb/factor
			self.list_factors[k] = factor

		self.n_bound = n_bound + 1
		self.D = len(profiles)
		self.nb_c = 2**depth - 1
		#param[:a0] = param_omega
		#param[a0:a1] = param_proba
		#param[a1:a2] = param_f
		self.a0 = self.nb_states*self.n_bound
		self.a1 = self.a0 + self.nb_states*self.D
		self.a2 = self.a1 + self.nb_states*self.nb_c
		self.param = np.ones(self.a2)

		#tab_input[k,:] = [c,*n_vec1,*n_vec2,n]
		#equal_vec = list of row indices where n_vec1==n_vec2
		#diff_vec = list of row indices where n_vec1!=n_vec2
		self.tab_input = np.array([[c,*n_vec1,*n_vec2,sum(n_vec1)] for c,n_vec1,n_vec2 in self.train_input],dtype=int)
		self.equal_vec = []
		self.diff_vec = []
		for ind,el in enumerate(self.train_input):
			if el[1]==el[2]:
				self.equal_vec.append(ind)
			else:
				self.diff_vec.append(ind)

		print('nb of training examples',len(self.train_input))
		print('nb of param',self.nb_states*(self.n_bound+self.D+self.nb_c))
		print('n_bound',self.n_bound)
		print('D',self.D)
		print('nb_c',self.nb_c)
		print('nb_states',self.nb_states)

	#unfold param to deduce the three param vectors ; also normalize the p_{e,k}
	def set_param(self,param):
		self.param[:] = param[:]
		#normalization of param_proba: index e starts at e*D
		for e in range(self.nb_states):
			norm = np.sum(param[self.a0+e*self.D:self.a0+(e+1)*self.D])
			self.param[self.a0+e*self.D:self.a0+(e+1)*self.D] /= norm

	#same as self.predict but applied to every element in self.train_input
	def predict_for_fit(self):
		return np.array([self.predict(el) for el in self.train_input])

	#returns the difference btw the model output for param and target values for elements of self.train_input
	def eval_func(self,param):
		self.set_param(param)
		return self.predict_for_fit_new()-self.train_target

	#find the \Omega_{e}(n) and p_{e,k} which minimize the gap btw a function on tuples of integers
	#(n_{1},...,n_{d}) and the function
	#g(\vec{n}) = \sum_{e}\Omega_{e}(n)\prod_{k=1}^{d}\frac{p_{e,k}^{n_{k}}}{n_{k}!}
	#where the nb of hidden states $e$ is given by nb_states and we restrict to values of n below n_bound
	#hence the nb of parameters is nb_states*(n_bound + d)
	def fit_model(self):
		return least_squares(self.eval_func,self.param,bounds=(0,np.inf)).x

	def predict_for_fit_new(self):
		#if n_vec1 == n_vec2
		res = np.zeros(self.train_target.shape)
		for e in range(self.nb_states):
			tab_prod = np.prod(np.power(self.param[self.a0+e*self.D:self.a0+(e+1)*self.D],self.tab_input[self.equal_vec,1:1+self.D]),axis=1)
			res[self.equal_vec] += self.param[self.a1+e*self.nb_c+self.tab_input[self.equal_vec,0]]*self.param[e*self.n_bound+self.tab_input[self.equal_vec,-1]]*tab_prod
		#if n_vec!=n_vec2
		for e in range(self.nb_states):
			prefix_tab = self.param[self.a1+e*self.nb_c+self.tab_input[self.diff_vec,0]]*self.param[e*self.n_bound+self.tab_input[self.diff_vec,-1]]
			prod = 0
			for k in range(2):
				prod += np.prod(np.power(self.param[self.a0+e*self.D:self.a0+(e+1)*self.D],self.tab_input[self.diff_vec,1+k*self.D:1+(k+1)*self.D]),axis=1)
			res[self.diff_vec] += prefix_tab*prod
		return res

	#return the value of the interpolated function for the input el = (c,n_vec1,n_vec2)
	def predict(self,el):
		c,n_vec1,n_vec2 = el
		n = sum(n_vec1)
		res = 0
		if n_vec1==n_vec2:
			for e in range(self.nb_states):
				prod = np.prod(np.power(self.param[self.a0+e*self.D:self.a0+(e+1)*self.D],n_vec1))
				res += self.param[self.a1+e*self.nb_c+c]*self.param[e*self.n_bound+n]*prod
		else:
			for e in range(self.nb_states):
				prod = 0
				for n_vec in [n_vec1,n_vec2]:
					prod += np.prod(np.power(self.param[self.a0+e*self.D:self.a0+(e+1)*self.D],n_vec))
				res += self.param[self.a1+e*self.nb_c+c]*self.param[e*self.n_bound+n]*prod
		return res

class Interpolate_distr_Old:
	"""
	find the f_{e}(c), \Omega_{e}(n) and p_{e,k} which minimize the gap btw a function
	on tuples of integers (c,n_{1},...,n_{d}) and the function
	g(c,\vec{n}) = \sum_{e}f_{e}(c)\Omega_{e}(n)\prod_{k=1}^{d}\frac{p_{e,k}^{n_{k}}}{n_{k}!}
	where the nb of hidden states $e$ is given by nb_states and we restrict to values of n below n_bound
	hence the nb of parameters is nb_states*(n_bound + d + nb_c)
	index location of parameters in self.param:
	 - \Omega_{e}(n): e*n_bound + n
	 - p_{e,k}: nb_states*n_bound + e*d + k
	 - f_{e}(c): nb_states*(n_bound+d) + e*nb_c + c
	"""
	def __init__(self,nb_states):
		self.nb_states = nb_states
		self.n_bound = 0
		self.D = 0
		self.nb_c = 0
		self.param = None
		self.train_input = []
		self.train_target = None
		self.param_omega = None
		self.param_proba = None
		self.param_f = None
		self.list_factors = None

	#preprocess the data from histo before fitting
	#compute n_bound, n_sum and diverse factorials to speed up computations
	#train_input = list of ECTN in map representation (xp distribution)
	#train_input[k] = (c,n1,n2) where n1 and n2 are image under symmetry 1/2
	#train_target[k] = (proba that should be returned by the model for n_vec = train_input[k]) /
	# (binomial coefficient)
	#list_factors[k] = binomial coefficient by which we should multiply train_target[k] or
	#self.predict_for_fit()[k] to recover motifs proba
	def preprocess(self,dic_ECTN,depth):
		nb_train = len(dic_ECTN)
		self.train_input = []
		self.train_target = np.zeros(nb_train)
		self.list_factors = np.zeros(nb_train)
		profiles = ETN.All_ECTN_profiles(depth)[1]
		n_bound = 0
		#profiles[12sym[k]] = image of profiles[k] under 1/2 symmetry
		prof_to_ind = {profile:i for i,profile in enumerate(profiles)}
		sym_12 = [prof_to_ind[ETN.Swap_12_in_profile(profile)] for profile in profiles]
		powers_tab = 2**np.arange(depth)

		#compute train_input and train_target
		for k,(seq,nb) in enumerate(dic_ECTN.items()):
			central_prof,n_vec1 = ETN.String_to_map_ECTN(depth,profiles,seq)
			n_vec1 = tuple(n_vec1)
			n_vec2 = (n_vec1[new_ind] for new_ind in sym_12)
			c = np.sum(np.array(list(central_prof),dtype=int)*powers_tab)-1
			self.train_input.append((c,n_vec1,tuple(n_vec2)))
			n_sum = sum(n_vec1)
			if n_bound<n_sum:
				n_bound = n_sum
			factor = math.factorial(n_sum)
			for n in n_vec1:
				factor /= math.factorial(n)
			self.train_target[k] = nb/factor
			self.list_factors[k] = factor

		self.n_bound = n_bound + 1
		self.D = len(profiles)
		self.nb_c = 2**depth - 1
		#param_omega[e,n] = \Omega_{e}(n)
		self.param_omega = np.ones((self.nb_states,self.n_bound))
		#param_proba[e,k] = p_{e,k}
		self.param_proba = np.ones((self.nb_states,self.D))
		#param_f[e,c] = f_{e}(c)
		self.param_f = np.ones((self.nb_states,self.nb_c))

		print('nb of training examples',len(self.train_input))
		print('nb of param',self.nb_states*(self.n_bound+self.D+self.nb_c))
		print('n_bound',self.n_bound)
		print('D',self.D)
		print('nb_c',self.nb_c)
		print('nb_states',self.nb_states)

	#combine the different param into a single vector of param
	def flatten_param(self):
		return np.array(self.param_omega.flatten().tolist()+\
			self.param_proba.flatten().tolist()+\
			self.param_f.flatten().tolist())

	#unfold param to deduce the three param vectors ; also normalize the p_{e,k}
	def set_param_from_flat(self,param):
		#unfold the parameters
		self.param_omega = param[:self.nb_states*self.n_bound].reshape(self.nb_states,self.n_bound)
		a = self.nb_states*self.n_bound; l = self.nb_states*self.D
		self.param_proba = param[a:a+l].reshape(self.nb_states,self.D)
		a += l; l = self.nb_states*self.nb_c
		self.param_f = param[a:a+l].reshape(self.nb_states,self.nb_c)
		#normalize the p_{e,k}
		self.param_proba = self.param_proba/np.tile(np.sum(self.param_proba,axis=0),(self.nb_states,1))

	#same as self.predict but applied to every element in self.train_input
	def predict_for_fit(self):
		return np.array([self.predict(el) for el in self.train_input])

	#returns the difference btw the model output for param and target values for elements of self.train_input
	def eval_func(self,param):
		self.set_param_from_flat(param)
		return self.predict_for_fit()-self.train_target

	#find the \Omega_{e}(n) and p_{e,k} which minimize the gap btw a function on tuples of integers
	#(n_{1},...,n_{d}) and the function
	#g(\vec{n}) = \sum_{e}\Omega_{e}(n)\prod_{k=1}^{d}\frac{p_{e,k}^{n_{k}}}{n_{k}!}
	#where the nb of hidden states $e$ is given by nb_states and we restrict to values of n below n_bound
	#hence the nb of parameters is nb_states*(n_bound + d)
	def fit_model(self):
		return least_squares(self.eval_func,self.flatten_param(),bounds=(0,np.inf)).x

	#return the value of the interpolated function for the input el = (c,n_vec1,n_vec2)
	def predict(self,el):
		c,n_vec1,n_vec2 = el
		n = sum(n_vec1)
		if n_vec1==n_vec2:
			prod = np.prod(np.power(self.param_proba,n_vec1),axis=1)
			return np.sum(self.param_f[:,c]*self.param_omega[:,n]*prod)
		else:
			prod = 0
			for n_vec in [n_vec1,n_vec2]:
				prod += np.prod(np.power(self.param_proba,n_vec),axis=1)
			return np.sum(self.param_f[:,c]*self.param_omega[:,n]*prod)

#if tuple_keys = (k0,k1,...,kn) then initialize to 1 the value dic[kn]...[k0]
def Initialize_dic(dic,tuple_keys):
	if len(tuple_keys)==1:
		key = tuple_keys[0]
		dic[key] = 1
		return None
	key = tuple_keys[-1]
	dic[key] = {}
	return Initialize_dic(dic[key],tuple_keys[:-1])

#if tuple_keys = (k0,k1,...,kn) then increase by 1 the value dic[kn]...[k0]
#and add the non-existing keys
def Increase_dic(dic,tuple_keys):
	if len(tuple_keys)==1:
		key = tuple_keys[0]
		if key in dic:
			dic[key] += 1
		else:
			dic[key] = 1
		return None
	key = tuple_keys[-1]
	if key in dic:
		return Increase_dic(dic[key],tuple_keys[:-1])
	return Initialize_dic(dic,tuple_keys)

class Gluing_mat1d:
	"""
	exact number of gluing matrices for an integer n and a vector integer m of arbitrary size
	"""
	def __init__(self):
		#memorize the numbers already computed
		self.hash_dic1d = {}

	#same as Succ_mat but n is an integer and m is a vector integer
	def Succ_mat1d(self,block,n,m):
		#conditions for the only row
		cond_row = np.sum(block)
		#try to increase the last element of the matrix (elements are ordered in western reading convention,
		#from left to right and top to bottom, i.e. increasing column and row indices)
		j = len(m)-1 #last position in the matrix
		while True:
			#try to increase of one unit the entry (i,j)
			if cond_row<n and block[j]<m[j]:
				block[j] += 1
				return None
			#if impossible, reset the entry (i,j) and increase the closest previous entry of one unit
			cond_row -= block[j]
			block[j] = 0
			#change for the previous entry
			j -= 1
			if j<0:
				return None

	def Exact_gluing_mat12(self,n,m1,m2):
		if m1+m2<n:
			return 1 + m2 +m1*(1+m2)
		a = min(n,m1); b = min(n,m2)
		return 1 + b + m1*(n-a) + (a + n - b)*(1 + a - n + b)

	#exact number of gluing matrices for an integer n and a vector integer m
	def Exact_gluing_mat1d(self,n,m):
		if len(m)==2:
			return self.Exact_gluing_mat12(n,m[0],m[1])
		if len(m)==1:
			return 1 + min(n,m[0])
		if len(m)==0:
			return 1
		if np.sum(m)<=n:
			res = 1
			for el in m:
				res *= (1+el)
			return res
		#if the value has already been computed
		if str(n)+'|'+str(m) in self.hash_dic1d:
			return self.hash_dic1d[str(n)+'|'+str(m)]
		'''
		if n<=np.min(m):
			pass
		'''
		#split the gluing matrix into two blocks of equal size and sum over the elements of one block
		sc = len(m)//2
		#sum over all possible out-of-diagonal blocks
		block_21 = np.zeros(sc,dtype=int)
		res = 0; keep = True
		while keep:
			res += self.Exact_gluing_mat1d(n-np.sum(block_21),m[sc:])
			self.Succ_mat1d(block_21,n,m[:sc])
			if (block_21==0).all():
				keep = False
		#store the result
		self.hash_dic1d[str(n)+'|'+str(m)] = res
		return res

class Gluing_mat(Gluing_mat1d):
	"""
	exact number of gluing matrices for n and m vector integers of arbitrary size, even
	possibly different
	"""
	def __init__(self):
		#memorize the numbers already computed
		self.hash_dic = {}
		super(Gluing_mat,self).__init__()

	#return the successor of the integer valued matrix of size len(n)xlen(m)
	#used in Exact_gluing_mat
	def Succ_mat(self,block,n,m):
		#conditions for each row
		cond_row = np.sum(block,axis=1)
		#conditions for each column
		cond_col = np.sum(block,axis=0)
		#try to increase the last element of the matrix (elements are ordered in western reading convention,
		#from left to right and top to bottom, i.e. increasing column and row indices)
		i,j = len(n)-1,len(m)-1 #last position in the matrix
		while True:
			#try to increase of one unit the entry (i,j)
			if cond_row[i]<n[i] and cond_col[j]<m[j]:
				block[i,j] += 1
				return None
			#if impossible, reset the entry (i,j) and increase the closest previous entry of one unit
			cond_row[i] -= block[i,j]
			cond_col[j] -= block[i,j]
			block[i,j] = 0
			#change for the previous entry
			j -= 1
			if j<0:
				j = len(m)-1
				i -= 1
				if i<0:
					return None

	#compute the exact number of gluing matrices for integer vectors of size 1
	def Exact_gluing_mat1(self,n,m):
		return 1 + min(n,m)
	def Exact_gluing_mat2(self,n,m):
		res = 0
		for b1 in range(min(n[0],m[0])+1):
			for b2 in range(min(n[1],m[1])+1):
				res += self.Exact_gluing_mat1(n[0]-b1,m[1]-b2)*self.Exact_gluing_mat1(n[1]-b2,m[0]-b1)
		return res
	def Exact_gluing_mat3(self,n,m):
		res = 0
		for b13 in range(min(n[0],m[2])+1):
			for b31 in range(min(n[2],m[0])+1):
				for b23 in range(min(n[1],m[2]-b13)+1):
					for b32 in range(min(n[2]-b31,m[1])+1):
						res += self.Exact_gluing_mat1(n[2]-b31-b32,m[2]-b13-b23)*self.Exact_gluing_mat2([n[0]-b13,n[1]-b23],[m[0]-b31,m[1]-b32])
		return res

	#take a vector and remove its zero components
	def Clean_zeroes(self,n):
		return np.array([el for el in n if el!=0])

	#compute the exact number of gluing matrices for integer vectors of arbitrary and possibly
	#different sizes
	def Exact_gluing_mat(self,n,m):
		n,m = self.Clean_zeroes(n),self.Clean_zeroes(m)
		if len(n)==1:
			return self.Exact_gluing_mat1d(n[0],m)
		if len(m)==1:
			return self.Exact_gluing_mat1d(m[0],n)
		if len(n)==0 or len(m)==0:
			return 1
		#if the value has already been computed
		if str(n)+'|'+str(m) in self.hash_dic:
			return self.hash_dic[str(n)+'|'+str(m)]
		elif str(m)+'|'+str(n) in self.hash_dic:
			return self.hash_dic[str(m)+'|'+str(n)]
		res = 0
		#split the gluing matrix into two diagonal blocks of equal size and sum over the elements outside
		#the blocks
		sl = len(n)//2; sc = len(m)//2
		#sum over all possible out-of-diagonal blocks
		block_12 = np.zeros((sl,len(m)-sc),dtype=int)
		block_21 = np.zeros((len(n)-sl,sc),dtype=int)
		keep = True
		while keep:
			keep2 = True
			s_row_b21 = np.sum(block_21,axis=0)
			s_col_b21 = np.sum(block_21,axis=1)
			while keep2:
				res += self.Exact_gluing_mat(n[:sl]-np.sum(block_12,axis=1),m[:sc]-s_row_b21)*self.Exact_gluing_mat(n[sl:]-s_col_b21,m[sc:]-np.sum(block_12,axis=0))
				self.Succ_mat(block_12,n[:sl],m[sc:])
				if (block_12==0).all():
					keep2 = False
			self.Succ_mat(block_21,n[sl:],m[:sc])
			if (block_21==0).all():
				keep = False
		#store the result
		self.hash_dic[str(n)+'|'+str(m)] = res
		return res

	#return the number of ECTN which contains n1 and n2 as NCTN sub-motifs
	def Exact_invert_nb(self,n1,n2):
		#identify the profiles common to n1 and n2
		common_prof = [i for i,(nb1,nb2) in enumerate(zip(n1,n2)) if nb1*nb2!=0]
		res = 0; vec = np.zeros(n1.shape,dtype=int)
		for i in common_prof:
			vec[i] = 1
			res += self.Exact_gluing_mat(n1-vec,n2-vec)
			vec[i] = 0
		return res

#return an estimate of the number of gluing matrices btw the integer vectors n1 and n2
def Approx_gluing_mat_nb(n1,n2):
	res = 1
	for x in n1:
		for y in n2:
			res *= 1 + min(x,y)
	return res

#return an estimate of the number of ECTN which contains n1 and n2 as NCTN sub-motifs
def Approx_invert_nb(n1,n2):
	#identify the profiles common to n1 and n2
	common_prof = [i for i,(nb1,nb2) in enumerate(zip(n1,n2)) if nb1*nb2!=0]
	res = 0; vec = np.zeros(n1.shape)
	for i in common_prof:
		vec[i] = 1
		res += Approx_gluing_mat_nb(n1-vec,n2-vec)
		vec[i] = 0
	return res

#normalize an histogram written as a dictionary of arbitrary depth:
#for depth 2, after normalization, histo[key1][key2] = Proba(key2|key1)
def Norm_dic_histo(histo):
	bottom_found = False; norm = 0
	for key,val in histo.items():
		if type(val)!=dict:
			bottom_found = True
			norm += val
	if bottom_found:
		return {key:val/norm for key,val in histo.items()}
	return {key:Norm_dic_histo(histo[key]) for key in histo.keys()}

#compute the not normalized histogram of the activity duration
#for the list of events event_train
def Duration_histo(event_train):
	histo = {}
	for val in event_train.values():
		for el in val:
			duration = el[1]-el[0]+1
			if duration in histo:
				histo[duration] += 1
			else:
				histo[duration] = 1
	#if no realization occurs, we put an impossible one
	if not histo:
		return {0.1:1}
	return histo

#return the interduration distr for event_train
def Interduration_histo(event_train):
	histo = {}
	for val in event_train.values():
		for n in range(1,len(val)):
			interduration = val[n][0]-val[n-1][1]-1
			if interduration in histo:
				histo[interduration] += 1
			else:
				histo[interduration] = 1
	#if no realization occurs, we put an impossible one
	if not histo:
		return {0.1:1}
	return histo

#compute the time weight from event_train
def Time_weight_histo(event_train):
	histo = {}
	for val in event_train.values():
		w = 0
		for el in val:
			duration = el[1]-el[0]+1
			w += duration
		if w in histo:
			histo[w] += 1
		else:
			histo[w] = 1
		w = 1
	#if no realization occurs, we put an impossible one
	if not histo:
		return {0.1:1}
	return histo

#compute the not normalized histogram of the weak duration for events in event_train
def Train_weak_duration_histo(event_train,delay=2):
	histo = {}
	for val in event_train.values():
		if len(val)==1:
			if 1 in histo:
				histo[1] += 1
			else:
				histo[1] = 1
		else:
			#nb of consecutive events
			nb = 1
			for n in range(1,len(val)):
				interduration = val[n][0]-val[n-1][1]-1
				if interduration<delay:
					nb += 1
				else:
					if nb in histo:
						histo[nb] += 1
					else:
						histo[nb] = 1
					nb = 1
	#if no realization occurs, we put an impossible one
	if not histo:
		return {0.1:1}
	return histo

#rewrite an ECTN profile by exchanging 1 and 2 so that the first 1 or 2 encountered is a 1
#allows to consider profiles under symmetry under 1<-->2
def Rewrite_ECTN_prof_sym12(seq):
	#find the first occurrence of a 2
	ind_first2 = seq.find('2')
	#if there is no 2, seq is unchanged
	if ind_first2<0:
		return seq
	ind_first1 = seq.find('1')
	#if we encounter 1 before 2, seq is unchanged
	if ind_first1>=0 and ind_first1<ind_first2:
		return seq
	#exchange 1 and 2
	new_seq = list(seq)
	for ind,letter in enumerate(seq):
		if letter=='2':
			new_seq[ind] = '1'
		elif letter=='1':
			new_seq[ind] = '2'
	return ''.join(new_seq)

#identify a profile with its image under 1<-->2
#also separate the transverse from single satellite profiles
def Get_trans_single_sym12prof(satellites_prof):
	sym_prof = {Rewrite_ECTN_prof_sym12(seq) for seq in satellites_prof}
	trans_prof = set(); single_prof = set()
	for seq in sym_prof:
		#transverse profile: satellite common to both nodes of the central edge
		if '3' in seq or '1' in seq and '2' in seq:
			trans_prof.add(seq)
		#single profile (note this is a NCTN profile)
		else:
			single_prof.add(seq)
	return trans_prof,single_prof

#compute the weight motif histogram: histo[w] = nb of distinct motifs that realize w times in the network
def Get_weight_histo(dic_CTN):
	histo = {}
	for w in dic_CTN.values():
		if w in histo:
			histo[w] += 1
		else:
			histo[w] = 1
	return histo

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

#return the nb first levels of degenerescence sorted by increasing order for the function
#lambda n,m: n*p + m*q, where p,q>0
def Get_ordering(p,q,nb):
	#list of ordered couples of integers
	res = []
	#set (unordered) of candidates for the successor of the maximal element in res
	adj_res = {(1,0),(0,1)}
	while len(res)<nb:
		#build the successor of res[-1]
		successor = min(list(adj_res),key=lambda cand:cand[0]*p + cand[1]*q)
		res.append(successor)
		#update the list of candidates
		adj_res.remove(successor)
		adj_res.add((successor[0]+1,successor[1]))
		adj_res.add((successor[0],successor[1]+1))
	return res

#return all the couples of integers (n,m) such that n + m = s
def Get_equal_sum(s):
	return [(n,s-n) for n in range(s+1)]

#compute the ECTN consistent asymmetry coefficient under TR, defined as:
#we extract the m in dic_ECTN such that m<TR(m) and return sum_{m}(dic_ECTN[TR(m)]-dic_ECTN[m])
def Asym_TR(dic_ECTN,depth):
	#extract asym ECTN and take the image under TR
	res = 0; nb_asym = 0
	for seq,nb in dic_ECTN.items():
		new_seq = ETN.TR_ECTNimage(seq,depth)
		if seq<new_seq:
			nb_asym += 1
			if new_seq in dic_ECTN:
				res += dic_ECTN[new_seq]-nb
			else:
				res -= nb
	#if no asym motif has been found, return the minimum value for the asym
	if nb_asym==0:
		return 0
	return res/nb_asym

#compute the ECTN proba under Maximum Entropy Principle
#under the constraints of given NCTN number dic_NCTN
def Get_indMEP_ECTN(dic_NCTN,depth,list_ECTN,restrict):
	#the first step is to compute the coefficients |E_{m,m'}\cup E_{m',m}|
	list_NCTN,vec_NCTN = zip(*dic_NCTN.items())
	vec_NCTN,list_NCTN = Restrict_seq(vec_NCTN,list_NCTN,freq_ratio=restrict)
	list_NCTN = list(list_NCTN); nb_NCTN = len(list_NCTN); vec_NCTN = np.asarray(vec_NCTN)
	seq_to_int = {seq:i for i,seq in enumerate(list_NCTN)}
	for i,seq in enumerate(list_NCTN):
		vec_NCTN[i] *= len(seq)
	NCTN_profiles = ETN.All_NCTN_profiles(depth)
	print(str(np.sum(list(dic_NCTN.values())))+' xp NCTN')
	print(str(np.sum(vec_NCTN)/(2*depth))+' ECTN')
	print(str(nb_NCTN)+' distinct NCTN')
	#compute the equation coefficients
	eq_coeff = np.zeros((nb_NCTN,nb_NCTN))
	#put strings under their ordered map representation
	list_map_NCTN = [ETN.String_to_map_NCTN(depth,NCTN_profiles,seq) for seq in list_NCTN]
	gluMat = Gluing_mat()
	print('begin to compute equation coefficients')
	for ind1,n1 in enumerate(list_map_NCTN):
		for ind2,n2 in enumerate(list_map_NCTN[ind1:]):
			eq_coeff[ind1,ind1+ind2] = gluMat.Exact_invert_nb(n1,n2)
			eq_coeff[ind1+ind2,ind1] = eq_coeff[ind1,ind1+ind2]
		eq_coeff[ind1,ind1] *= 2
	print('equation coefficients computed')
	#solve the equation of unknowns gamma_{m} : add the fprime with col_deriv = 1
	gamma0 = np.random.random(nb_NCTN)*2
	gamma = least_squares(lambda gamma:gamma*np.dot(eq_coeff,gamma)-vec_NCTN,gamma0,\
		bounds = (0,np.inf)\
		).x
	#print the residual
	print('residual error:',np.sum(abs(gamma*np.dot(eq_coeff,gamma)-vec_NCTN)))
	#deduce the ECTN distribution
	th_proba = []
	for seq in list_ECTN:
		seq1,seq2 = ETN.Sub_NCTN(seq,depth)
		if seq1 in seq_to_int and seq2 in seq_to_int:
			th_proba.append(gamma[seq_to_int[seq1]]*gamma[seq_to_int[seq2]])
		else:
			th_proba.append(0)
	th_norm = np.sum(th_proba)
	for ind,el in enumerate(th_proba):
		th_proba[ind] /= th_norm
	return th_proba

def Get_ind_NCTN(datapath,depth,restrict):
	list_seq = Load_list_seq_restrict(datapath,restrict)
	xp_proba = Load_xp_proba_restrict(datapath,restrict)
	xp_proba /= np.sum(xp_proba)
	#compute Nb_sat_histo and prof_histo
	Nb_sat_histo = {}; prof_histo = {profile:0 for profile in ETN.All_NCTN_profiles(depth)}
	for seq,nb in zip(list_seq,xp_proba):
		nb_sat = len(seq)//depth
		if nb_sat in Nb_sat_histo:
			Nb_sat_histo[nb_sat] += nb
		else:
			Nb_sat_histo[nb_sat] = nb
		for i in range(len(seq)//depth):
			prof_histo[seq[i*depth:(i+1)*depth]] += nb
	Nb_sat_histo = Norm_dic_histo(Nb_sat_histo); prof_histo = Norm_dic_histo(prof_histo)

	Y = []; nb_profile = {profile:0 for profile in ETN.All_NCTN_profiles(depth)}
	for seq in list_seq:
		nb_sat = len(seq)//depth
		for i in range(nb_sat):
			nb_profile[seq[i*depth:(i+1)*depth]] += 1

		th_proba = Nb_sat_histo[nb_sat]*math.factorial(nb_sat)
		for prof,nb in nb_profile.items():
			if nb>0:
				th_proba *= (prof_histo[prof]**nb)/math.factorial(nb)
		Y.append(th_proba)
		for key in nb_profile.keys():
			nb_profile[key] = 0
	return Y

class Motifs_tp:
	"""
	shares a lot with Temp_net but is a proper and smaller version
	"""
	def __init__(self,savename,where=None):
		if where is not None:
			self.data = np.loadtxt(where,dtype=int)
		#measurement data
		elif type(savename)==str:
			try:
				self.data = np.loadtxt(PROJECT_ROOT+'data/'+savename+'.txt',dtype=int)
			except:
				self.data = tp.Load_TN(Savename_to_name(savename))
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

def Compare_NCTN_to_theory(P_01,P_11,lamb):
	depth = 2
	motifs = Motifs_tp(np.loadtxt('min_EW3_RS.txt',dtype=int))
	motifs.Get_TN(1)
	xp_proba = motifs.Get_dic_NCTN(depth)
	norm = sum(xp_proba.values())
	th_proba = {}
	dic_nb = {profile:0 for profile in All_NCTN_profiles(depth)}
	for seq,val in xp_proba.items():
		xp_proba[seq] = val/norm
		for i in range(len(seq)//depth):
			dic_nb[seq[i*depth:(i+1)*depth]] += 1
		n_0 = dic_nb['01']; n_1 = dic_nb['10']; n_2 = dic_nb['11']
		th_proba[seq] = lamb*P_01**(n_0+n_1)*P_11**n_2/(math.factorial(n_0)*math.factorial(n_1)*math.factorial(n_2))
		for key in dic_nb.keys():
			dic_nb[key] = 0
	#plot the two distributions against each other
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	X = []; Y = []
	for seq in xp_proba.keys():
		X.append(th_proba[seq])
		Y.append(xp_proba[seq])
	ax.set_xlabel('th_proba')
	ax.set_ylabel('xp_proba')
	ax.plot(X,Y,'.')
	ax.plot(X,X,'--')
	plt.show()

#compare the true NCTN distribution with the NCTN proba computed under hypothesis
#of complete spatial independence btw activity profiles
def Check_spatial_correlations_NCTN(name,depth):
	motifs = Motifs_tp(name)
	motifs.Get_TN(1)
	dic_NCTN = motifs.Get_dic_NCTN(depth)
	#compute the NCTN proba under the hypothesis of no spatial correlations btw activity profiles
	X,Y = motifs.Get_strong_ind_NCTN(dic_NCTN,depth)
	#compare the two distributions
	fontsize = 16
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_xlabel('xp_proba',fontsize=fontsize)
	ax.set_ylabel('th_proba',fontsize=fontsize)
	ax.plot(X,Y,'.')
	ax.plot(X,X,'--')
	#check folders exist
	path = 'figures/NCTN/ind_hyp/'+name
	if not os.path.isdir(path):
		os.mkdir(path)
	path += '/Check_spatial_correlations_NCTN'
	if not os.path.isdir(path):
		os.mkdir(path)
	path += '/'
	plt.savefig(path+'depth'+str(depth)+'.png')
	plt.show()

#similar as Check_spatial_correlations_NCTN but assuming another independence hypothesis, which leads
#to the functional form:
#P(NCTN) = P_agg_deg(nb of satellites)*fact(s)*\prod_{k}[(\tilde{P}_k)**n_k/fact(n_k)]
def Check_weak_ind_hyp_NCTN(name,depth):
	motifs = Motifs_tp(name)
	motifs.Get_TN(1)
	#compute the NCTN proba under the weak spatial ind hypothesis
	X,Y = motifs.Get_weak_ind_NCTN(motifs.Get_dic_NCTN(depth),depth)
	#compare the two distributions
	fontsize = 16
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_xlabel('xp_proba',fontsize=fontsize)
	ax.set_ylabel('th_proba',fontsize=fontsize)
	ax.plot(X,Y,'.')
	ax.plot(X,X,'--')
	#check folders exist
	path = 'figures/NCTN/ind_hyp/'+name
	if not os.path.isdir(path):
		os.mkdir(path)
	path += '/Check_weak_ind_hyp_NCTN'
	if not os.path.isdir(path):
		os.mkdir(path)
	path += '/'
	plt.savefig(path+'depth'+str(depth)+'.png')
	plt.show()

#plot the aggregated degree distribution transformed so as to obtain a straight line in a certain limit
def Plot_transfo_degree(name,agg):
	motifs = Motifs_tp(name)
	motifs.Get_TN(agg)
	deg_proba = {}
	for graph in motifs.TN:
		for i in graph.nodes:
			degree = graph.degree(i)
			if degree in deg_proba:
				deg_proba[degree] += 1
			else:
				deg_proba[degree] = 1
	norm = sum(deg_proba.values())
	for key,val in deg_proba.items():
		deg_proba[key] = val/norm
	fontsize = 14
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_xlabel('degree aggregated on '+str(agg),fontsize=fontsize)
	ax.set_ylabel('probability',fontsize=fontsize)
	X,Y = zip(*deg_proba.items())
	list_alpha = [0,0.5]
	for alpha in list_alpha:
		ax.plot(X,np.log10(Y)+alpha*np.log10(X),'.',label=r'$\alpha = $'+str(alpha))
	M_X = np.max(X)
	for x in X:
		if x==1:
			m_Y = math.log10(deg_proba[x])
		elif x==M_X:
			M_Y = math.log10(deg_proba[x])
	ax.plot([1,M_X],[m_Y,M_Y],'--',label='straight line')
	ax.legend(fontsize=fontsize)
	plt.show()

def Compare_degree_distr(name,agg,entropic_factor=False,gaussian_hyp=False):
	motifs = Motifs_tp(name)
	motifs.Get_TN(agg)
	deg_proba = {0:0}
	for graph in motifs.TN:
		for i in graph.nodes:
			degree = graph.degree(i)
			if degree in deg_proba:
				deg_proba[degree] += 1
			else:
				deg_proba[degree] = 1
		deg_proba[0] += motifs.N-len(graph.nodes)
	norm = sum(deg_proba.values()); q = deg_proba[0]/norm
	print('q = '+str(q))
	del deg_proba[0]
	fontsize = 14
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_xlabel('degree aggregated on '+str(agg),fontsize=fontsize)
	ax.set_ylabel('probability',fontsize=fontsize)
	X,Y = zip(*deg_proba.items())
	X = np.asarray(X); Y = np.asarray(Y); Y = np.log(Y/np.max(Y))
	if entropic_factor:
		savefig = 'entropic_factor'
		ax.plot(X,Y+(X+0.5)*np.log(X),'.')
	elif gaussian_hyp:
		savefig = 'gaussian_hyp'
		ax.plot(X,np.sqrt(-Y),'.')
	else:
		savefig = ''
		ax.plot(X,Y,'.')
	path = 'figures/deg_inst/'+name+'/'
	if not os.path.isdir(path):
		os.mkdir(path)
	plt.savefig(path+'agg_'+str(agg)+savefig+'.png')
	#ax.plot(X,X*math.log(q/(1-q)),'.')
	plt.show()

def Flow_degree_distr(name):
	motifs = Motifs_tp(name)
	list_agg = [i for i in range(1,31)]
	list_q = []
	for agg in list_agg:
		print(agg)
		motifs.Get_TN(agg)
		#sample degree distribution
		deg_proba = {}
		for graph in motifs.TN:
			for i in graph.nodes:
				degree = graph.degree(i)
				if degree in deg_proba:
					deg_proba[degree] += 1
				else:
					deg_proba[degree] = 1
		#compute q such that P(deg=s)\propto q^{s}
		X,Y = zip(*deg_proba.items())
		X = np.asarray(X); Y = np.log(np.asarray(Y))
		#perform linear regression to extract log(q)
		x_m = np.mean(X); y_m = np.mean(Y)
		beta = np.sum((X-x_m)*(Y-y_m))/np.sum((X-x_m)**2)
		list_q.append(math.exp(beta))
	#plot log(q_{agg+1}-q_{agg}) vs log(agg)
	fontsize = 14; list_q = np.asarray(list_q)
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_xlabel(r'$\log(n)$',fontsize=fontsize)
	ax.set_ylabel(r'$\log(q_{n+1}-q_{n})$',fontsize=fontsize)
	ax.plot(np.log(list_agg[:-1]),np.log(list_q[1:]-list_q[:-1]),'.')
	path = 'figures/deg_inst/'+name+'/'
	if not os.path.isdir(path):
		os.mkdir(path)
	plt.savefig(path+'degree_flow.png')
	#plt.show()

def Plot_interduration(name,agg=1):
	motifs = Motifs_tp(name)
	motifs.Get_TN(agg)
	histo = motifs.Interduration_histo(motifs.Edge_event_train())
	fig,ax = plt.subplots(constrained_layout=True)
	fontsize = 15
	ax.set_xlabel(r'$\log_{10}(\Delta\tau)$',fontsize=fontsize)
	ax.set_ylabel(r'$\log_{10}(P)$',fontsize=fontsize)
	ax.plot(*Raw_to_binned(histo),'.')
	plt.show()

#take realizations for d_t+1/d_t-1 and sample the resulting asymptotic degree distribution
#the goal is to check whether it is a geometric law
def Draw_degree_from_ratio(values):
	pass

#print the map representations of NCTN in decreasing frequency
def Print_map_NCTN(name,agg,depth):
	motifs = Motifs_tp(name)
	motifs.Get_TN(agg)
	dic_NCTN = motifs.Get_dic_NCTN(depth)
	list_seq = sorted(dic_NCTN.keys(),key=lambda seq:dic_NCTN[seq],reverse=True)
	list_map = []; profiles = All_NCTN_profiles(depth)
	print(profiles)
	for i in range(20):
		print(String_to_map_NCTN(depth,profiles,list_seq[i]))

def Check_is_dir(path):
	list_dir = path.split('/')
	direct = '.'
	for name in list_dir:
		direct +='/'+name
		if not os.path.isdir(direct):
			os.mkdir(direct)

#check there is no missing NCTN wrt weak ind hyp
def Check_missing_NCTN_weak_ind_hyp(name,agg,depth):
	motifs = Motifs_tp(name)
	motifs.Get_TN(agg)
	dic_NCTN = motifs.Get_dic_NCTN(depth)
	list_seq = sorted(dic_NCTN.keys(),key=lambda seq:dic_NCTN[seq],reverse=True)
	min_proba = dic_NCTN[list_seq[-1]]/sum(dic_NCTN.values())
	print('min_proba:',min_proba)
	dic_ind_NCTN = motifs.Get_dic_NCTN_seq_weak_ind_hyp(dic_NCTN,depth,min_proba)
	th_seq = sorted(dic_ind_NCTN.keys(),key=lambda seq:dic_ind_NCTN[seq],reverse=True)
	for i in range(30):
		if th_seq[i] not in dic_NCTN:
			print(i,th_seq[i])
	#plot the number of NCTN that are missing in name and have a proba>=1-p in th_seq vs p
	nb_missing = 0; Y = []; X = []; current_proba = 1
	for seq,proba in sorted(dic_ind_NCTN.items(),key=lambda el:el[1],reverse=True):
		if seq not in dic_NCTN:
			nb_missing += 1
		if proba!=current_proba:
			X.append(1-proba); Y.append(nb_missing)
			current_proba = proba
	fontsize = 14; X = np.asarray(X); Y = np.asarray(Y)
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_xlabel(r'$p$',fontsize=fontsize)
	ax.set_ylabel(r'nb of missing NCTN of proba$\geq1-p$',fontsize=fontsize)
	ax.plot(X,Y,'.')
	savepath = 'figures/NCTN/ind_hyp/'+name+'/Check_weak_ind_hyp_NCTN/'
	Check_is_dir(savepath)
	plt.savefig(savepath+'missing_depth'+str(depth)+'agg'+str(agg)+'.png')
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_xlabel(r'$p$',fontsize=fontsize)
	ax.set_ylabel(r'nb of missing NCTN of proba$\geq1-p$'+'\n('+r'$\log_{10}$)',fontsize=fontsize)
	ax.plot(X,np.log10(Y+1),'.')
	plt.savefig(savepath+'missing_depth'+str(depth)+'agg'+str(agg)+'zoom.png')
	plt.show()

#unweighted aggregated degree/weighted aggregated degree vs aggregation level
def Draw_agg_degree_ratio(name,list_agg,savefig):
	motifs = Motifs_tp(name)
	Y = []
	for agg in list_agg:
		print(agg)
		motifs.Get_weighted_TN(agg)
		y = 0; num = 0
		for graph in motifs.TN:
			for i in graph.nodes:
				y += graph.degree(i,weight=None)/graph.degree(i,weight='weight')
				num += 1
		Y.append(y/num)
	#plot and save the figure
	savepath = 'figures/deg_inst/'+name+'/'
	Check_is_dir(savepath)
	xlabel = 'aggregation level'
	ylabel = 'ratio between unweighted and\nweighted aggregated degrees'
	#plot the figures X,Y ; X,log(Y) ; and log(X),log(Y)
	X = np.asarray(list_agg); Y = np.asarray(Y)
	for (add_xlabel,add_ylabel),(func_x,func_y),add_name in zip([('',''),('','\n'+r'($\log_{10}$)'),('\n'+r'($\log_{10}$)','\n'+r'($\log_{10}$)')],[(lambda x:x,lambda x:x),(lambda x:x,lambda x:np.log10(x)),(lambda x:np.log10(x),lambda x:np.log10(x))],['','_lin_log','_log_log']):
		fig,ax = Setup_Plot(xlabel+add_xlabel,ylabel+add_ylabel,fontsize=14)
		ax.plot(func_x(X),func_y(Y),'.')
		plt.savefig(savepath+'Draw_agg_degree_ratio_'+savefig+add_name+'.png')
	plt.close()

#ECTN and NCTN similarity matrices btw data sets
def Draw_simat_motifs(list_names,agg,depth):
	nb_name = len(list_names); ECTN_data = []; NCTN_data = []; ECTN_data_trans = []
	for name in list_names:
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(agg)
		ECTN_data.append(motifs.Get_dic_ECTN(depth))
		ECTN_data_trans.append(motifs.Get_dic_ECTN(depth,transverse=True))
		NCTN_data.append(motifs.Get_dic_NCTN(depth))
	print('simat')
	ECTN_simat = np.eye(nb_name)
	for i in range(nb_name):
		for j in range(i+1,nb_name):
			ECTN_simat[i,j] = Cosim(ECTN_data[i],ECTN_data[j])
			ECTN_simat[j,i] = ECTN_simat[i,j]
	ECTN_simat_trans = np.eye(nb_name)
	for i in range(nb_name):
		for j in range(i+1,nb_name):
			ECTN_simat_trans[i,j] = Cosim(ECTN_data_trans[i],ECTN_data_trans[j])
			ECTN_simat_trans[j,i] = ECTN_simat_trans[i,j]
	NCTN_simat = np.eye(nb_name)
	for i in range(nb_name):
		for j in range(i+1,nb_name):
			NCTN_simat[i,j] = Cosim(NCTN_data[i],NCTN_data[j])
			NCTN_simat[j,i] = NCTN_simat[i,j]
	#save the simat
	fontsize = 14
	if not os.path.isdir('figures/simat/codata'):
		os.mkdir('figures/simat/codata')
	savepath_txt = 'figures/simat/codata/Draw_simat_motifs_agg'+str(agg)+'depth'+str(depth)
	savepath_png = 'figures/simat/Draw_simat_motifs_agg'+str(agg)+'depth'+str(depth)
	#plot the simat and save the figure
	Draw_simat(ECTN_simat,list_names,fontsize,savepath_png+'ECTN.png')
	np.savetxt(savepath_txt+'ECTN.txt',ECTN_simat)
	Draw_simat(ECTN_simat_trans,list_names,fontsize,savepath_png+'ECTN_trans.png')
	np.savetxt(savepath_txt+'ECTN_trans.txt',ECTN_simat_trans)
	Draw_simat(NCTN_simat,list_names,fontsize,savepath_png+'NCTN.png')
	np.savetxt(savepath_txt+'NCTN.txt',NCTN_simat)
	#save the ratio btw simat
	Draw_simat(ECTN_simat/NCTN_simat,list_names,fontsize,savepath_png+'EN_ratio.png')
	np.savetxt(savepath_txt+'EN_ratio.txt',ECTN_simat/NCTN_simat)

#probe the TO (time ordering) symmetry in xp, ADM and pedestrian models
#e.g. at depth 3, we compute the ratios N(101)/N(110) and N(100)/N(010)
def Compute_ratios_NCTN_TO(agg,depth):
	if depth==3:
		list_ratios = [('101','011'),('100','010')]
	elif depth==4:
		list_ratios = [('1011','0111'),('1010','1100'),('0110','1001'),('0110','1100')]
	else:
		raise ValueError('sorry, depth should be either 3 or 4^^')
	savepath = 'figures/time_ordering/'
	list_names = Get_tot_TN_not_randomized() + ['ADM17utah','ADM17conf16','ADM19conf16']
	data_to_save = [['a','a']+list_names]
	for el in list_ratios:
		data_to_save.append(list(el))
	for name in list_names:
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(agg)
		#compute histo of edge activity profiles
		histo_prof = motifs.Edge_act_prof_histo(depth)
		#compute the desired ratios
		for k,el in enumerate(list_ratios):
			data_to_save[k+1].append(histo_prof[el[0]]/histo_prof[el[1]])
	np.savetxt(savepath+'codata/prof_ratios_depth'+str(depth)+'agg'+str(agg)+'.txt',np.asarray(data_to_save,dtype=str),fmt='%s')

#probe the TR (time reversal) symmetry in xp, ADM and pedestrian models at the level of ECTN satellite
#profiles by computing the ratios N(100)/N(001), N(110)/N(011), N(300)/N(003), N(130)/N(031),
#N(113)/N(311), N(330)/N(033), N(331)/N(133)
def Compute_ratios_ECTN_TR(agg,transverse=False):
	depth = 3
	list_ratios = [('100','001'),('110','011'),('300','003'),('130','031'),('113','311'),('330','033')]
	list_ratios += [('331','133')]
	savepath = 'figures/time_reversal/codata/'
	Check_is_dir(savepath)
	list_names = Get_tot_TN_not_randomized() + ['ADM17utah','ADM17conf16','ADM19conf16']
	data_to_save = [['a','a']+list_names]
	for el in list_ratios:
		data_to_save.append(list(el))
	for name in list_names:
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(agg)
		#compute histo of ECTN satellite profiles
		histo_prof = motifs.ECTN_sat_prof_histo(depth,transverse=transverse)
		#compute the desired ratios
		for k,el in enumerate(list_ratios):
			if histo_prof[el[1]]>0:
				data_to_save[k+1].append(histo_prof[el[0]]/histo_prof[el[1]])
			else:
				data_to_save[k+1].append(-1)
	suffix = 'ECTN_depth'+str(depth)+'agg'+str(agg)+'.txt'
	if transverse:
		suffix = 'transverse'+suffix
	np.savetxt(savepath+'ratios_'+suffix,np.asarray(data_to_save,dtype=str),fmt='%s')

#compute the degree of symmetry under TR in xp, ADM and pedestrian models at the level of ECTN motifs
#to do this, we restrict to the ECTN m such that m<TR(m) and return sum_{m}(dic_ECTN[TR(m)]-dic_ECTN[m])
def Compute_sym_ECTN_TR(agg,transverse=False):
	depth = 3
	savepath = 'figures/time_reversal/codata/'
	Check_is_dir(savepath)
	list_names = Get_tot_TN_not_randomized() + ['ADM17utah','ADM17conf16','ADM19conf16']
	data_to_save = [list_names,[]]
	for name in list_names:
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(agg)
		#compute histo of ECTN satellite profiles
		dic_ECTN = motifs.Get_dic_ECTN(depth,transverse=transverse)
		#compute the desired asymmetry under Time Reversal
		data_to_save[1].append(Asym_TR(dic_ECTN,depth))
	suffix = 'ECTN_depth'+str(depth)+'agg'+str(agg)+'.txt'
	if transverse:
		suffix = 'transverse'+suffix
	np.savetxt(savepath+'asym_'+suffix,np.asarray(data_to_save,dtype=str),fmt='%s')

#one figure, showing TN and their associated ECTN satellite profiles ratios for depth 3:
#plot each ratio vs TN on the same figure (data are from Compute_ratios_ECTN_TR)
def Plot_ratios_ECTN_TR(agg,transverse=False):
	depth = 3; savepath = 'figures/time_reversal/'; fontsize = 14
	suffix = 'ECTN_depth'+str(depth)+'agg'+str(agg)+'.txt'
	if transverse:
		suffix = 'transverse'+suffix
	ratio_tab = np.loadtxt(savepath+'codata/ratios_'+suffix,dtype=str)
	#curves figure (ratios vs data sets)
	list_labels = ['N('+ratio_tab[k,0]+')/N('+ratio_tab[k,1]+')' for k in range(1,np.size(ratio_tab,0))]
	xlabel = 'temporal network'; ylabel = 'profile ratio'
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	for k in range(np.size(ratio_tab,0)-1):
		marker = LIST_MARKER[k%len(LIST_MARKER)]; color = LIST_COLOR[k%len(LIST_COLOR)]
		ax.plot(np.float64(ratio_tab[k+1,2:]),marker,color=color,label=list_labels[k])
	ax.legend(fontsize=fontsize-2,ncol=3,bbox_to_anchor=(0.5,1.23),loc="upper center")
	#TN appear in this order: xp from indices 0 to 14, pedestrian from 15 to 21, ADM and EW from 22 to end
	nb_TN = np.size(ratio_tab,1)-2
	ax.plot([0,nb_TN-1],[1]*2,'--',color='k')
	ax.axvspan(0,14.5,facecolor='gray')
	ax.axvspan(21.5,nb_TN-1,facecolor='gray')
	ax.set_xticks(range(nb_TN))
	tick_labels = ['']*nb_TN
	tick_labels[7] = 'empirical\ndata'
	tick_labels[17] = 'pedestrian\nmodels'
	tick_labels[21+(nb_TN-22)//2] = 'ADM and EW\nmodels'
	ax.set_xticklabels(tick_labels,fontsize=fontsize-2)
	suffix = 'ECTN_curve_depth'+str(depth)+'agg'+str(agg)+'.png'
	if transverse:
		suffix = 'transverse'+suffix
	plt.savefig(savepath+'ratios_'+suffix)
	plt.close()

#one figure, showing TN and their associated ECTN asymmetry coefficient under TR for depth 3
#(data are from Compute_sym_ECTN_TR)
def Plot_sym_ECTN_TR(agg,transverse=False):
	depth = 3; savepath = 'figures/time_reversal/'; fontsize = 14
	suffix = 'ECTN_depth'+str(depth)+'agg'+str(agg)+'.txt'
	if transverse:
		suffix = 'transverse'+suffix
	ratio_tab = np.loadtxt(savepath+'codata/asym_'+suffix,dtype=str)
	xlabel = 'temporal network'; ylabel = 'ECTN asym coefficient under TR'
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	ax.plot(np.float64(ratio_tab[1,:]),'.')
	#TN appear in this order: xp from indices 0 to 14, pedestrian from 15 to 21, ADM and EW from 22 to end
	nb_TN = np.size(ratio_tab,1)
	ax.axvspan(0,14.5,facecolor='gray')
	ax.axvspan(21.5,nb_TN-1,facecolor='gray')
	ax.set_xticks(range(nb_TN))
	tick_labels = ['']*nb_TN
	tick_labels[7] = 'empirical\ndata'
	tick_labels[17] = 'pedestrian\nmodels'
	tick_labels[21+(nb_TN-22)//2] = 'ADM and EW\nmodels'
	ax.set_xticklabels(tick_labels,fontsize=fontsize-2)
	suffix = 'ECTN_curve_depth'+str(depth)+'agg'+str(agg)+'.png'
	if transverse:
		suffix = 'transverse'+suffix
	plt.savefig(savepath+'asym_'+suffix)
	plt.close()

#two figures, showing TN and their associated profiles ratios for depth 3:
# - place TN as dots in the plane defined by the two profile ratios
# - plot each ratio vs TN on the same figure
def Plot_profiles_ratios_depth3(agg):
	depth = 3; savepath = 'figures/time_ordering/'
	ratio_tab = np.loadtxt(savepath+'codata/prof_ratios_depth'+str(depth)+'agg'+str(agg)+'.txt',dtype=str)
	X = np.float64(ratio_tab[1,2:])
	Y = np.float64(ratio_tab[2,2:])
	#ratios plane figure
	xlabel = 'N('+ratio_tab[1,0]+')/N('+ratio_tab[1,1]+')'
	ylabel = 'N('+ratio_tab[2,0]+')/N('+ratio_tab[2,1]+')'
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=14)
	ax.set_facecolor('gray')
	ax.plot(X,Y,'.',markersize=10)
	for x,y,name in zip(X,Y,ratio_tab[0,2:]):
		ax.annotate(name,(x,y))
	plt.savefig(savepath+'prof_ratios_plane_depth'+str(depth)+'agg'+str(agg)+'.png')
	plt.close()
	#curves figure (ratios vs data sets)
	fontsize = 14
	list_labels = ['N('+ratio_tab[k,0]+')/N('+ratio_tab[k,1]+')' for k in range(1,np.size(ratio_tab,0))]
	xlabel = 'temporal network'; ylabel = 'profile ratio'
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	for k in range(np.size(ratio_tab,0)-1):
		marker = LIST_MARKER[k%len(LIST_MARKER)]; color = LIST_COLOR[k%len(LIST_COLOR)]
		ax.plot(np.float64(ratio_tab[k+1,2:]),marker,color=color,label=list_labels[k])
	ax.legend(fontsize=fontsize)
	#TN appear in this order: xp from indices 0 to 14, pedestrian from 15 to 21, ADM and EW from 22 to end
	nb_TN = np.size(ratio_tab,1)-2
	ax.plot([0,nb_TN-1],[1]*2,'--',color='k')
	ax.axvspan(0,14.5,facecolor='gray')
	ax.axvspan(21.5,nb_TN-1,facecolor='gray')
	ax.set_xticks(range(nb_TN))
	tick_labels = ['']*nb_TN
	tick_labels[7] = 'empirical\ndata'
	tick_labels[17] = 'pedestrian\nmodels'
	tick_labels[21+(nb_TN-22)//2] = 'ADM and EW\nmodels'
	ax.set_xticklabels(tick_labels,fontsize=fontsize-2)
	plt.savefig(savepath+'prof_ratios_curve_depth'+str(depth)+'agg'+str(agg)+'.png')
	plt.close()

#one figure, showing TN and their associated profiles ratios for depth 4:
# - plot each ratio vs TN on the same figure
def Plot_profiles_ratios_depth4(agg):
	depth = 4; savepath = 'figures/time_ordering/'; fontsize = 14
	ratio_tab = np.loadtxt(savepath+'codata/prof_ratios_depth'+str(depth)+'agg'+str(agg)+'.txt',dtype=str)
	#curves figure (ratios vs data sets)
	list_labels = ['N('+ratio_tab[k,0]+')/N('+ratio_tab[k,1]+')' for k in range(1,np.size(ratio_tab,0))]
	xlabel = 'temporal network'; ylabel = 'profile ratio'
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	for k in range(np.size(ratio_tab,0)-1):
		marker = LIST_MARKER[k%len(LIST_MARKER)]; color = LIST_COLOR[k%len(LIST_COLOR)]
		ax.plot(np.float64(ratio_tab[k+1,2:]),marker,color=color,label=list_labels[k])
	ax.legend(fontsize=fontsize)
	#TN appear in this order: xp from indices 0 to 14, pedestrian from 15 to 21, ADM and EW from 22 to end
	nb_TN = np.size(ratio_tab,1)-2
	ax.plot([0,nb_TN-1],[1]*2,'--',color='k')
	ax.axvspan(0,14.5,facecolor='gray')
	ax.axvspan(21.5,nb_TN-1,facecolor='gray')
	ax.set_xticks(range(nb_TN))
	tick_labels = ['']*nb_TN
	tick_labels[7] = 'empirical\ndata'
	tick_labels[17] = 'pedestrian\nmodels'
	tick_labels[21+(nb_TN-22)//2] = 'ADM and EW\nmodels'
	ax.set_xticklabels(tick_labels,fontsize=fontsize-2)
	plt.savefig(savepath+'prof_ratios_curve_depth'+str(depth)+'agg'+str(agg)+'.png')
	plt.close()

#consider a sequence of TN and compute the ECTN similarity along the sequence
#then save data in figures/ECTN/sim/codata/ folder
#this computation is used in particular for the particular sequences:
# - increasing clustering, fixed NCTN
# - increasing degree assortativity, fixed NCTN
#list_names[ref] is the TN taken as reference for the ECTN similarity
def Compute_motifs_sim_along_seq(list_names,agg,depth,namepath,ref=0,cc_fixed=False,choice_motifs='ECTN'):
	prefix = 'figures/'+choice_motifs+'/sim/codata/'
	Check_is_dir(prefix)
	suffix = namepath+'_ref'+str(ref)+'depth'+str(depth)+'agg'+str(agg)+'.txt'
	if cc_fixed:
		suffix = 'cc_fixed'+suffix
	motifs = Motifs_tp(list_names[ref%len(list_names)])
	motifs.Get_TN(agg)
	if choice_motifs=='ECTN':
		ref_ECTN = motifs.Get_dic_ECTN(depth)
	else:
		ref_ECTN = motifs.Get_dic_NCTN(depth)
	Y = []; list_cc = []
	for name in list_names:
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(agg)
		list_cc.append(np.mean([nx.average_clustering(G) for G in motifs.TN]))
		if choice_motifs=='ECTN':
			Y.append(Cosim(motifs.Get_dic_ECTN(depth),ref_ECTN))
		else:
			Y.append(Cosim(motifs.Get_dic_NCTN(depth),ref_ECTN))
	np.savetxt(prefix+suffix,np.array([list_cc,Y]))

#dic_names[key] is a set of TN, name_ref is a TN reference, we compute the ECTN sim btw the ref
#and each TN in dic_names[key], for each key, then save the data in tab, where
#tab[i,j+1] = ECTN sim btw the jth TN with property corresponding to key i, and name_ref
#tab[i,0] = property corresponding to key i
def Compute_ECTN_sim_along_dic(dic_names,agg,depth,name_ref,namepath):
	prefix = 'figures/ECTN/sim/codata/'
	Check_is_dir(prefix)
	suffix = namepath+'depth'+str(depth)+'agg'+str(agg)+'.txt'
	motifs = Motifs_tp(name_ref)
	motifs.Get_TN(agg)
	ref_ECTN = motifs.Get_dic_ECTN(depth)
	tab = np.zeros((len(dic_names),11))
	for i,(key,val) in enumerate(dic_names.items()):
		tab[i,0] = float(key)
		for j,name in enumerate(val):
			print(name)
			motifs = Motifs_tp(name)
			motifs.Get_TN(agg)
			tab[i,j+1] = Cosim(motifs.Get_dic_ECTN(depth),ref_ECTN)
	np.savetxt(prefix+suffix,tab)

#dic_names[key] is a set of TN, we compute the ECTN sim btw the TN in
#dic_names[key], for each key, then save the data in tab, where
#tab[i,1] = average ECTN sim btw the TN with property corresponding to key i
#tab[i,0] = property corresponding to key i
def Compute_intra_ECTN_sim_along_dic(dic_names,agg,depth,namepath):
	prefix = 'figures/ECTN/sim/codata/'
	Check_is_dir(prefix)
	suffix = namepath+'intra_depth'+str(depth)+'agg'+str(agg)+'.txt'
	tab = np.zeros((len(dic_names),2))
	for i,(key,val) in enumerate(dic_names.items()):
		print(key)
		tab[i,0] = float(key)
		list_val = []
		list_ref_ECTN = []
		for name in val:
			print(name)
			motifs = Motifs_tp(name)
			motifs.Get_TN(agg)
			list_ref_ECTN.append(motifs.Get_dic_ECTN(depth))
		for j,vec1 in enumerate(list_ref_ECTN[:-1]):
			for vec2 in list_ref_ECTN[j+1:]:
				list_val.append(Cosim(vec1,vec2))
		tab[i,1] = np.mean(list_val)
	np.savetxt(prefix+suffix,tab)

#return a sequence of paths containing TN with various cc but same NCTN of depth 3
def Get_seq_cc_Giulia(cc_fixed=False):
	Giulia_path = os.path.join(PROJECT_ROOT,'data/primary_school/')
	#list_names contains paths locating the TN
	if cc_fixed:
		datapath = os.path.join(Giulia_path,'Primary_School_mu_1.000_pcl_0.4/')
		return [os.path.join(datapath,el) for el in os.listdir(datapath)]
	else:
		list_names = []
		for dir_ in os.listdir(Giulia_path):
			datapath = os.path.join(Giulia_path,dir_)
			list_names.append(os.path.join(datapath,rd.choice(os.listdir(datapath))))
	return list_names

#return a dictionary: dic_names[key] = set of paths to TN with the property key in common
#only the folders with keyword in their names are considered
def Get_seq_Giulia(namepath,keyword):
	Giulia_path = os.path.join(PROJECT_ROOT,'data/'+namepath+'/')
	#dic_names contains paths locating the TN
	dic_names = {}
	for dir_ in os.listdir(Giulia_path):
		if keyword in dir_:
			key = dir_.split('_')[-1]
			datapath = os.path.join(Giulia_path,dir_)
			dic_names[key] = {os.path.join(datapath,file) for file in os.listdir(datapath)}
	return dic_names

#plot the ECTN sim vs cc, where data have been computed by Compute_ECTN_sim_along_seq
def Draw_motifs_sim_vs_cc(agg,depth,namepath,ref=0,cc_fixed=False,choice_motifs='ECTN'):
	fontsize = 14; xlabel = 'average clustering coefficient'; ylabel = choice_motifs+' similarity'
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	prefix = 'figures/'+choice_motifs+'/sim/'
	suffix = namepath+'_ref'+str(ref)+'depth'+str(depth)+'agg'+str(agg)
	if cc_fixed:
		suffix = 'cc_fixed'+suffix
	tab = np.loadtxt(prefix+'codata/'+suffix+'.txt')
	ax.plot(tab[0,:],tab[1,:],'.')
	plt.savefig(prefix+suffix+'.png')

#NCTN and ECTN similarity matrices including models at various aggregation levels
def ECTN_NCTN_simat_comparison():
	list_names = ['conf16','utah','baboon','highschool3','work2','ABPpi4','ADM9conf16','ADM18conf16','min_ADM2','min_EW3']
	list_agg = range(1,21); depth = 3
	for agg in list_agg:
		print(agg)
		Draw_simat_motifs(list_names,agg,depth)

#like ECTN_NCTN_simat_comparison but with a different similarity measure:
#the motifs are separated in two families (rare and frequent) and the sim is the product of the sim btw
#the two families
def ECTN_NCTN_simat_comparison_stronger():
	list_names = ['conf16','utah','baboon','highschool3','work2','ABPpi4','ADM9conf16','ADM18conf16','min_ADM2','min_EW3']
	agg = 1; depth = 3
	nb_name = len(list_names); NCTN_data = []; ECTN_data = []
	for name in list_names:
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(agg)
		ECTN_data.append(motifs.Get_dic_ECTN(depth,trunc=None))
		NCTN_data.append(motifs.Get_dic_NCTN(depth))
	print('simat')
	ECTN_simat = np.zeros((nb_name,nb_name))
	for i in range(nb_name):
		for j in range(i+1,nb_name):
			ECTN_simat[i,j] = Cosim_triple(ECTN_data[i],ECTN_data[j],nb_parts=2)
			ECTN_simat[j,i] = ECTN_simat[i,j]
	NCTN_simat = np.zeros((nb_name,nb_name))
	for i in range(nb_name):
		for j in range(i+1,nb_name):
			NCTN_simat[i,j] = Cosim_triple(NCTN_data[i],NCTN_data[j],nb_parts=2)
			NCTN_simat[j,i] = NCTN_simat[i,j]
	#save the simat
	fontsize = 14
	savepath = 'figures/simat/Draw_simat_motifs_newsimagg'+str(agg)+'depth'+str(depth)
	#plot the simat and save the figure
	Draw_simat(ECTN_simat,list_names,fontsize,savepath+'ECTN.png')
	Draw_simat(NCTN_simat,list_names,fontsize,savepath+'NCTN.png')
	plt.show()

#similar as Check_weak_ind_hyp_NCTN but with ECTN:
#the goal is to find an ind hyp which is satisfied for ECTN in XP
def Check_weak_ind_hyp_ECTN(name,agg,depth,hyp_nb):
	motifs = Motifs_tp(name)
	motifs.Get_TN(agg)
	#compute the ECTN proba under the weak spatial ind hypothesis
	X,Y = motifs.Get_ind_ECTN(motifs.Get_dic_ECTN(depth),depth,hyp_nb)
	#compare the two distributions
	fontsize = 16; xlabel = 'xp_proba'; ylabel = 'th_proba'
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	ax.plot(X,Y,'.')
	ax.plot(X,X,'--')
	path = 'figures/ECTN/ind_hyp/'+name+'/'
	#check folders exist and create them if not
	Check_is_dir(path)
	#plt.savefig(path+'depth'+str(depth)+'agg'+str(agg)+'.png')
	plt.show()
	plt.close()

#complete workflow to obtain ECTN sim btw french and Giulia model version with n first time steps fixed vs n
def Complete_workflow_ECTN_sim_timestamps():
	namepath = 'networks_init_cond'; agg = 1; depth = 3
	Get_Giulia_formatted(namepath)
	dic_names = Get_seq_Giulia(namepath,'_init_')
	#we take french as reference
	Compute_ECTN_sim_along_dic(dic_names,agg,depth,'french',namepath)
	#plot the ECTN sim vs nb of time steps fixed
	fontsize = 14; xlabel = 'nb of frozen first steps'; ylabel = 'ECTN similarity with french'
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	prefix = 'figures/ECTN/sim/'
	suffix = namepath+'depth'+str(depth)+'agg'+str(agg)
	tab = np.loadtxt(prefix+'codata/'+suffix+'.txt')
	#sort the lines by increasing value of the first column
	tab = np.array(sorted(tab,key=lambda line:line[0]))
	tab_mean = np.mean(tab[:,1:],axis=1)
	tab_std = np.std(tab[:,1:],axis=1)
	ax.plot(tab[:,0],tab_mean,'.')
	ax.fill_between(tab[:,0],tab_mean-tab_std/2,tab_mean+tab_std/2,color='gray')
	plt.savefig(prefix+suffix+'.png')

#complete workflow to obtain ECTN sim vs clustering coefficient (primary school is actually french)
def Complete_workflow_ECTN_sim_cc():
	namepath = 'primary_school'
	Get_Giulia_formatted()
	list_names = Get_seq_cc_Giulia()
	agg = 1; depth = 3
	for ref in range(len(list_names)):
		print('ref',ref)
		Compute_motifs_sim_along_seq(list_names,agg,depth,namepath,ref=ref)
	#plot the ECTN sim vs cc
	for ref in range(len(list_names)):
		print('ref',ref)
		Draw_motifs_sim_vs_cc(agg,depth,namepath,ref=ref)

#complete workflow to visualize the ten most frequent NCTN or ECTN of name
def Complete_motifs_visu(name,agg,depth,choice_motifs,transverse=False):
	prefix = 'figures/'+choice_motifs+'/visu/'+name+'/'
	suffix = 'depth'+str(depth)+'agg'+str(agg)+'.png'
	Check_is_dir(prefix)
	motifs = Motifs_tp(name)
	motifs.Get_TN(agg)
	if choice_motifs=='NCTN':
		ETN.Draw_ten_freq_NCTN(motifs.Get_dic_NCTN(depth),depth,prefix+suffix)
	elif choice_motifs=='ECTN':
		if transverse:
			dic_ECTN = motifs.Get_dic_ECTN(depth)
			new_dic = {}
			for seq,nb in dic_ECTN.items():
				if '3' in seq:
					new_dic[seq] = nb
				elif '1' in seq[depth:] and '2' in seq[depth:]:
					new_dic[seq] = nb
			ETN.Draw_ten_freq_ECTN(new_dic,depth,prefix+'transverse_'+suffix)
		else:
			ETN.Draw_ten_freq_ECTN(motifs.Get_dic_ECTN(depth),depth,prefix+suffix)

#complete workflow to obtain average ECTN sim btw Giulia model versions with n first time steps fixed vs n
def Complete_workflow_ECTN_sim_timestamps_intra():
	namepath = 'networks_init_cond'; agg = 1; depth = 3
	#Get_Giulia_formatted(namepath)
	dic_names = Get_seq_Giulia(namepath,'_init_')
	Compute_intra_ECTN_sim_along_dic(dic_names,agg,depth,namepath)
	#plot the average ECTN intra sim vs nb of time steps fixed
	fontsize = 14; xlabel = 'nb of frozen first steps'
	ylabel = 'ECTN similarity between\nindependent realizations'
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	prefix = 'figures/ECTN/sim/'
	suffix = namepath+'intra_depth'+str(depth)+'agg'+str(agg)
	tab = np.loadtxt(prefix+'codata/'+suffix+'.txt')
	ax.plot(tab[:,0],tab[:,1],'.')
	plt.savefig(prefix+suffix+'.png')

#complete workflow for ECTN ratios testing TR sym
def Workflow_ECTN_ratios_TR_sym():
	agg = 1
	Compute_ratios_ECTN_TR(agg)
	Plot_ratios_ECTN_TR(agg)
	print('\ttrans begins')
	Compute_ratios_ECTN_TR(agg,transverse=True)
	Plot_ratios_ECTN_TR(agg,transverse=True)

#compute the ECTN and xp_probas for name,namepath for depths 2 and 3, then save data
def Collect_ECTN(agg,name,namepath):
	prefix = 'figures/ECTN/ind_hyp/'
	suffix = '/codata/Collect_ECTN_agg'+str(agg)+'depth'
	motifs = Motifs_tp(name,where=namepath)
	motifs.Get_TN(agg)
	Check_is_dir(prefix+name+'/codata')
	for depth in [2,3]:
		print('\tdepth: '+str(depth))
		datapath = prefix+name+suffix+str(depth)
		list_seq,xp_proba = zip(*motifs.Get_dic_ECTN(depth).items())
		xp_norm = np.sum(xp_proba)
		np.savetxt(datapath+'string.txt',list_seq,fmt='%s')
		np.savetxt(datapath+'xp_proba.txt',[x/xp_norm for x in xp_proba])

#compute the NCTN and xp_probas for name,namepath for depths 2 and 3, then save data
def Collect_NCTN(agg,name,namepath):
	prefix = 'figures/NCTN/ind_hyp/'
	suffix = '/codata/Collect_NCTN_agg'+str(agg)+'depth'
	mtp = Motifs_tp(name,where=namepath)
	mtp.Get_TN(agg)
	Check_is_dir(prefix+name+'/codata')
	for depth in [2,3]:
		print('\tdepth: '+str(depth))
		datapath = prefix+name+suffix+str(depth)
		list_seq,xp_proba = zip(*mtp.Get_dic_NCTN(depth).items())
		xp_norm = np.sum(xp_proba)
		np.savetxt(datapath+'string.txt',list_seq,fmt='%s')
		np.savetxt(datapath+'xp_proba.txt',[x/xp_norm for x in xp_proba])

def Load_list_seq_restrict(datapath,restrict):
	path_exist = True
	if restrict==0:
		path = datapath+'string.txt'
		if os.path.exists(path):
			return np.loadtxt(path,dtype=str)
		else:
			path_exist = False
	if restrict==20:
		path = datapath+'string_restrict.txt'
		if os.path.exists(path):
			return np.loadtxt(path,dtype=str)
		else:
			path_exist = False
	path = datapath+'string_restrict'+str(restrict)+'.txt'
	if os.path.exists(path):
		return np.loadtxt(path,dtype=str)
	path = datapath+'string.txt'
	if not os.path.exists(path):
		path_exist = False
	else:
		xp_proba = np.loadtxt(datapath+'xp_proba.txt')
		list_seq = np.loadtxt(datapath+'string.txt',dtype=str)
		xp_proba,list_seq = Restrict_seq(xp_proba,list_seq,freq_ratio=restrict)
		np.savetxt(datapath+'xp_proba_restrict'+str(restrict)+'.txt',xp_proba)
		np.savetxt(datapath+'string_restrict'+str(restrict)+'.txt',list_seq,fmt='%s')
		return np.asarray(list_seq)
	if not path_exist:
		return []

def Load_xp_th_proba_restrict(datapath,restrict,hyp_nb):
	#load xp proba
	if restrict==0:
		list_path = ['_restrict0','']
	elif restrict==20:
		list_path = ['_restrict20','_restrict']
	else:
		list_path = ['_restrict'+str(restrict)]
	keep = True; ind = 0
	while keep:
		path = datapath+'xp_proba'+list_path[ind]+'.txt'
		if os.path.exists(path):
			xp_proba = np.loadtxt(path)
			keep = False
		ind += 1
	#load th proba
	list_path = ['_restrict'+str(restrict)+'hyp'+str(hyp_nb)]
	if restrict==0:
		list_path.append(str(hyp_nb))
	elif restrict==20:
		list_path.append('_restrict'+str(hyp_nb))
	keep = True; ind = 0
	while keep:
		path = datapath+'th_proba'+list_path[ind]+'.txt'
		if os.path.exists(path):
			th_proba = np.loadtxt(path)
			keep = False
		ind += 1
		if ind==len(list_path):
			keep = False
	#if th_proba not found, generate it from restrict = 0
	if ind==len(list_path):
		#load th_proba for restrict = 0
		keep = True; ind = 0; list_path = ['_restrict0hyp'+str(hyp_nb),str(hyp_nb)]
		while keep:
			path = datapath+'th_proba'+list_path[ind]+'.txt'
			if os.path.exists(path):
				th_proba = np.loadtxt(path)
				keep = False
			ind += 1
		#load list_seq for restrict = 0
		list_seq = Load_list_seq_restrict(datapath,0)
		dic_thproba = dict(zip(list_seq,th_proba))
		#deduce th_proba for restrict
		th_proba = [dic_thproba[seq] for seq in Load_list_seq_restrict(datapath,restrict)]
		#save it for later call of the same function
		path = datapath+'th_proba'+'_restrict'+str(restrict)+'hyp'+str(hyp_nb)+'.txt'
		np.savetxt(path,th_proba)
	return xp_proba,th_proba

def Load_xp_proba_restrict(datapath,restrict):
	#load xp proba
	if restrict==0:
		list_path = ['_restrict0','']
	elif restrict==20:
		list_path = ['_restrict20','_restrict']
	else:
		list_path = ['_restrict'+str(restrict)]
	keep = True; ind = 0
	while keep:
		path = datapath+'xp_proba'+list_path[ind]+'.txt'
		if os.path.exists(path):
			xp_proba = np.loadtxt(path)
			keep = False
		ind += 1
	return xp_proba

#compute the maximum th-xp relative gap across ECTN for a TN
#for depths 2 and 3 as coordinates
def Compute_gap_TN_hyp_ind(name,namepath,agg,hyp_nb,restrict=20):
	prefix = 'figures/ECTN/ind_hyp/'
	suffix = '/codata/Collect_ECTN_agg'+str(agg)+'depth'
	res = ()
	#load xp and th probas
	for depth in [2,3]:
		print('\tdepth',depth)
		datapath = prefix+name+suffix+str(depth)
		xp_proba,th_proba = Load_xp_th_proba_restrict(datapath,restrict,hyp_nb)
		#!!!make sure probas are normalized the same way!!!
		xp_proba /= np.sum(xp_proba)
		th_proba /= np.sum(th_proba)
		#compute the th-xp gap
		tab = abs(xp_proba-th_proba)/(th_proba+xp_proba)
		#remove the missing ECTN (we used only a part of the NCTN)
		tab = tab[np.nonzero(np.atleast_1d(tab!=1))]
		gap = 1-np.max(tab)
		res += (gap,)
	return res

#return gap_dic[seq] = th-xp gap for the ECTN seq wrt the ind hyp hyp_nb
def Compute_gap(datapath,restrict,hyp_nb,logscale=True):
	xp_proba,th_proba = Load_xp_th_proba_restrict(datapath,restrict,hyp_nb)
	#!!!make sure probas are normalized the same way!!!
	xp_proba /= np.sum(xp_proba)
	th_proba /= np.sum(th_proba)
	#compute the relative gap for each ECTN
	gap = 1-abs(xp_proba-th_proba)/(th_proba+xp_proba)
	#remove the missing ECTN (we used only a part of the NCTN)
	missing_ECTN = np.nonzero(np.atleast_1d(gap))
	gap = gap[missing_ECTN]
	list_seq = Load_list_seq_restrict(datapath,restrict)[missing_ECTN]
	if logscale:
		gap = np.log10(gap)
	return dict(zip(list_seq,gap))

#return name,namepath associated to Giulia data set
def Load_Giulia():
	prev_path = PROJECT_ROOT
	for word in ['data','networks_init_cond','Primary_School_gap_20_loc_split_1h_mu_1.000_pcl_0.4_init_10','0.dat']:
		namepath = os.path.join(prev_path,word)
		prev_path = namepath
	return 'Giulia_french', namepath

#truncate xp_proba and list_ECTN to the most frequent items
def Restrict_seq(xp_proba,list_ECTN,freq_ratio=20):
	if freq_ratio==0:
		return xp_proba,list_ECTN
	probas = sorted(zip(xp_proba,list_ECTN),key=lambda el:el[0],reverse=True)
	#sort out the small frequencies to reduce noise sampling
	ind_max = 0
	while probas[ind_max][0]/probas[-1][0]>freq_ratio:
		ind_max += 1
		if ind_max==len(probas):
			break
	print('\t'+str(100*(1-ind_max/len(probas)))+' % of items eliminated')
	#return the truncated iterables
	return zip(*probas[:ind_max])

#test the MEP hypothesis: draw th_proba vs xp_proba
def Test_MEP_hypothesis(name,namepath,depth,restrict,hyp_nb):
	#name,namepath = Load_Giulia()
	agg = 1
	prefix = 'figures/ECTN/ind_hyp/'
	suffix = '/codata/Collect_ECTN_agg'+str(agg)+'depth'+str(depth)
	datapath = prefix+name+suffix
	xp_proba,th_proba = Load_xp_th_proba_restrict(datapath,restrict,hyp_nb)
	#!!!make sure probas are normalized the same way!!!
	xp_proba /= np.sum(xp_proba)
	th_proba /= np.sum(th_proba)
	#plot th_proba vs xp_proba
	fig,ax = plt.subplots(constrained_layout=True)
	ax.plot(xp_proba,th_proba,'.')
	ax.plot(xp_proba,xp_proba,'--')
	savepath = prefix+str(hyp_nb)+'hyp/Test_'+name+'_agg'+str(agg)+'restrict'+str(restrict)+'depth'+str(depth)
	plt.savefig(savepath+'.png')
	plt.close()

#remove a list of items from a list
def Remove_names_from_list(list_names,name_to_remove):
	for name in name_to_remove:
		ind = 0
		while list_names[ind][0]!=name:
			ind += 1
		list_names.remove(list_names[ind])
	return None

def Min_EW3_with_varying_activity():
	gen = False
	figpath = 'figures/Min_EW/'
	name = 'var_act' #cst_act var_act conf16
	if gen:
		N,duree,frac = 138,3000,2
		print('nb_new',frac*N*(N-1)/(2*duree))
		if name=='cst_act':
			min_ew = atn.Min_EW(N,duree,frac,shift=True,removal='edge_unif')
		elif name=='var_act':
			#nb of time steps btw two suractivations of edges
			delay = 300
			#nb of injected edges: every delay, we activate n_inj edges taken at random
			n_inj = 200
			min_ew = atn.Min_EW(N,duree,frac,shift=True,removal='edge_unif',var_act=True,delay=delay,n_inj=n_inj)
		np.savetxt(name+'.txt',min_ew.Evolve(t_min=2000),fmt='%d')

	try:
		data = np.loadtxt(name+'.txt',dtype=int)
		motifs = Motifs_tp(data)
	except:
		motifs = Motifs_tp(name)
	motifs.Get_TN(1)
	edge_event = motifs.Edge_event_train()
	#motifs = Motifs_tp('conf16'); motifs.Get_TN(1)

	#nb of active edges per time step
	fig,ax = Setup_Plot('t','nb of active edges')
	ax.plot([len(graph.edges) for graph in motifs.TN],'.')
	plt.savefig(figpath+name+'_edge_act.png')

	#edge activity duration
	fig,ax = Setup_Plot('edge activity duration (log10)','proba (log10)')
	durat_histo = Duration_histo(edge_event)
	ax.plot(*Raw_to_binned(durat_histo),'.')
	plt.savefig(figpath+name+'_edge_duration.png')

	#edge activity interduration
	fig,ax = Setup_Plot('edge activity\ninterduration (log10)','proba (log10)')
	inter_histo = Interduration_histo(edge_event)
	ax.plot(*Raw_to_binned(inter_histo),'.')
	plt.savefig(figpath+name+'_edge_interduration.png')

	#instantaneous degree
	fig,ax = Setup_Plot('instantaneous degree','proba (log10)')
	histo = motifs.Get_direct_deg_histo()
	X,Y = zip(*histo.items())
	ax.plot(X,np.log10(Y),'.')
	plt.savefig(figpath+name+'_inst_deg.png')

	#size of connected components
	fig,ax = Setup_Plot('cc size in nodes (log10)','proba (log10)')
	histo = motifs.Get_cc_size()
	ax.plot(*Raw_to_binned(histo),'.')
	plt.savefig(figpath+name+'_cc_node_size.png')
	plt.show()

def Parallel_compute_ECTN_th_proba_MEP(threshold):
	hyp_nb = 6;	agg = 1
	for name,namepath in Build_list_names():
		Compute_ECTN_th_proba(agg,name,namepath,hyp_nb,restrict=threshold)

#add other datasets of Giulia

#rewrite TN data located at old_name into formatted data located at new_name
#old_name and new_name are paths relative to PROJECT_ROOT/data/
def Format_TN_data(old_name,new_name,duration=np.inf):
	datapath = os.path.join(PROJECT_ROOT,'data/')
	net = tp.Temp_net(np.loadtxt(datapath+old_name,dtype=int))
	net.Format(duration=duration)
	np.savetxt(datapath+new_name+'.txt',net.data,fmt='%d')

#rewrite Giulia data into our format
#Giulia_path = os.path.join(PROJECT_ROOT,'data/primary_school/')
#OLD
def Get_Giulia_formatted(namepath):
	Giulia_path = os.path.join(PROJECT_ROOT,'data/'+namepath+'/')
	for dir_ in os.listdir(Giulia_path):
		print(dir_)
		for file in os.listdir(Giulia_path+dir_):
			net = tp.Temp_net(np.loadtxt(Giulia_path+dir_+'/'+file,dtype=int))
			net.Format()
			np.savetxt(Giulia_path+dir_+'/'+file,net.data,fmt='%d')

#return old_names,new_names with old_names containing the paths to the original TN of Giulia
def Giulia_path_to_raw_TN():
	old_names = []
	new_names = []
	dir1_to_name = {'clust+mod_static':'CMS','clust+mod_temp':'CMT','clust+mod_temp+memory':'CMTM','ETN':''}
	dir2_to_name = {'h_school11':'highschool1','h_school13':'highschool3','hypertext':'hypertext','pr_school':'french'}
	#chosen_dir[dir] = suffix of the folders that contain the TN data to rewrite
	#if dir is not in chosen_dir, we rewrite the content of every folder in dir
	chosen_dir = {'clust+mod_static':'_gap_20_mu_1.000_pcl_1.00'}
	base_path = os.path.join(PROJECT_ROOT,'data/')
	Giulia_path = base_path+'original_tij/reseaux_Giulia/'
	for dir1 in os.listdir(Giulia_path):
		#ensure dir1 is a folder whose content we want to rewrite
		if dir1 in dir1_to_name:
			#select the correct subfolders
			if dir1 in chosen_dir:
				list_folders = [prefix+chosen_dir[dir1] for prefix in dir2_to_name]
			else:
				list_folders = [folder.name for folder in os.scandir(Giulia_path+dir1+'/') if folder.is_dir()]
			#rewrite the subfolders content
			for folder in list_folders:
				chunks = folder.split('_')
				if chunks[0]=='h' or chunks[0]=='pr':
					dir2 = chunks[0]+'_'+chunks[1]
				else:
					dir2 = chunks[0]
				old_names.append('original_tij/reseaux_Giulia/'+dir1+'/'+folder+'/5.dat')
				new_names.append(dir2_to_name[dir2]+'_G'+dir1_to_name[dir1])
	return old_names,new_names

#rewrite Giulia data into our format
def Giulia_formatted():
	for old_name,new_name in zip(*Giulia_path_to_raw_TN()):
		print(new_name)
		Format_TN_data(old_name,new_name)

#rewrite Juliette data into our format
def Juliette_formatted():
	target_duration = 5000
	old_folder = 'juliette_tij/'
	list_old_names = ['tij_2DRW-coll-border_periodic-D.5-R.5-RD1.5-N1000-t50000',\
	'tij_abp-coll-border_periodic-v.5-noisepi_4-R.5-RD1.5-N1000-t20000',\
	'tij_ABP-noisepi4-R3-RD9-N100-t10000','tij_vicsek-coll-border_periodic-v.5-noisepi_2-R.5-RD1.5-N1000-t20000',\
	'tij-abp-coll-border_mirror-v.5-noisepi_4-R.5-RD1.5-N1000-t20000',\
	'tij-vicsek-coll-border_mirror-v.5-noisepi_2-R.5-RD1.5-N1000-t20000-counts']
	list_new_names = ['RW_periodic','ABPpi4_periodic','ABPpi4_jul','Vicsekpi2_periodic',\
	'ABPpi4_mirror','Vicsekpi2_mirror']
	for old_name,new_name in zip(list_old_names,list_new_names):
		print(new_name)
		Format_TN_data(old_folder+old_name+'.txt',new_name,duration=target_duration)

#build list_names as tuples (name,namepath)
def Build_list_names():
	list_names = [(name,None) for name in Get_tot_TN_not_randomized() + ['ADM17utah','ADM17conf16','ADM19conf16']]
	list_names.append(('min_EW3_RS.txt','min_EW3_RS.txt'))
	#add Giulia data set
	list_names.append(Load_Giulia())
	return list_names

#return Giulia TN names
def Build_list_names_Giulia():
	L1 = ['french','highschool1','highschool3','hypertext']
	L2 = ['','CMS','CMT','CMTM']
	#res = [(name,None) for name in L1[:-1]]
	res = []
	for suffix in L2:
		for prefix in L1:
			res.append((prefix+'_G'+suffix,None))
	return res

#return Juliette TN names
def Build_list_names_Juliette():
	L1 = ['ABPpi4','Vicsekpi2']
	res = [(name+'_'+suffix,None) for name in L1 for suffix in ['periodic','mirror']]
	return res + [('ABPpi4_jul',None),('RW_periodic',None)]

#return dic[name] = name of the family that contains name (conf, schools, ADM, pedestrian,
#Giulia, others)
def Build_list_names_with_family():
	dic = {}
	dic['conf'] = ['conf1'+str(k) for k in range(6,10)]
	#schools
	dic['schools'] = ['highschool'+str(k) for k in range(1,4)] + ['french','utah']
	#other xp
	dic['other_xp'] = ['work1','work2','hospital']
	#model_graph
	dic['models_1'] = ['ADM'+str(k)+'conf16' for k in (9,17,18,19)] + ['ADM17utah']
	dic['models_1'] += ['min_ADM'+str(k) for k in range(1,3)] + ['min_EW'+str(k) for k in range(1,4)]
	#add Giulia data set
	dic['models_2'] = list(tuple(zip(*Build_list_names_Giulia()))[0])
	#pedestrian models
	dic['model_pedestrian'] = list(tuple(zip(*Build_list_names_Juliette()))[0])
	return dic

def Load_dic_ECTN(datapath,restrict):
	list_seq = Load_list_seq_restrict(datapath,restrict)
	xp_proba = Load_xp_proba_restrict(datapath,restrict)
	xp_proba /= np.sum(xp_proba)
	return dict(zip(list_seq,xp_proba))

class Motifs_analyze:
	"""docstring for Motifs_analyze"""
	def __init__(self,folder,agg=1,logscale=True,hyp_nb=1,restrict=0):
		self.agg = agg
		self.hyp_nb = hyp_nb
		self.restrict = restrict
		self.folder = folder
		self.logscale = logscale
		self.prefix = 'figures/ECTN/ind_hyp/'
		self.suffix = '/codata/Collect_ECTN_agg'+str(agg)+'depth'

	#compute A_{1}/A_{2} for ECTN of depths 2 and 3 (MEP property)
	def MEP_ratio(self,name,namepath):
		res = ()
		for depth in [2,3]:
			print('\t'+str(depth))
			datapath = self.prefix+name+self.suffix+str(depth)
			#load all the ECTN motifs
			dic_ECTN = Load_dic_ECTN(datapath,self.restrict)
			tab = Compute_MEP_ratio(dic_ECTN,depth)
			if tab is None:
				return None
			n,bins = np.histogram(tab,density=True,bins='auto')
			th_index = 0
			while n[th_index+1]<=n[th_index]:
				th_index += 1
			th_index += 1
			n = n[:th_index]
			bins = bins[:th_index+1]
			A_1 = np.sum(n*(bins[1:]-bins[:-1]))
			res += (A_1/(1-A_1),)
		return res

	#plot the histogram of the MEP ratio
	def Plot_MEP_ratio_histo(self,name,namepath,depth,filtered=True):
		datapath = self.prefix+name+self.suffix+str(depth)
		#load all the ECTN motifs
		dic_ECTN = Load_dic_ECTN(datapath,self.restrict)
		tab = Compute_MEP_ratio(dic_ECTN,depth,filtered=filtered)
		if tab is None:
			return None
		fig,ax = Setup_Plot(r"$e(m,m')$",'probability density',fontsize=16)
		ax.hist(tab,density=True,bins='auto',histtype='step',linewidth=4)

	#compute the ECTN MEP ratio for each TN and place each TN on a plane with
	#this gap for depths 2 and 3 as coordinates
	def Compare_TN_MEP_ratio(self,list_names):
		folder = self.folder+'MEP_ratio/'
		#dic_coord[name] = (x,y) with x(y) = goodness of fit of hyp_nb for ECTN of depth 2(3)
		dic_coord = {}
		failures = []
		for name,namepath in list_names:
			print(name)
			val = self.MEP_ratio(name,namepath)
			if val is None:
				failures.append(name)
			else:
				dic_coord[name] = val
		#place the TN on a plane
		xlabel = 'MEP ratio for depth 2'
		ylabel = 'MEP ratio for depth 3'
		fig,ax = Setup_Plot(xlabel,ylabel,fontsize=14)
		ax.set_facecolor('gray')
		labels,data = zip(*dic_coord.items())
		X,Y = zip(*data)
		ax.plot(X,Y,'.',markersize=10)
		for x,y,name in zip(X,Y,labels):
			ax.annotate(name,(x,y))
		Check_is_dir(self.prefix+'6hyp/'+folder)
		savepath = self.prefix+'6hyp/'+folder+'Compare_TN_MEP_ratio_agg'+str(self.agg)
		plt.savefig(savepath+'.png')
		plt.close()
		print('those TN have not enough ECTN:')
		for name in failures:
			print(failures)

	#compute the maximum th-xp relative gap across ECTN for each TN and place each TN on a plane with
	#this gap for depths 2 and 3 as coordinates
	#if custom_path, the figure will be saved at custom_path
	#if name_to_cs, the TN names will not be displayed but they will be grouped by color and shape
	#according to name_to_cs
	def Compare_TN_hyp_ind(self,list_names,custom_path='',name_to_cs={},marker_to_label={}):
		folder = self.folder
		#dic_coord[name] = (x,y) with x(y) = goodness of fit of hyp_nb for ECTN of depth 2(3)
		dic_coord = {}
		for name,namepath in list_names:
			print(name)
			dic_coord[name] = ()
			#load xp and th probas
			for depth in [2,3]:
				print('\tdepth',depth)
				datapath = self.prefix+name+self.suffix+str(depth)
				xp_proba,th_proba = Load_xp_th_proba_restrict(datapath,self.restrict,self.hyp_nb)
				#!!!make sure probas are normalized the same way!!!
				xp_proba /= np.sum(xp_proba)
				th_proba /= np.sum(th_proba)
				#compute the th-xp gap
				tab = abs(xp_proba-th_proba)/(th_proba+xp_proba)
				#remove the missing ECTN (we used only a part of the NCTN)
				tab = tab[np.nonzero(np.atleast_1d(tab!=1))]
				print('nb of valid distinct motifs:',len(tab))
				gap = 1-np.max(tab)
				if self.logscale:
					gap = np.log10(gap)
				dic_coord[name] += (gap,)
		#place the TN on a plane
		xlabel = 'compatibility with hyp '+str(self.hyp_nb)+' for depth 2'
		ylabel = 'compatibility with hyp '+str(self.hyp_nb)+' for depth 3'
		fig,ax = Setup_Plot(xlabel,ylabel,fontsize=14)
		ax.set_facecolor('gray')
		if name_to_cs:
			#group data by marker
			marker_to_name = {}
			for name,(color,marker) in name_to_cs.items():
				if marker in marker_to_name:
					marker_to_name[marker].append((name,color))
				else:
					marker_to_name[marker] = [(name,color)]
			for marker,val in marker_to_name.items():
				TN_names,list_color = zip(*val)
				X,Y = zip(*[dic_coord[name] for name in TN_names])
				ax.scatter(X,Y,marker=marker,s=40,c=list_color,label=marker_to_label[marker])
			#build the legend
			ax.legend(fontsize=14,ncol=min(3,len(marker_to_label)),bbox_to_anchor=(0.5,1.23),loc="upper center")
		else:
			labels,data = zip(*dic_coord.items())
			X,Y = zip(*data)
			ax.plot(X,Y,'.',markersize=10)
			for x,y,name in zip(X,Y,labels):
				ax.annotate(name,(x,y))
		if custom_path:
			plt.savefig(custom_path)
		else:
			Check_is_dir(self.prefix+str(self.hyp_nb)+'hyp/'+folder)
			savepath = self.prefix+str(self.hyp_nb)+'hyp/'+folder+'Compare_TN_hyp_ind_agg'+str(self.agg)+'restrict'+str(self.restrict)
			if self.logscale:
				savepath += 'LOGSCALE'
			plt.savefig(savepath+'.png')
		plt.close()

	#compute the location of the most probable value for the MEP ratio error for every ECTN in list_names
	#and associate this error to each TN for depths 2 and 3 as coordinates
	#if custom_path, the figure will be saved at custom_path
	#if name_to_cs, the TN names will not be displayed but they will be grouped by color and shape
	#according to name_to_cs
	def Compare_TN_hyp_ind_MEP_prop(self,list_names,custom_path='',name_to_cs={},marker_to_label={}):
		folder = self.folder
		#dic_coord[name] = (x,y) with x(y) = goodness of fit of hyp_nb for ECTN of depth 2(3)
		dic_coord = {}; fontsize = 15
		for name,namepath in list_names:
			print(name)
			dic_coord[name] = ()
			#load xp and th probas
			for depth in [2,3]:
				print('\tdepth',depth)
				datapath = self.prefix+name+self.suffix+str(depth)
				#load all the ECTN motifs and compute the filtered MEP ratio
				tab = Compute_MEP_ratio(Load_dic_ECTN(datapath,0),depth)
				n,bins = np.histogram(tab,density=True,bins='auto')
				ind = np.argmax(n)
				glob_location = (bins[ind] + bins[ind+1])/2
				'''
				fig,ax = Setup_Plot(r"$e(m,m')$",'nb of occurrences',fontsize=fontsize)
				ax.hist(tab,density=True,bins='auto',histtype='step',linewidth=4)
				ax.plot([glob_location]*2,ax.get_ylim(),'--',color='red')
				plt.savefig(PROJECT_ROOT+'/complenet2024/MEP_loc_example.png')
				exit()
				'''
				if self.logscale:
					glob_location = np.log10(glob_location)
				dic_coord[name] += (glob_location,)
		#place the TN on a plane
		xlabel = 'most probable error ratio for depth 2'
		ylabel = 'most probable error ratio for depth 3'
		fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
		ax.set_facecolor('gray')
		if name_to_cs:
			#group data by marker
			marker_to_name = {}
			for name,(color,marker) in name_to_cs.items():
				if marker in marker_to_name:
					marker_to_name[marker].append((name,color))
				else:
					marker_to_name[marker] = [(name,color)]
			for marker,val in marker_to_name.items():
				TN_names,list_color = zip(*val)
				X,Y = zip(*[dic_coord[name] for name in TN_names])
				ax.scatter(X,Y,marker=marker,s=40,c=list_color,label=marker_to_label[marker])
			#build the legend
			ax.legend(fontsize=fontsize,ncol=min(3,len(marker_to_label)),bbox_to_anchor=(0.5,1.23),loc="upper center")
		else:
			labels,data = zip(*dic_coord.items())
			X,Y = zip(*data)
			ax.plot(X,Y,'.',markersize=10)
			for x,y,name in zip(X,Y,labels):
				ax.annotate(name,(x,y))
		if custom_path:
			plt.savefig(custom_path)
		else:
			Check_is_dir(self.prefix+str(self.hyp_nb)+'hyp/'+folder)
			savepath = self.prefix+str(self.hyp_nb)+'hyp/'+folder+'Compare_filtered_MEP_ratio_agg'+str(self.agg)+'restrict'+str(self.restrict)
			if self.logscale:
				savepath += 'LOGSCALE'
			plt.savefig(savepath+'.png')
		plt.close()

	#figure: sample the th-xp gap for different types of ECTN motifs
	#then display the resulting distributions on the same figure
	#the th-xp gap is sampled on all TN from list_names
	def Relative_gap_vs_ECTN_hyp1(self,list_names,depth,custom_path=''):
		folder = self.folder
		freq_ratio = 20
		fontsize = 14
		gap_list = []
		for name,_ in list_names:
			print('\t'+name)
			#load xp and th probas, and ECTN strings
			datapath = self.prefix+name+self.suffix+str(depth)
			xp_proba,th_proba = Load_xp_th_proba_restrict(datapath,self.restrict,self.hyp_nb)
			#!!!make sure probas are normalized the same way!!!
			xp_proba /= np.sum(xp_proba)
			th_proba /= np.sum(th_proba)
			list_seq = Load_list_seq_restrict(datapath,self.restrict)
			#compute the relative gap for each ECTN
			gap = 1-abs(xp_proba-th_proba)/(th_proba+xp_proba)
			#remove the missing ECTN (we used only a part of the NCTN)
			gap = gap[np.nonzero(np.atleast_1d(gap))]
			if self.logscale:
				gap = np.log10(gap)
			gap_list.append((list_seq,gap))
		#figure: separate motifs with triangles from motifs without and sample the th-xp gap in both cases
		#then display the two distributions on the same figure
		motifs_tri = []; motifs_tri_mult = []; motifs_not_tri = []
		for data in gap_list:
			for seq,gap in zip(*data):
				nb_sat_three = 0
				for ind in range(1,len(seq)//depth):
					profile = seq[ind*depth:(ind+1)*depth]
					if '3' in profile or ('1' in profile and '2' in profile):
						nb_sat_three += 1
				if nb_sat_three==0:
					motifs_not_tri.append(gap)
				elif nb_sat_three==1:
					motifs_tri.append(gap)
				else:
					motifs_tri_mult.append(gap)
		#compute and plot histograms
		density = False
		xlabel = 'ECTN compatibility'; ylabel = 'number of occurrences'
		fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
		ax.hist(motifs_tri_mult,density=density,bins='auto',histtype='step',label='multiple triplet profiles')
		ax.hist(motifs_tri,density=density,bins='auto',histtype='step',label='one triplet profile')
		ax.hist(motifs_not_tri,density=density,bins='auto',histtype='step',label='no triplet profile')
		ax.legend(fontsize=fontsize)
		if custom_path:
			plt.savefig(custom_path)
		else:
			Check_is_dir(self.prefix+str(self.hyp_nb)+'hyp/'+folder)
			savepath = self.prefix+str(self.hyp_nb)+'hyp/'+folder+'Relative_gap_vs_triOrNot_agg'+str(self.agg)+'restrict'+str(self.restrict)+'depth'+str(depth)
			if self.logscale:
				savepath += 'LOGSCALE'
			plt.savefig(savepath+'.png')
		plt.close()

	#sample the ECTN th-xp gap
	#then display the resulting distribution
	#the th-xp gap is sampled on all TN from list_names
	def Relative_gap_vs_ECTN_old(self,list_names,depth):
		folder = self.folder
		fontsize = 14
		gap_list = []
		for el in list_names:
			name = el[0]
			print('\t'+name)
			#load xp, th probas and ECTN strings
			datapath = self.prefix+name+self.suffix+str(depth)
			xp_proba,th_proba = Load_xp_th_proba_restrict(datapath,self.restrict,self.hyp_nb)
			#!!!make sure probas are normalized the same way!!!
			xp_proba /= np.sum(xp_proba)
			th_proba /= np.sum(th_proba)
			list_seq = Load_list_seq_restrict(datapath,self.restrict)
			#compute the relative gap for each ECTN
			gap = 1-abs(xp_proba-th_proba)/(th_proba+xp_proba)
			#remove the missing ECTN (we used only a part of the NCTN)
			gap = gap[np.nonzero(np.atleast_1d(gap))]
			if self.logscale:
				gap = np.log10(gap)
			gap_list += list(gap)
		#compute and plot histogram
		density = True
		fig,ax = plt.subplots(constrained_layout=True)
		ax.hist(gap_list,density=density,bins='auto',histtype='step',label='all motifs')
		ax.set_xlabel('ECTN compatibility',fontsize=fontsize)
		ax.set_ylabel('number of occurrences',fontsize=fontsize)
		ax.legend(fontsize=fontsize)
		Check_is_dir(self.prefix+str(self.hyp_nb)+'hyp/'+folder)
		savepath = self.prefix+str(self.hyp_nb)+'hyp/'+folder+'Relative_gap_vs_ECTN_old_agg'+str(self.agg)+'restrict'+str(self.restrict)+'depth'+str(depth)
		if self.logscale:
			savepath += 'LOGSCALE'
		plt.savefig(savepath+'.png')
		plt.close()

	#ECTN compatibility vs frequency
	def ECTN_compa_vs_freq(self,list_names,depth):
		folder = self.folder
		fontsize = 14
		X = []; Y = []
		for el in list_names:
			name = el[0]
			print('\t'+name)
			#load xp and th probas, and ECTN strings
			datapath = self.prefix+name+self.suffix+str(depth)
			xp_proba,th_proba = Load_xp_th_proba_restrict(datapath,self.restrict,self.hyp_nb)
			#!!!make sure probas are normalized the same way!!!
			xp_proba /= np.sum(xp_proba)
			th_proba /= np.sum(th_proba)
			#compute the relative gap for each ECTN
			gap = 1-abs(xp_proba-th_proba)/(th_proba+xp_proba)
			#remove the missing ECTN (we used only a part of the NCTN)
			missing_ECTN = np.nonzero(np.atleast_1d(gap))
			gap = gap[missing_ECTN]
			xp_proba = xp_proba[missing_ECTN]
			if self.logscale:
				gap = np.log10(gap)
				xp_proba = np.log10(xp_proba)
			X += list(xp_proba)
			Y += list(gap)
		#figure: the x-axis is the ECTN frequency and the y-axis is the compatibility
		xlabel = 'ECTN frequency'
		ylabel = 'ECTN compatibility with hyp '+str(self.hyp_nb)
		fig,ax = Setup_Plot(xlabel,ylabel,fontsize=14)
		ax.set_facecolor('gray')
		ax.plot(X,Y,'.',markersize=10,alpha=0.2)
		#save the plot
		Check_is_dir(self.prefix+str(self.hyp_nb)+'hyp/'+folder)
		savepath = self.prefix+str(self.hyp_nb)+'hyp/'+folder+'ECTN_compa_vs_freq_agg'+str(self.agg)+'restrict'+str(self.restrict)+'depth'+str(depth)
		if self.logscale:
			savepath += 'LOGSCALE'
		plt.savefig(savepath+'.png')
		plt.close()

	#compute the th_probas for name,namepath for depths 2 and 3, then save data
	def Compute_ECTN_th_proba(self,name,namepath,collect=True):
		mtp = Motifs_tp(name,where=namepath)
		mtp.Get_TN(self.agg)
		for depth in [2,3]:
			print('\tdepth: '+str(depth))
			datapath = self.prefix+name+self.suffix+str(depth)
			list_seq = Load_list_seq_restrict(datapath,self.restrict)
			if collect:
				Collect_ECTN(self.agg,name,namepath)
				list_seq = Load_list_seq_restrict(datapath,self.restrict)
			print('nb of distinct xp ECTN',len(list_seq))
			path = datapath+'th_proba_restrict'+str(self.restrict)+'hyp'+str(self.hyp_nb)+'.txt'
			np.savetxt(path,mtp.Get_ind_ECTN(list_seq,depth,self.hyp_nb,self.restrict,datapath))
			'''
			if not os.path.exists(path):
				np.savetxt(path,mtp.Get_ind_ECTN(list_seq,depth,self.hyp_nb,self.restrict,datapath))
			else:
				pass
			'''

	#same as Compute_ECTN_th_proba but for NCTN
	def Compute_NCTN_th_proba(self,name,namepath,collect=True):
		if collect:
			mtp = Motifs_tp(name,where=namepath)
			mtp.Get_TN(self.agg)
		Check_is_dir(self.prefix+name+'/codata')
		for depth in [2,3]:
			print('\tdepth: '+str(depth))
			datapath = self.prefix+name+self.suffix+str(depth)

			#collect the NCTN
			if collect:
				list_seq,xp_proba = zip(*mtp.Get_dic_NCTN(depth).items())
				xp_norm = np.sum(xp_proba)
				np.savetxt(datapath+'string.txt',list_seq,fmt='%s')
				np.savetxt(datapath+'xp_proba.txt',[x/xp_norm for x in xp_proba])

			#compute th_proba
			path = datapath+'th_proba_restrict'+str(self.restrict)+'hyp'+str(self.hyp_nb)+'.txt'
			np.savetxt(path,Get_ind_NCTN(datapath,depth,self.restrict))

	#compute the activity timeline of name and compare it with:
	# - the nb of outliers at each time
	# - the max th/xp gap at each time
	# - the average th/xp gap at each time
	def Locate_outliers_ind_hyp(self,name,namepath,depth):
		folder = self.folder+'locate_outliers/'
		datapath = self.prefix+name+self.suffix+str(depth)
		#compute the activity timeline
		motifs = Motifs_tp(name,where=namepath)
		motifs.Get_TN(self.agg)
		activity = np.array([len(g.edges) for g in motifs.TN])
		#gap_dic[seq] = th-xp gap for the ECTN seq wrt the ind hyp hyp_nb
		gap_dic = Compute_gap(datapath,self.restrict,self.hyp_nb,logscale=self.logscale)
		#collect the ECTN by remembering the timestamps:
		#dic_ECTN[t][seq] = nb of occurrences of seq at time t
		dic_ECTN = motifs.Get_dic_ECTN(depth,timestamps=True)
		#restrict to the same motifs as gap_dic
		keys_to_remove = []
		for t,dic in dic_ECTN.items():
			dic_ECTN[t] = {seq:nb for seq,nb in dic.items() if seq in gap_dic}
			if len(dic_ECTN[t])==0:
				keys_to_remove.append(t)
		for key in keys_to_remove:
			del dic_ECTN[key]
		#gap_vs_time[t] = list of gap values for motifs occurring at t
		gap_vs_time = []; list_times = dic_ECTN.keys()
		for t,dic in dic_ECTN.items():
			gap_vs_time.append([gap_dic[seq] for seq in dic.keys()])
		#avg_gap[t] = avg gap at time t; max_gap[t] = avg max_gap at time t
		avg_gap = [np.mean(el) for el in gap_vs_time]
		max_gap = [np.min(el) for el in gap_vs_time]
		list_times = np.array(list(list_times))
		#plot the results
		fig,ax = plt.subplots(2,1,constrained_layout=True)
		fontsize = 12
		ax[0].set_ylabel('edge activity',fontsize=fontsize)
		ax[1].set_xlabel('time',fontsize=fontsize)
		ax[0].plot(list_times,activity[list_times],'.')
		ax[1].plot(list_times,avg_gap,'.',label='avg_gap')
		ax[1].plot(list_times,max_gap,'x',label='min_gap')
		ax[1].legend(fontsize=fontsize)
		Check_is_dir(self.prefix+str(self.hyp_nb)+'hyp/'+folder)
		savepath = self.prefix+str(self.hyp_nb)+'hyp/'+folder+'Locate_outliers_ind_hyp_agg'+str(self.agg)+'depth'+str(depth)+name
		if self.logscale:
			savepath += 'LOGSCALE'
		plt.savefig(savepath+'.png')
		plt.close()

	#plot at each time the value of the dominant mode then aggregate this time serie on a sliding window
	def Locate_modes_MEP_ratio(self,name,namepath,depth):
		folder = self.folder+'MEP_ratio/'
		datapath = self.prefix+name+self.suffix
		#compute the activity timeline
		motifs = Motifs_tp(name,where=namepath)
		motifs.Get_TN(self.agg)
		activity = np.array([len(g.edges) for g in motifs.TN])
		seq_to_mode = Mode_content_MEP_ratio(datapath,depth)
		#collect the ECTN by remembering the timestamps:
		#dic_ECTN[t][seq] = nb of occurrences of seq at time t
		dic_ECTN = motifs.Get_dic_ECTN(depth,timestamps=True)
		#restrict to the same motifs as seq_to_mode
		keys_to_remove = []
		for t,dic in dic_ECTN.items():
			dic_ECTN[t] = {seq:nb for seq,nb in dic.items() if seq in seq_to_mode}
			if len(dic_ECTN[t])==0:
				keys_to_remove.append(t)
		for key in keys_to_remove:
			del dic_ECTN[key]
		#dominant_mode[t] = k where mode k is more present at time t
		#dominant_mode_diff more present in diversity
		#dominant_mode_tot more present in total number
		dominant_mode = []
		for t,dic in dic_ECTN.items():
			n1 = 0; n2 = 0; ntot1 = 0; ntot2 = 0
			for seq in dic.keys():
				if seq_to_mode[seq]==1:
					n1 += 1
					ntot1 += dic_ECTN[t][seq]
				else:
					n2 += 1
					ntot2 += dic_ECTN[t][seq]
			if n1<n2:
				diff = 2
			else:
				diff = 1
			if ntot1<ntot2:
				dominant_mode.append((diff,2))
			else:
				dominant_mode.append((diff,1))

		#aggregate dominant_mode on a sliding window
		for choice,mode in zip(['diff','tot'],zip(*dominant_mode)):
			b = 10
			s = np.sum(mode[:b])
			res = [s]
			for k in range(len(mode)-b):
				s += mode[k+b]-mode[k]
				res.append(s)

			list_times = np.array(list(dic_ECTN.keys()))
			#plot the results
			fig,ax = plt.subplots(2,1,constrained_layout=True)
			fontsize = 12
			ax[0].set_ylabel('edge activity',fontsize=fontsize)
			ax[1].set_xlabel('time',fontsize=fontsize)
			ax[0].plot(list_times,activity[list_times],'.')
			ax[1].plot(list_times[:-b+1],res,'.',color='blue')
			Check_is_dir(self.prefix+'6hyp/'+folder)
			savepath = self.prefix+'6hyp/'+folder+'Locate_modes_MEP_ratio'+choice+'_agg'+str(self.agg)+'depth'+str(depth)+name
			plt.savefig(savepath+'.png')
			plt.close()

	#compute th_proba, project the TN on the th/xp gap plane, sample the motif gap vs freq and string
	def Analyze_hyp(self,list_names,locate=False):
		for name,namepath in list_names:
			print(name)
			self.Compute_ECTN_th_proba(name,namepath)
		print('HYP COMPUTED')
		self.Compare_TN_hyp_ind(list_names)
		print('TN COMPARED')
		for depth in [2,3]:
			print('depth',depth)
			self.ECTN_compa_vs_freq(list_names,depth)
			if self.hyp_nb==1:
				self.Relative_gap_vs_ECTN_hyp1(list_names,depth)
			else:
				self.Relative_gap_vs_ECTN_old(list_names,depth)
		if self.hyp_nb==6:
			print('MEP RATIO BEGINS')
			self.Compare_TN_MEP_ratio(list_names)
			print('LOCATE MEP OUTLIERS')
			for name,namepath in list_names:
				print(name)
				for depth in [2,3]:
					print('\t'+str(depth))
					self.Locate_modes_MEP_ratio(name,namepath,depth)

def Analyze_Utah_MEP(logscale=True,restrict=20):
	name = 'french'
	if name=='utah':
		list_proba = [((0.02,1),(0.04,1)),((0.011,0.017),(0.022,0.034)),((0.008,0.009),(0.015,0.016))]
	elif name=='min_ADM1':
		list_proba = [((0.00715,1),(0.0184,1))]
	elif name=='work1':
		list_proba = [((0.0255,1),(0.0255,1))]
	else:
		list_proba = []
	namepath = None
	agg = 1
	hyp_nb = 6
	depth = 3
	agg = 1
	prefix = 'figures/ECTN/ind_hyp/'
	suffix = '/codata/Collect_ECTN_agg'+str(agg)+'depth'+str(depth)
	datapath = prefix+name+suffix
	#display th_proba vs xp_proba (Test_MEP_hypothesis)
	xp_proba,th_proba = Load_xp_th_proba_restrict(datapath,restrict,hyp_nb)
	#!!!make sure probas are normalized the same way!!!
	xp_proba /= np.sum(xp_proba)
	th_proba /= np.sum(th_proba)
	#plot th_proba vs xp_proba
	fig,ax = plt.subplots(constrained_layout=True)
	ax.plot(xp_proba,th_proba,'.')
	ax.plot(xp_proba,xp_proba,'--')
	plt.show()
	#extract some motifs
	deviant_motifs = []; list_seq = Load_list_seq_restrict(datapath,restrict)
	dic_thproba = dict(zip(list_seq,th_proba))
	dic_xpproba = dict(zip(list_seq,xp_proba))
	for el in list_proba:
		for seq,proba in dic_xpproba.items():
			if proba<el[0][1] and proba>el[0][0]:
				if dic_thproba[seq]<el[1][1] and dic_thproba[seq]>el[1][0]:
					deviant_motifs.append(seq)
	for seq in deviant_motifs:
		print(seq)
	#plot the xp_proba vs nb of satellites
	fig,ax = plt.subplots(constrained_layout=True)
	nb_sat = np.array([len(seq)//depth-1 for seq in list_seq])
	ax.plot(nb_sat,xp_proba,'.')
	plt.show()

	#for each motif, draw a point of coordinates (xp_proba,th_proba) with a color equal
	#to the nb of satellites
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_facecolor('gray')
	ax.scatter(np.log10(xp_proba),np.log10(th_proba),c=nb_sat/np.max(nb_sat),cmap='gnuplot2')
	ax.plot(np.log10(xp_proba),np.log10(xp_proba),'--',color='orange')
	plt.show()

	exit()

	#compute the activity timeline
	motifs = Motifs_tp(name,where=namepath)
	motifs.Get_TN(agg)
	activity = np.array([len(g.edges) for g in motifs.TN])
	#gap_dic[seq] = th-xp gap for the ECTN seq wrt the ind hyp hyp_nb
	gap_dic = Compute_gap(datapath,restrict,hyp_nb,logscale=logscale)
	#collect the ECTN by remembering the timestamps:
	#dic_ECTN[t][seq] = nb of occurrences of seq at time t
	dic_ECTN = motifs.Get_dic_ECTN(depth,timestamps=True)
	#restrict to the same motifs as gap_dic
	keys_to_remove = []
	for t,dic in dic_ECTN.items():
		dic_ECTN[t] = {seq:nb for seq,nb in dic.items() if seq in gap_dic}
		if len(dic_ECTN[t])==0:
			keys_to_remove.append(t)
	for key in keys_to_remove:
		del dic_ECTN[key]
	#gap_vs_time[t] = list of gap values for motifs occurring at t
	gap_vs_time = []; list_times = dic_ECTN.keys()
	for t,dic in dic_ECTN.items():
		gap_vs_time.append([gap_dic[seq] for seq in dic.keys()])
	#avg_gap[t] = avg gap at time t; max_gap[t] = avg max_gap at time t
	avg_gap = [np.mean(el) for el in gap_vs_time]
	max_gap = [np.min(el) for el in gap_vs_time]
	list_times = np.array(list(list_times))
	#plot the results
	'''
	#remove the times of too large activity
	gap_fold = zip(list_times,avg_gap,max_gap)
	gap_fold = [el for el in gap_fold if activity[el[0]]<=20]
	list_times,avg_gap,max_gap = zip(*gap_fold)
	list_times = np.array(list_times)
	'''
	fig,ax = plt.subplots(2,1,constrained_layout=True)
	fontsize = 12
	ax[0].set_ylabel('edge activity',fontsize=fontsize)
	ax[1].set_xlabel('time',fontsize=fontsize)
	ax[0].plot(activity[list_times],'.')
	ax[1].plot(list_times,avg_gap,'.',label='avg_gap')
	ax[1].plot(list_times,max_gap,'x',label='min_gap')
	ax[1].legend(fontsize=fontsize)
	#plot the activity distribution
	fig,ax = plt.subplots(constrained_layout=True)
	ax.hist(activity,density=True,bins='auto',histtype='step')
	plt.show()

#dic_ECTN should contain all the motifs (restrict = 0, not truncated)
def Compute_MEP_ratio(dic_ECTN,depth,filtered=True):
	#check whether the ECTN sharing the same sub-NCTN have the same proba
	#cluster the ECTN by sub-NCTN pairs
	dic = {}
	for seq in dic_ECTN:
		key = ETN.Sub_NCTN(seq,depth)
		if key[0]>key[1]:
			key = key[::-1]
		if key in dic:
			dic[key].append(seq)
		else:
			dic[key] = [seq]
	#compute the avg and std proba for each sub-NCTN pair, but consider only the pairs
	#which are shared by more than one ECTN and we also remove the pairs that are shared exclusively
	#by ECTN that appear only once in the whole network
	min_freq = min(dic_ECTN.values())
	dic_info = []
	if filtered:
		for val in dic.values():
			if len(val)>1:
				cond = False
				list_prob = []
				for seq in val:
					if dic_ECTN[seq]>min_freq*1.1:
						cond = True
					list_prob.append(dic_ECTN[seq])
				if cond:
					dic_info.append(np.std(list_prob)/np.mean(list_prob))
	else:
		for val in dic.values():
			if len(val)>1:
				list_prob = [dic_ECTN[seq] for seq in val]
				dic_info.append(np.std(list_prob)/np.mean(list_prob))
	if len(dic_info)==0:
		return None
	return np.array(dic_info)

#return res where res[seq] = 1 (resp.2) if the ECTN seq belongs to the mode 1 (resp.2)
#of the MEP ratio distribution
def Mode_content_MEP_ratio(dic_ECTN,depth):
	#check whether the ECTN sharing the same sub-NCTN have the same proba
	#cluster the ECTN by sub-NCTN pairs
	dic = {}
	for seq in dic_ECTN:
		key = ETN.Sub_NCTN(seq,depth)
		if key[0]>key[1]:
			key = key[::-1]
		if key in dic:
			dic[key].append(seq)
		else:
			dic[key] = [seq]
	#compute the avg and std proba for each sub-NCTN pair, but consider only the pairs
	#which are contained by more than one ECTN
	dic_info = {}
	for key,val in dic.items():
		if len(val)>1:
			list_prob = [dic_ECTN[seq] for seq in val]
			dic_info[key] = np.std(list_prob)/np.mean(list_prob)
	if len(dic_info)==0:
		return None
	tab = np.array(list(dic_info.values()))
	n,bins = np.histogram(tab,density=True,bins='auto')
	th_index = 0
	while n[th_index+1]<=n[th_index]:
		th_index += 1
	threshold = bins[th_index+1]
	'''
	fig,ax = Setup_Plot(r"$e(m,m')$","probability density",fontsize=15)
	ax.hist(tab,density=True,bins='auto',histtype='step',linewidth=2)
	ax.plot([threshold]*2,ax.get_ylim(),'--',color='red')
	plt.show()
	return None
	'''
	res = {}
	for key,val in dic_info.items():
		mode = 2-int(val<=threshold)
		for seq in dic[key]:
			res[seq] = mode
	return res

#run the hyp 1 and 6 on Giulia datasets
def Analyze_Giulia():
	#Giulia_formatted()
	folder = 'Giulia/'
	list_names = Build_list_names_Giulia()
	motifs = Motifs_analyze(folder)
	for hyp_nb,restrict in [(1,0),(6,20)]:
		motifs.hyp_nb = hyp_nb
		motifs.restrict = restrict
		motifs.Analyze_hyp(list_names,locate=True)

#run the hyp 1 and 6 on Juliette datasets
def Analyze_Juliette():
	#Juliette_formatted()
	folder = 'Juliette/'
	list_names = Build_list_names_Juliette()
	motifs = Motifs_analyze(folder)
	for hyp_nb,restrict in [(1,0),(6,20)]:
		motifs.hyp_nb = hyp_nb
		motifs.restrict = restrict
		motifs.Analyze_hyp(list_names,locate=True)

pass
#compute the MEP distribution associated to the empirical dic_ECTN and compare the two
def Get_MEP_distribution(dic_ECTN,depth):
	#compute the gamma coefficients
	dic_gamma = {}; train_set = set()
	for seq,proba in dic_ECTN.items():
		seq1,seq2 = Sub_NCTN(seq,depth)
		if seq2==seq1:
			if seq1 not in dic_gamma:
				dic_gamma[seq1] = np.sqrt(proba)
				train_set.add(seq)
	#compute the missing gamma
	keep = True; nb_iter = 0
	while keep:
		keep = False
		print('nb_iter',nb_iter)
		for seq,proba in dic_ECTN.items():
			seq1,seq2 = Sub_NCTN(seq,depth)
			if not seq1 in dic_gamma and not seq2 in dic_gamma:
				keep = True
			else:
				if not seq1 in dic_gamma:
					dic_gamma[seq1] = proba/dic_gamma[seq2]
					train_set.add(seq)
				elif not seq2 in dic_gamma:
					dic_gamma[seq2] = proba/dic_gamma[seq1]
					train_set.add(seq)
		nb_iter += 1
		if nb_iter==2:
			keep = False

	#compute th_proba (the MEP distribution)
	list_seq,xp_proba = zip(*dic_ECTN.items())
	th_proba = np.zeros(len(xp_proba))
	for k,seq in enumerate(list_seq):
		seq1,seq2 = Sub_NCTN(seq,depth)
		if not seq1 in dic_gamma and not seq2 in dic_gamma:
			pass
		else:
			if not seq1 in dic_gamma:
				dic_gamma[seq1] = xp_proba[k]/dic_gamma[seq2]
				train_set.add(seq)
			elif not seq2 in dic_gamma:
				dic_gamma[seq2] = xp_proba[k]/dic_gamma[seq1]
				train_set.add(seq)
			th_proba[k] = dic_gamma[seq1]*dic_gamma[seq2]
	#display the distributions
	fig,ax = plt.subplots(constrained_layout=True)
	train_data = []; test_data = []
	for k,seq in enumerate(list_seq):
		if seq in train_set:
			train_data.append((xp_proba[k],th_proba[k]))
		else:
			test_data.append((xp_proba[k],th_proba[k]))
	ax.set_facecolor('gray')
	ax.plot(*zip(*train_data),'.',color='red')
	ax.plot(*zip(*test_data),'.',color='blue')
	ax.plot(xp_proba,xp_proba,'--',color='orange')
	plt.show()

#plot the inst degree distribution for Juliette models:
#what is the distribution of 'ABPpi4_jul' ?? (decrease faster than an exponential
#so maybe x^{-\alpha}\exp(k*x)) or \exp(k*x^{alpha}))
def Juliette_degree():
	agg = 1
	prefix = 'figures/deg_inst/Juliette/'
	Check_is_dir(prefix)
	for name,namepath in [('ABPpi4_jul',None)]:#Build_list_names_Juliette():
		print(name)
		motifs = Motifs_tp(name,where=namepath)
		motifs.Get_TN(agg)
		X,Y = zip(*motifs.Get_direct_deg_histo().items())
		xlabel = 'degree'
		ylabel = r'$\log_{10}(P)$'
		fig,ax = Setup_Plot(xlabel,ylabel,fontsize=14)
		ax.plot(X,np.log10(Y),'.')
		#savepath = prefix+'node_deg_agg'+str(agg)+name
		#plt.savefig(savepath+'.png')
		plt.show()

#visualize the Juliette motifs
def Visu_Juliette():
	#Juliette_formatted()
	restrict = 0
	list_names = Build_list_names_Juliette()
	motifs = Motifs_analyze(''); depth = 3
	folder = 'figures/ECTN/visu/Juliette/'
	Check_is_dir(folder)
	for name,namepath in list_names:
		print(name)
		datapath = motifs.prefix+name+motifs.suffix+str(depth)
		savepath = folder+name+'depth'+str(depth)
		list_seq = Load_list_seq_restrict(datapath,restrict)
		xp_proba = Load_xp_proba_restrict(datapath,restrict)
		ETN.Draw_ten_freq_ECTN(dict(zip(list_seq,xp_proba)),depth,savepath)

#return a dic with dic[name] = path where the TN with original time steps is stored
#this allows to split the TN in multiple days
def Build_original_TN_with_multiple_days():
	dic = {}
	prefix = PROJECT_ROOT+'/data/original_tij/'

	##empirical TN
	folder = 'empirical/'
	#conferences
	old_names = ['tij_WS16.dat','tij_ICCSS17.dat','tij_ECSS18.dat','tij_ECIR19.dat']
	new_names = ['conf1'+str(k) for k in range(6,10)]
	#highschools
	old_names += ['thiers_2011.csv','thiers_2012.csv','High-School_data_2013.csv']
	new_names += ['highschool'+str(k) for k in range(1,4)]
	#workplaces
	old_names += ['tij_InVS.dat','tij_InVS15.dat']
	new_names += ['work1','work2']
	#french primary school, malawi and hospital
	#malawi, baboons should be discarded because they are too small
	#hospital is just at the limit soshould be discarded as well
	old_names += ['primaryschool.csv','tnet_malawi_pilot2.txt','detailed_list_of_contacts_Hospital.dat']
	new_names += ['french','malawi','hospital']

	for old_name,name in zip(old_names,new_names):
		dic[name] = prefix + folder + old_name

	##Giulia data sets
	folder = 'reseaux_Giulia/'

	for old_name,name in zip(*Giulia_path_to_raw_TN()):
		dic[name] = PROJECT_ROOT+'/data/'+old_name
	return dic

#return data_time for raw TN, i.e. with original times
#data_time[k] = (t,n1,n2) with tij_data[n1:n2,:] being equal to the interactions occurring at time t
def Get_data_time(tij_data):
	data_time = []
	n1 = 0; n_max = np.size(tij_data,0)
	for n in range(1,n_max):
		if tij_data[n,0]>tij_data[n-1,0]:
			data_time += [(tij_data[n-1,0],n1,n)]
			n1 = n
	#take care of the last line of tij_data
	data_time += [(tij_data[n,0],n1,n_max)]
	return data_time

#return a list of formatted tables tij, each corresponding to one day
#the TN analyzed is loaded from path
def Split_days(path):
	tij_data = np.loadtxt(path,dtype=str)[:,:3].astype(int)
	data_time = Get_data_time(tij_data)

	#list of indices in data_time separating the days
	day_indices = [0]
	#time resolution: smallest step btw two consecutive measures
	resolution = data_time[1][0] - data_time[0][0]
	for ind,el in enumerate(data_time[1:],start=1):
		step = el[0] - data_time[ind-1][0]
		if step>50*resolution:
			day_indices += [ind-1,ind]
	day_indices.append(len(data_time)-1)
	#day_couples[k] = (start,end) the start and end of the day as indices of data_time
	day_couples = [(day_indices[2*k],day_indices[2*k+1]) for k in range(len(day_indices)//2)]

	#remove the days that contain too few data
	#day_info[day] = (nb of measured times in the day , nb of interactions in the day)
	day_info = {day:(day[1]-day[0],data_time[day[1]][2]-data_time[day[0]][1]) for day in day_couples}
	avg_info = ()
	for tab in zip(*day_info.values()):
		avg_info += (np.mean(tab)/10,)
	confirmed_days = [day for day,info in day_info.items() if (info>avg_info).all()]

	#last check: the maximum nb of interactions measured per time step should exceed 3 on each day
	day_info = {}
	for day in confirmed_days:
		max_nb = max([data_time[ind][2]-data_time[ind][1] for ind in range(day[0],day[1]+1)])
		day_info[day] = max_nb
	confirmed_days = [day for day,info in day_info.items() if info>3]

	list_tp = []
	for day in confirmed_days:
		tp = Motifs_tp(tij_data[data_time[day[0]][1]:data_time[day[1]][2],:])
		tp.Format()
		list_tp.append(tp)
	return list_tp

def Plot_raw_timeline(tij_data,title):
	data_time = Get_data_time(tij_data)
	#plot the activity to assess the splitting
	fig,ax = Setup_Plot(r'$t$','number of interactions',fontsize=14,title=title)
	t,n1,n2 = zip(*data_time)
	ax.plot(t,np.array(n2)-np.array(n1),'.')

#load a TN containing multiple days (xp or Giulia)
#return a instance of Motifs_tp for each TN
def Load_split_TN(name):
	name_to_path = Build_original_TN_with_multiple_days()
	return Split_days(name_to_path[name])

def Get_list_names_with_labels():
	#blue circle = conf
	#green 'v' = schools
	#black '>' = models_1 (ADM + EW)
	#orange '<' = models_2 Giulia
	#red square = model_pedestrian
	#white 'x' = other_xp (workplace + hospital)
	family_to_cs = {'conf':('blue','o'),'schools':('green','v'),'models_1':('black','>')}
	family_to_cs['models_2'] = ('orange','<')
	family_to_cs['model_pedestrian'] = ('red','s')
	family_to_cs['other_xp'] = ('white','x')
	family_to_name = Build_list_names_with_family()
	name_to_cs = {name:val for family,val in family_to_cs.items() for name in family_to_name[family]}
	marker_to_label = {val[1]:key for key,val in family_to_cs.items()}
	list_names = []
	for val in family_to_name.values():
		list_names += list(zip(val,[None]*len(val)))
	return family_to_name,name_to_cs,marker_to_label,list_names

#investigate the alternative second hypothesis of independence, which states that edges are
#independent when conditionned on some hidden variables
def Complete_workflow_ind_hyp(hyp_nb,restrict):
	#if MEP, Remove_names_from_list(list_names,['ADM17utah','Vicsekpi4','brownD1','ABP2pi'])
	savepath = PROJECT_ROOT+'/manuscript/chapter5/'
	mta = Motifs_analyze('',hyp_nb=hyp_nb,restrict=restrict)
	#build list_names
	family_to_name,name_to_cs,marker_to_label,list_names = Get_list_names_with_labels()

	#remove RW_periodic
	if hyp_nb==2:
		ind = 0
		while list_names[ind][0]!='RW_periodic':
			ind += 1
		del list_names[ind]
		del name_to_cs["RW_periodic"]
		family_to_name["model_pedestrian"] = [name for name in family_to_name["model_pedestrian"] if name!='RW_periodic']

	#compute th_proba for hyp_nb
	'''
	for name,namepath in list_names:
		print(name)
		mta.Compute_ECTN_th_proba(name,namepath,collect=False)
	'''
	print('HYP COMPUTED')
	#evaluate the hyp2 for all TN
	mta.logscale = True
	mta.Compare_TN_hyp_ind(list_names,custom_path=savepath+'hyp'+str(mta.hyp_nb)+'_LOGSCALE'+str(restrict)+'.png',name_to_cs=name_to_cs,marker_to_label=marker_to_label)
	mta.logscale = False
	mta.Compare_TN_hyp_ind(list_names,custom_path=savepath+'hyp'+str(mta.hyp_nb)+str(restrict)+'.png',name_to_cs=name_to_cs,marker_to_label=marker_to_label)

#investigate the hypothesis 1 of independence for NCTN
def Complete_workflow_hyp_NCTN(restrict):
	savepath = PROJECT_ROOT+'/manuscript/chapter5/'
	#build list_names
	family_to_name,name_to_cs,marker_to_label,list_names = Get_list_names_with_labels()
	mta = Motifs_analyze('',hyp_nb=1,restrict=restrict)
	mta.prefix = 'figures/NCTN/ind_hyp/'
	mta.suffix = '/codata/Collect_NCTN_agg'+str(mta.agg)+'depth'

	'''
	#compute th_proba for hyp1
	for name,namepath in list_names:
		print(name)
		mta.Compute_NCTN_th_proba(name,namepath,collect=False)
	'''

	print('HYP COMPUTED')
	#evaluate the hyp1 for all TN
	mta.logscale = True
	mta.Compare_TN_hyp_ind(list_names,custom_path=savepath+'hyp1_NCTN_LOGSCALE'+str(restrict)+'.png',name_to_cs=name_to_cs,marker_to_label=marker_to_label)
	mta.logscale = False
	mta.Compare_TN_hyp_ind(list_names,custom_path=savepath+'hyp1_NCTN'+str(restrict)+'.png',name_to_cs=name_to_cs,marker_to_label=marker_to_label)

def Test_version_workflow_2ind_hyp():
	name = 'conf16'; depth = 3; nb_states = 2; agg = 1
	list_tp = Load_split_TN(name)
	print('TN split')
	for day_nb,tp in enumerate(list_tp):
		tp.Get_TN(agg)
		dic_ECTN = tp.Get_dic_ECTN(depth)
		list_ECTN,xp_proba = zip(*dic_ECTN.items())
		xp_proba,list_ECTN = Restrict_seq(xp_proba,list_ECTN,freq_ratio=20)
		dic_ECTN = dict(zip(list_ECTN,xp_proba))
		print('dic_ECTN computed')

		'''
		#test the hyp 1 for the sake of one's mind
		th_proba = tp.Get_ind1_ECTN(list_ECTN,depth)
		th_proba = np.array(th_proba)/sum(th_proba)
		xp_proba /= np.sum(xp_proba)
		fig,ax = plt.subplots(constrained_layout=True)
		ax.plot(xp_proba,th_proba,'.')
		ax.plot(th_proba,th_proba,'--')
		plt.show()
		exit()
		'''

		hyp2 = Interpolate_distr(nb_states)
		hyp2.preprocess(dic_ECTN,depth)
		print('preprocess done')

		param = hyp2.fit_model()
		hyp2.set_param_from_flat(param)
		print('fit done')

		#compute xp and th probas
		xp_proba = hyp2.train_target*hyp2.list_factors
		th_proba = hyp2.predict_for_fit()*hyp2.list_factors

		fig,ax = plt.subplots(constrained_layout=True)
		ax.plot(xp_proba,th_proba,'.')
		ax.plot(th_proba,th_proba,'--')
		plt.show()
		exit()

#plot the ten most frequent ECTN in each one of the two MEP modes
#the goal is to check the two contain triangles
def Plot_first_ECTN_MEP_modes():
	name = 'conf16'; depth = 3
	prefix = PROJECT_ROOT+'/complenet2024/'+name
	motifs = Motifs_analyze('')
	#load dic_ECTN
	datapath = motifs.prefix+name+motifs.suffix+str(depth)
	dic_ECTN = Load_dic_ECTN(datapath,motifs.restrict)
	norm = sum(dic_ECTN.values())
	for seq,val in dic_ECTN.items():
		dic_ECTN[seq] = val/norm

	#separate the two modes (two modes named 1 and 2)
	seq_to_mode = Mode_content_MEP_ratio(dic_ECTN,depth)
	dic1 = {}; dic2 = {}
	for seq,mode in seq_to_mode.items():
		if mode==1:
			dic1[seq] = dic_ECTN[seq]
		else:
			dic2[seq] = dic_ECTN[seq]

	#histo of the nb of distinct profiles
	#motifs from mode 1 are more complex than motifs from mode 2
	tab1 = []
	for seq in dic1:
		nb = 0; old_prof = ''
		for i in range(1,len(seq)//depth):
			prof = seq[i*depth:(i+1)*depth]
			if prof!=old_prof:
				nb += 1
				old_prof = prof
		tab1.append(nb)
	tab2 = []
	for seq in dic2:
		nb = 0; old_prof = ''
		for i in range(1,len(seq)//depth):
			prof = seq[i*depth:(i+1)*depth]
			if prof!=old_prof:
				nb += 1
				old_prof = prof
		tab2.append(nb)
	fontsize = 15
	fig,ax = Setup_Plot('number of distinct profiles','number of occurrences',fontsize=fontsize,title='')
	ax.hist(tab1,density=True,bins='auto',histtype='step',linewidth=2,label='MEP-compatible mode')
	ax.hist(tab2,density=True,bins='auto',histtype='step',linewidth=2,label='MEP-outlier mode')
	ax.legend(fontsize=fontsize)
	plt.savefig(prefix+'diff_prof_MEP.png')

	#mode1 = rare motifs
	#mode2 = rare motifs + frequent motifs
	fig,ax = Setup_Plot('ECTN log-frequency','number of occurrences',fontsize=fontsize)
	tab1 = np.log10([dic_ECTN[seq] for seq in dic1])
	tab2 = np.log10([dic_ECTN[seq] for seq in dic2])
	ax.hist(tab1,density=True,bins='auto',histtype='step',linewidth=2,label='MEP-compatible mode')
	ax.hist(tab2,density=True,bins='auto',histtype='step',linewidth=2,label='MEP-outlier mode')
	ax.legend(fontsize=fontsize)
	plt.savefig(prefix+'MEP_mode_freq.png')

	tot_nb3 = 0
	for seq in dic1:
		for letter in seq:
			if letter=='3':
				tot_nb3 += 1
	print('nb of 3 in mode 1:',tot_nb3)
	tot_nb3 = 0
	for seq in dic2:
		for letter in seq:
			if letter=='3':
				tot_nb3 += 1
	print('nb of 3 in mode 2:',tot_nb3)

	#plot the ten most frequent ECTN for each mode
	savepath1 = prefix+'visu_mode1_'
	savepath2 = prefix+'visu_mode2_'
	for rank,suffix in zip([0,10],['0to10','10to20']):
		ETN.Draw_ten_freq_ECTN(dic1,depth,savepath1+suffix+'.png',starting_rank=rank,normalize=False)
		ETN.Draw_ten_freq_ECTN(dic2,depth,savepath2+suffix+'.png',starting_rank=rank,normalize=False)

def Compare_error_ratio():
	savepath = PROJECT_ROOT+'/manuscript/chapter5/'
	mta = Motifs_analyze('',hyp_nb=6,restrict=0)
	family_to_name,name_to_cs,marker_to_label,list_names = Get_list_names_with_labels()
	mta.logscale = True
	custom_path = savepath+'MEP_compa_LOGSCALE.png'
	mta.Compare_TN_hyp_ind_MEP_prop(list_names,custom_path=custom_path,name_to_cs=name_to_cs,marker_to_label=marker_to_label)

#predict and compare with xp the distribution of the nb of bursty periods assuming the edge interduration
#follows a Markov process
def compare_th_xp_bursty_distr():
	list_names = ['conf16','french','utah','highschool3','work2']

	tot_data = []
	for name in list_names:
		print(name)
		mtp = Motifs_tp(name)
		mtp.Get_TN(1)
		edge_event = mtp.Edge_event_train()

		#compute the distribution of the nb of bursty periods
		nb_bursty = Train_weak_duration_histo(edge_event)

		#compute the partition function and T_{11}
		partition = {}; norm = 0; n11 = 0; contacts = []
		for timeline in edge_event.values():
			contacts.append(len(timeline))
			#compute T_{11}
			for i,x in enumerate(timeline[1:-1],start=1):
				delta_0 = x[0]-timeline[i-1][1]-1
				if delta_0==1:
					delta_1 = timeline[i+1][0]-x[1]-1
					norm += 1
					if delta_1==1:
						n11 += 1
		T_11 = n11/norm
		#compute partition
		contacts.sort()
		for n in nb_bursty:
			i = 0
			while contacts[i]<n:
				i += 1
			partition[n] = sum(contacts[i:])-(n-1)*(len(contacts)-i)
		#deduce the theoretical distribution for the nb of bursty periods
		th_nb_bursty = {}; keys_to_remove = []
		for n in nb_bursty:
			if n+1 in partition:
				th_nb_bursty[n] = partition[n+1]*T_11**n
			else:
				keys_to_remove.append(n)
		for key in keys_to_remove:
			del nb_bursty[key]

		data_xp = Raw_to_binned(Norm_dic_histo(nb_bursty))
		data_th = Raw_to_binned(Norm_dic_histo(th_nb_bursty))
		tot_data.append((data_xp,zip(*sorted(zip(*data_th),key=lambda el:el[0]))))


	#compare the two histo
	fontsize = 15; xlabel = r'$\log_{10}(|b|)$'; ylabel = r"$\log_{10}(P)$"
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	shift = 0
	for name,data,marker,color in zip(list_names,tot_data,LIST_MARKER,LIST_COLOR):
		x0,y0 = data[0]; x1,y1 = data[1]
		ax.plot(x0,np.array(y0)+shift,marker,color=color,label=name)
		ax.plot(x0,np.array(y1)+shift,'--',color='gray')
		shift -= 1
	ax.legend(fontsize=fontsize)
	FIGPATH = os.path.join(PROJECT_ROOT,'manuscript/chapter2/obs_distr/th_nb_bursty.png')
	plt.savefig(FIGPATH)

'''
Compare_error_ratio()
exit()
mta = Motifs_analyze(''); depth = 3
savepath = PROJECT_ROOT+'/manuscript/chapter5/'
for name in ['conf16','utah','work2','ADM9conf16']:
	print(name)
	mta.Plot_MEP_ratio_histo(name,None,depth,filtered=False)
	plt.savefig(savepath+name+'MEP_ratio_histo.png')
	plt.close()
exit()
family_to_name,name_to_cs,marker_to_label,list_names = Get_list_names_with_labels()
for key,val in family_to_name.items():
	print(key+': '+str(len(val)))
	for name in val:
		print('\t'+name)
print(sum([len(val) for val in family_to_name.values()]))
exit()
'''

def func_x(n):
	return (n+2)*(n+1)/2
def func_y(n):
	return (n+6)*(n+5)*(n+4)*(n+3)*(n+2)*(n+1)/720

'''
name = 'utah'; agg = 1
mtp = Motifs_tp(name)
mtp.Get_TN(agg)
for depth in [2,3]:
	print(depth)
	fontsize = 15
	if depth==2:
		func = func_x
	else:
		func = func_y
	dic_NCTN = mtp.Get_dic_NCTN(depth)
	#sample the number of satellites
	sat_histo = {}
	#compute the number of distinct NCTN vs number of satellites and compare it with the maximum nb
	histo = {}
	for seq,nb in dic_NCTN.items():
		nb_sat = len(seq)//depth
		if nb_sat in sat_histo:
			sat_histo[nb_sat] += nb
			histo[nb_sat] += 1
		else:
			sat_histo[nb_sat] = nb
			histo[nb_sat] = 1
	for nb_sat,val in histo.items():
		histo[nb_sat] = np.log10(val/func(nb_sat))

	#plot the observed diversity of NCTN vs maximum possible
	xlabel = "number of satellites"; ylabel = "log of the proportion of\nobserved distinct NCTN"
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	ax.plot(*zip(*histo.items()),'.')
	plt.savefig(PROJECT_ROOT+'/manuscript/chapter5/omega_hyp/diversity'+name+'depth'+str(depth)+'.png')


	sat_histo = Norm_dic_histo(sat_histo)
	#compute the NCTN proba in case N_{n}(m) depends only on |m|
	xp_proba = []; th_proba = []; norm = sum(dic_NCTN.values())
	for seq,nb in dic_NCTN.items():
		xp_proba.append(nb/norm)
		nb_sat = len(seq)//depth
		binom = func(nb_sat)
		th_proba.append(sat_histo[nb_sat]/binom)
	m = np.min(xp_proba)
	X = []; Y = []
	for x,y in zip(xp_proba,th_proba):
		if y<m:
			pass
		else:
			X.append(x); Y.append(y)
	xp_proba = X; th_proba = Y
	#first figure: th_proba vs xp_proba
	xlabel = "observed probability"; ylabel = "theoretical probability"
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	ax.plot(xp_proba,th_proba,'.')
	m = np.min(xp_proba); M = np.max(xp_proba)
	ax.plot([m,M],[m,M],'--')
	plt.savefig(PROJECT_ROOT+'/manuscript/chapter5/omega_hyp/xpth_proba'+name+'depth'+str(depth)+'.png')
	#second figure: xp_proba and th_proba plotted vs NCTN string
	fig,ax = Setup_Plot(xlabel,ylabel,fontsize=fontsize)
	ax.plot(np.log10(xp_proba),np.log10(th_proba),'.')
	ax.plot([np.log10(m),np.log10(M)],[np.log10(m),np.log10(M)],'--')
	plt.savefig(PROJECT_ROOT+'/manuscript/chapter5/omega_hyp/xpth_probaLOG'+name+'depth'+str(depth)+'.png')

exit()
'''
def Fig_complenet2024():
	savepath = PROJECT_ROOT+'/complenet2024/'
	#use color and shape code to distinguish btw families of TN:
	#blue circle = conf
	#green 'v' = schools
	#black '>' = models_1 (ADM + EW)
	#orange '<' = models_2 Giulia
	#red square = model_pedestrian
	#white 'x' = other_xp (workplace + hospital)
	family_to_name,name_to_cs,marker_to_label,list_names = Get_list_names_with_labels()
	motifs = Motifs_analyze('',hyp_nb=1,restrict=0)
	##MEP (hyp6): remove the MEP artifact to compute the MEP ratio distribution, then characterize a TN
	##by the most probable ratio value for depths 2 and 3 ; visualize the TN in this plane
	motifs.logscale = False
	motifs.Compare_TN_hyp_ind_MEP_prop(list_names,custom_path=savepath+'MEP_prop_TN.png',name_to_cs=name_to_cs,marker_to_label=marker_to_label)
	motifs.logscale = True
	##hyp1: compare the TN according the xp-th gap
	motifs.Compare_TN_hyp_ind(list_names,custom_path=savepath+'hyp1_TN.png',name_to_cs=name_to_cs,marker_to_label=marker_to_label)
	##hyp1: display the ECTN compatibility distr (modes = triplets) for xp TN only
	motifs.restrict = 20
	list_names = []
	for key in ['conf','schools','other_xp']:
		val = family_to_name[key]
		list_names += list(zip(val,[None]*len(val)))
	motifs.Relative_gap_vs_ECTN_hyp1(list_names,3,custom_path=savepath+'triplets.png')
	##MEP: draw the histogram of the error wrt same group --> same probability
	depth = 3; motifs.restrict = 0
	for name in ['conf16','highschool3']:
		print(name)
		motifs.Plot_MEP_ratio_histo(name,None,depth)
		plt.savefig(savepath+name+'MEP_ratio_histo_trunc.png')
		plt.close()
	##MEP: plot the cc size, contact duration and node degree distr before and after removing interactions
	##associated to the outlier mode
	day_nb = 1; depth = 3; agg = 1; save_folder = savepath
	for name in ['conf16','highschool3']:
		print(name)
		tp = Load_split_TN(name)[day_nb]
		#plot the old timeline
		Plot_raw_timeline(tp.data,'original network')
		plt.savefig(save_folder+name+'activity_raw.png')
		tp.Get_TN(agg)
		old_contact = Duration_histo(tp.Edge_event_train())
		old_cc_size = tp.Get_cc_size()
		old_degree = tp.Get_direct_deg_histo()

		dic_ECTN = tp.Get_dic_ECTN(depth)
		seq_to_mode = Mode_content_MEP_ratio(dic_ECTN,depth)
		#identify the interactions in which the motifs of each mode are involved
		mode_to_interactions = tp.Get_mode_to_interactions_ECTN(depth,seq_to_mode)
		#remove the interactions associated to the mode 2
		data = tp.Remove_interactions_from(mode_to_interactions[2])
		#plot the new timeline
		Plot_raw_timeline(data,'filtered network')
		plt.savefig(save_folder+name+'activity_filtered.png')
		#plt.show()
		#compute the distribution of cc size, contact duration and node degree
		tp.data = data
		tp.Format()
		tp.Get_TN(agg)

		#contact duration
		histo = Duration_histo(tp.Edge_event_train())
		fontsize = 15
		title = 'contact duration'
		xlabel = r'$\log_{10}(\tau)$'
		ylabel = r'$\log_{10}(P)$'
		fig,ax = Setup_Plot(xlabel,ylabel,title=title,fontsize=fontsize)
		ax.plot(*Raw_to_binned(old_contact),'.',label='original')
		ax.plot(*Raw_to_binned(histo),'.',label='filtered')
		ax.legend(fontsize=fontsize)
		plt.savefig(save_folder+name+'contact_duration.png')

		#cc size in nodes
		title = 'size in nodes of connected components'
		xlabel = r'$\log_{10}(n)$'
		fig,ax = Setup_Plot(xlabel,ylabel,title=title,fontsize=fontsize)
		ax.plot(*Raw_to_binned(old_cc_size),'.',label='original')
		ax.plot(*Raw_to_binned(tp.Get_cc_size()),'.',label='filtered')
		ax.legend(fontsize=fontsize)
		plt.savefig(save_folder+name+'cc_size.png')

		#node degree
		title = 'node degree'
		xlabel = r'$k$'
		fig,ax = Setup_Plot(xlabel,ylabel,title=title,fontsize=fontsize)
		X,Y = zip(*old_degree.items())
		ax.plot(X,np.log10(Y),'.',label='original')
		X,Y = zip(*tp.Get_direct_deg_histo().items())
		ax.plot(X,np.log10(Y),'.',label='filtered')
		ax.legend(fontsize=fontsize)
		plt.savefig(save_folder+name+'node_degree.png')
		plt.close('all')

if __name__=="__main__":
	pass
	#envoyer un mail à Fabio
	#idea: maybe we should rank the motifs according to the nb of interactions they contain rather than
	#their nb of occurrences
	pass
	#consider one TN
	#extract ECTN for each day, sample MEP ratio, compute seq_to_mode, remove the interactions
	#corresponding to the ECTN of the second mode, then visualize the new timeline and differences

	#plot at each time, the number of interactions contained in the ECTN removed under MEP filtering
	#plot the fraction nb of interactions at t / nb of such interactions at t vs t
	pass

	pass
	#Complete_workflow_2ind_hyp()
	exit()

	pass
	#plot the number of interactions contained in all the ECTN of rank < r as a fct of r
	#(or vs the fraction of distinct ECTN)
	pass

	name = 'conf19'
	save_folder = 'figures/ECTN/ind_hyp/6hyp/MEP_ratio/'+name+'/'
	Check_is_dir(save_folder)

	exit()
	tp = Motifs_tp(day_data)
	print()
	'''
	#load all the ECTN motifs
	list_seq = Load_list_seq_restrict(datapath,0)
	xp_proba = Load_xp_proba_restrict(datapath,0)
	xp_proba /= np.sum(xp_proba)
	dic_ECTN = dict(zip(list_seq,xp_proba))
	'''
	exit()


	#create a file days_length.txt located in PROJECT_ROOT/data/ with two columns:
	#first is the TN name, second is the number of days it contains
	days_length = []
	TN_paths = Build_original_TN_with_multiple_days()
	for name,path in TN_paths.items():
		print(name)
		list_days = Split_days(path)
		days_length.append([name,str(len(list_days))])
	np.savetxt(PROJECT_ROOT+'/data/days_length.txt',np.array(days_length,dtype=str),fmt='%s')
	exit()

	#Analyze_Juliette();exit()
	Analyze_Giulia()
	print()
	Analyze_Juliette();exit()

	Analyze_Utah_MEP(); exit()

	agg = 1; depth = 3
	name = 'conf16'; namepath = None
	pass
	exit()
	Complete_workflow_2ind_hyp()
	exit()

	hyp_nb = 6; agg = 1
	threshold = 20
	list_names = Build_list_names()
	'''
	namepath = None; depth = 3; restrict = 20; hyp_nb = 6
	for name in list_names:
		print(name)
		Test_MEP_hypothesis(name,namepath,depth,restrict,hyp_nb)
	exit()
	'''
	#run the MEP hypothesis for different thresholds when restricting to most frequent motifs
	list_threshold = [20]#[2,5,10,20]
	'''
	with mp.Pool() as p:
		p.map(Parallel_compute_ECTN_th_proba_MEP,list_threshold)
	'''

	#test for ECTN TR at the level of profiles:
	#identify '012' with '021' but not '012' and '011', i.e. consider profiles equivalent iif they are image
	#of one another under 1<-->2
	agg = 1
	#compute the degree of symetry under TR for all TN
	Compute_sym_ECTN_TR(agg,transverse=True)
	Plot_sym_ECTN_TR(agg,transverse=True)
	exit()

	#results from ECTN visualization:
	# - min_EW3 is TR symmetric but min_ADM2 and xp data are not
	# - we have TR at the level of profiles, but not at the level of motifs

	agg = 1; depth = 3
	list_names = ['french','min_EW3','conf16','utah','min_ADM2']
	for name in list_names:
		print(name)
		Complete_motifs_visu(name,agg,depth,'ECTN',transverse=True)
	exit()

	#visualize the ten most frequent NCTN of each TN realization
	cc_fixed = True; choice_motifs = 'NCTN'
	list_names = Get_seq_cc_Giulia(cc_fixed=cc_fixed)
	agg = 1; depth = 3
	ref_name = 'french'
	motifs = Motifs_tp(ref_name)
	motifs.Get_TN(agg)
	ref_ECTN = motifs.Get_dic_ECTN(depth)
	#ECTN sim with xp ref is 0.27
	#NCTN sim with xp ref is 0.94
	for name in list_names:
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(agg)
		dic_NCTN = motifs.Get_dic_ECTN(depth)
		print(Cosim(dic_NCTN,ref_ECTN))
	exit()

	#note: the NCTN similarity btw min_EW3 and xp data (conf16 in particular) is way too high... (>0.94)
	#!!!the NCTN sim cannot be used alone as a sim measure!!!
	#something is needed:
	# - how to explain this similarity: compute the theoretical NCTN similarity
	#  (possible thanks to the ind hyp and the analytical treatment of min_EW3)
	# - restrict to some relevant motifs, either chosen by hand or chosen on the basis of a randomized model
	#  (take a motif iif it is either more or less frequent than in the randomized version)

	#results of TO probing:
	#min_ADM2 is the closest from xp data, whereas it is the farfest from the point of view of the ECTN and
	#NCTN similarities. Moreover, min_EW3 is very (way too much) close to XP wrt motifs sim but is very far
	#from xp wrt TO sym
	#probe the TO (time ordering) symmetry in xp, ADM and pedestrian models
	#first trial:
	# - compute the ratios N(101)/N(110) ; N(100)/N(010)
	# - compute the ratios N(1011)/N(1110) ; N(1010)/N(1100) ; N(0110)/N(1001) ; N(0110)/N(1100)
	#   ; N(1000)/N(0100)
	#figure:
	# - depth=4: each TN is assigned a color (adm, xp and pedestrian), then four plots, one for each ratio
	#   on the same figure
	#   eventually, other figure: place TN as colored labelled dots in the 2D-PCA plane
	#   (each TN is characterized by its ratios vector)
	# - depth=3: same as depth=4 + other figure: x is one ratio and y is the other, then place TN as dots
	#   in this plane, adding colors or labels for dots
	#plot these ratios vs data sets

	agg = 1; depth = 4
	Plot_profiles_ratios_depth4(agg)

	pass
	#results for pedestrian degree: how is it that ABPpi4 has a maximum degree of 6 at aggregation level 100??
	#this said:
	#brownD01 != xp
	#ABPpi4 != xp (cannot identify the distribution because the degree is much too small)
	# the distribution is close to xp case (geometric law)
	#overall, pedestrian models are extremely sparse !!!! --> impossible to compare with xp

	#unweighted aggregated degree/weighted aggregated degree vs aggregation level
	name = 'ADM9conf16'; agg = 1; depth = 3
	list_names = ['conf16','utah','baboon','ADM9conf16','ADM18conf16','min_ADM2','min_EW3']
	list_agg = range(1,21); savefig = 'small_agg'
	for name in list_names:
		print(name)
		Draw_agg_degree_ratio(name,list_agg,savefig)

	############################################################################

	#semi-stochastic process: the goal is to obtain a geometric law as asymptotic distribution
	#hyp on degree:
	# - bounded btw 0 and 15
	# - continuous variations (most of variations equal to zero or 1)
	#actually interaction duration follows a power-law ; high values for the degree may have two origins:
	# - interactions of duration less than the temporal resolution of the network (i.e. we have
	#aggregated interactions instead of simultaneous)
	# - false signals due to the nature of the sensors: in a dense environment, signals are exchanged even
	#between non-discussing people
	#however, at maximum, a single individual is talking to 3 other people at the same time
	#any degree higher than 3 includes either false signals (due to crowd density)
	#or aggregated interactions (due to a succession of short contacts)

	#what is the contribution of these two factors to the degree distribution?

	name = "conf16"; agg = 5
	motifs = Motifs_tp(name)
	motifs.Get_TN(agg)
	motifs.Draw_degree_profile()

	exit()

	for name in ["conf16","utah"]:
		print(name)
		Plot_interduration(name)
	exit()
	for name in ["min_EW3","conf16","utah","ADM9conf16",'ADM18conf16','min_ADM2']:
		print(name)
		Flow_degree_distr(name)
	#Plot_transfo_degree(name,depth)
	exit()
	for name in ['min_EW3','min_ADM2']:
		print(name)
		Compare_degree_distr(name,20,gaussian_hyp=True)
	exit()
	for name in ["min_EW3","conf16","utah","ADM9conf16",'ADM18conf16','min_ADM2']:
		print(name)
		for agg in [1,10]:
			print(agg)
			for entropic_factor in [True,False]:
				print(entropic_factor)
				Compare_degree_distr(name,agg,entropic_factor=entropic_factor)
	#Check_weak_ind_hyp_NCTN(name,depth)

###################################################

#figure for Complenet2024 conference:
#compare weight distribution for NCTN and ECTN for min_EW3, conf16 and utah
def Fig1_2024(choice):
	depth = 3
	list_names = Get_tot_XPTN()#["conf16",'conf18','utah']
	list_marker = ['.','<','s']; list_color = ['blue','red','green']
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	fontsize = 16
	ax.set_xlabel(r"nb of occurrences ($\log_{10}$)",fontsize=fontsize)
	ax.set_ylabel(r"frequency ($\log_{10}$)",fontsize=fontsize)
	for ind,name in enumerate(list_names):
		marker = list_marker[ind%len(list_marker)]; color = list_color[ind%len(list_color)]
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(1)
		if choice=='NCTN':
			dic_CTN = motifs.Get_dic_NCTN(depth)
		elif choice=='ECTN':
			dic_CTN = motifs.Get_dic_ECTN(depth)
		ax.plot(*Raw_to_binned(Get_weight_histo(dic_CTN)),marker,color=color)#,label=name)
	#ax.legend(fontsize=fontsize)
	plt.savefig("figures/weight/"+choice+"_complenet2024_all.png")
	plt.show()

#figure for Complenet2024 conference:
#ECTN and NCTN similarity matrices btw empirical data sets
def Fig2_2024():
	list_names = Get_tot_XPTN(); depth = 3; nb_name = len(list_names)
	ECTN_data = []; NCTN_data = []
	for name in list_names:
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(1)
		ECTN_data.append(motifs.Get_dic_ECTN(depth,trunc=None))
		NCTN_data.append(motifs.Get_dic_NCTN(depth))
	print('simat')
	ECTN_simat = np.eye(nb_name)
	for i in range(nb_name):
		for j in range(i+1,nb_name):
			ECTN_simat[i,j] = Cosim(ECTN_data[i],ECTN_data[j])
			ECTN_simat[j,i] = ECTN_simat[i,j]
	NCTN_simat = np.eye(nb_name)
	for i in range(nb_name):
		for j in range(i+1,nb_name):
			NCTN_simat[i,j] = Cosim(NCTN_data[i],NCTN_data[j])
			NCTN_simat[j,i] = NCTN_simat[i,j]
	#plot the simat
	Draw_simat(ECTN_simat,list_names,14,os.path.join(PROJECT_ROOT,'conferences/Exeter2024/ECTN_simat.png'))
	Draw_simat(NCTN_simat,list_names,14,os.path.join(PROJECT_ROOT,'conferences/Exeter2024/NCTN_simat.png'))
	plt.show()

#plot the theoretical motif distr against the true one under weak and strong ind hyp
#for conf16, min_EW3 and utah (two figures, one per ind hyp)
def Fig3_2024():
	depth = 2
	fontsize = 16; list_color = ['blue','red']; list_marker = ['.','<']
	#weak ind hyp: conf16 and utah
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_xlabel('measured probability',fontsize=fontsize)
	ax.set_ylabel('estimated probability',fontsize=fontsize)
	m,M = 1,0
	for name,marker,color in zip(['conf16','utah'],list_marker,list_color):
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(1)
		X,Y = motifs.Get_weak_ind_NCTN(motifs.Get_dic_NCTN(depth),depth)
		ax.plot(X,Y,marker,color=color,label=name)
		m = min(np.min(X),m)
		M = max(np.max(X),M)
	ax.plot([m,M],[m,M],'--',color='orange')
	ax.legend(fontsize=fontsize)
	plt.savefig(os.path.join(PROJECT_ROOT,'conferences/Exeter2024/weak_ind_hyp.png'))
	#strong ind hyp: conf16 and min_EW3
	fig,ax = plt.subplots(constrained_layout=True)
	ax.set_xlabel('measured probability',fontsize=fontsize)
	ax.set_ylabel('estimated probability',fontsize=fontsize)
	m,M = 1,0
	for name,marker,color in zip(['conf16','min_EW3'],list_marker,list_color):
		print(name)
		motifs = Motifs_tp(name)
		motifs.Get_TN(1)
		X,Y = motifs.Get_strong_ind_NCTN(motifs.Get_dic_NCTN(depth),depth)
		ax.plot(X,Y,marker,color=color,label=name)
		m = min(np.min(X),m)
		M = max(np.max(X),M)
	ax.plot([m,M],[m,M],'--',color='orange')
	ax.legend(fontsize=fontsize)
	plt.savefig(os.path.join(PROJECT_ROOT,'conferences/Exeter2024/strong_ind_hyp.png'))

#name = "min_EW3_RS"; depth = 2; nb_motifs = 20
#Compare_first_NCTN_to_ind_hyp(name,depth,nb_motifs)
'''
#model parameters
N,duree,frac = 138,8000,0.79
#analysis parameters
e_r = 2e-2
e_c = frac*N/duree
P_01 = (e_c/e_r)*(e_r + (1-e_r)**2/6)
P_11 = (e_c/e_r)*(1-e_r)*(1+e_r/2)/3
lamb = 1/(math.exp(2*P_01+P_11)-1)
Compare_NCTN_to_theory(P_01,P_11,lamb)
'''
