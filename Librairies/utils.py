import os
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from Librairies import atn
from math import sqrt

#GLOBAL VARIABLES
################################################################################################################################################

LIST_COLOR = ['red','black','blue','orange','green','purple','brown','pink']
LIST_MARKER = ['.','<','s','x','p','^','>','*','v','1','2','3','4','P']

#RANDOM GENERATOR
################################################################################################################################################

#return a number drawn from the power law with fixed bounds and exponent -gamma (P(x)=x^{-gamma})
def power_law(bounds,gamma,size=1):
	if gamma==1:
		return bounds[0]*(bounds[1]/bounds[0])**np.random.random(size)
	return bounds[0]*(1 + np.random.random(size)*((bounds[1]/bounds[0])**(1-gamma) - 1))**(1/(1-gamma))

#draw samples from tab with specified normalized weights
def draw_from_tab(tab,weights):
	x = rd.random()
	s = 0; i = 0
	while s<x:
		s += weights[i]
		i += 1
	return tab[i-1]

#HANDLE HISTOGRAMS AND DICT
################################################################################################################################################
#truncate the histogram histo such that we keep only the trunc most frequent elements starting from the starting_rank most frequent element
def truncate_histo(histo,trunc,starting_rank=0):
	most_freq = sorted(histo.keys(),key=lambda seq:histo[seq],reverse=True)[starting_rank:starting_rank+trunc]
	return {key:histo[key] for key in most_freq}

#return the cosine similarity btw the 2 histograms histo1 and histo2
def cosine_similarity(histo1,histo2):
	norm1 = sum([val**2 for val in histo1.values()])
	norm2 = sum([val**2 for val in histo2.values()])
	s = 0
	for key in set(histo1.keys()).intersection(set(histo2.keys())):
		s += histo1[key]*histo2[key]
	return s/sqrt(norm1*norm2)

#normalize an histogram written as a dictionary of arbitrary depth:
#for depth 2, after normalization, histo[key1][key2] = Proba(key2|key1)
def norm_dic_histo(histo):
	bottom_found = False; norm = 0
	for key,val in histo.items():
		if type(val)!=dict:
			bottom_found = True
			norm += val
	if bottom_found:
		return {key:val/norm for key,val in histo.items()}
	return {key:norm_dic_histo(histo[key]) for key in histo.keys()}

#if tuple_keys = (k0,k1,...,kn) then initialize to 1 the value dic[kn]...[k0]
def initialize_dic(dic,tuple_keys):
	if len(tuple_keys)==1:
		key = tuple_keys[0]
		dic[key] = 1
		return None
	key = tuple_keys[-1]
	dic[key] = {}
	return initialize_dic(dic[key],tuple_keys[:-1])

#if tuple_keys = (k0,k1,...,kn) then increase by 1 the value dic[kn]...[k0]
#and add the non-existing keys
def increase_dic(dic,tuple_keys):
	if len(tuple_keys)==1:
		key = tuple_keys[0]
		if key in dic:
			dic[key] += 1
		else:
			dic[key] = 1
		return None
	key = tuple_keys[-1]
	if key in dic:
		return increase_dic(dic[key],tuple_keys[:-1])
	return initialize_dic(dic,tuple_keys)

#return log-binned data to enhance the quality of power-law distributions
def raw_to_binned(data,nb=50):
	if len(data)==1:
		return np.log10(list(data.keys())[0]),0
	XY = list(data.items())
	XY.sort(key=lambda el:el[0])
	X,Y = zip(*XY)
	X = np.asarray(X,dtype=float); Y = np.asarray(Y,dtype=float)

	ind = 1
	cond = (abs(np.log10(X[ind]/X[ind-1]))>0.01)
	while cond:
		ind += 1
		if ind<len(X):
			cond = (abs(np.log10(X[ind]/X[ind-1]))>0.01)
		else:
			cond = False
	if ind==len(X):
		bins = np.zeros(ind)
		bins[:ind] = X[:ind]-0.5
	else:
		bins = np.zeros(ind+nb)
		bins[:ind] = X[:ind]-0.5; bins[ind:] = np.logspace(np.log10(X[ind]-0.5),np.log10(X[-1]+0.5),num=nb)
	widths = bins[1:]-bins[:-1]

	# Calculate histogram
	hist = np.histogram(X,bins=bins,weights=Y)
	res_X = []; res_Y = []
	for x,y in zip((bins[:-1]+bins[1:])/2,hist[0]/widths):
		if y>0:
			res_X.append(np.log10(x))
			res_Y.append(np.log10(y))
	return res_X,res_Y

#FIGURE
################################################################################################################################################
#return a 1D figure with the correct fontsize and x,y labels
def setup_plot(xlabel,ylabel,fontsize=14,title=''):
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.set_xlabel(xlabel,fontsize=fontsize)
	ax.set_ylabel(ylabel,fontsize=fontsize)
	if title:
		ax.set_title(title,fontsize=fontsize)
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(fontsize)
	return fig,ax

#setup a figure for matrix visualization (2D figure)
def setup_simat(simat,tick_labels,fontsize):
	fig,ax = plt.subplots(constrained_layout=True)
	img = ax.imshow(simat,cmap='gnuplot2')
	im_ratio = simat.shape[0]/simat.shape[1]
	fig.colorbar(img,ax=ax,fraction=0.05*im_ratio)
	tick_loc = list(range(len(tick_labels)))
	ax.set_xticks(tick_loc)
	ax.set_xticklabels(tick_labels,rotation=90,fontsize=fontsize)
	ax.set_yticks(tick_loc)
	ax.set_yticklabels(tick_labels,fontsize=fontsize)
	return fig,ax

#what is written here will not run at utils.py importation as a module
if __name__=='__main__':
	pass
