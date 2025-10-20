"""
define constants and methods to draw generic plots
"""
import matplotlib.pyplot as plt

# constants
LIST_COLOR = ['red', 'black', 'blue', 'orange', 'green', 'purple', 'brown', 'pink']
LIST_MARKER = ['.', '<', 's', 'x', 'p', '^', '>', '*', 'v', '1', '2', '3', '4', 'P']

def draw_simat(simat, tick_labels, fontsize, savepath):
	fig, ax = plt.subplots(constrained_layout=True)
	img = ax.imshow(simat,cmap='gnuplot2')
	im_ratio = simat.shape[0]/simat.shape[1]
	fig.colorbar(img,ax=ax,fraction=0.05*im_ratio)
	tick_loc = list(range(len(tick_labels)))
	ax.set_xticks(tick_loc)
	ax.set_xticklabels(tick_labels,rotation=90,fontsize=fontsize)
	ax.set_yticks(tick_loc)
	ax.set_yticklabels(tick_labels,fontsize=fontsize)
	plt.savefig(savepath)
	plt.close()

# return a figure with the correct fontsize and x,y labels
def setup_plot(xlabel,ylabel,fontsize=14,title=''):
	fig,ax = plt.subplots(1,1,constrained_layout=True)
	ax.set_xlabel(xlabel,fontsize=fontsize)
	ax.set_ylabel(ylabel,fontsize=fontsize)
	if title:
		ax.set_title(title,fontsize=fontsize)
	for label in ax.get_xticklabels()+ax.get_yticklabels():
		label.set_fontsize(fontsize)
	return fig,ax

if __name__=='__main__':
	pass
