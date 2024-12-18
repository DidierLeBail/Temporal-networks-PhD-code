import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(PROJECT_ROOT)

'''
build the relevant folder arborescence to store realizations of observables for the models and reference datasets
'''
if __name__ == '__main__':
	ADM_DIR = os.path.join(PROJECT_ROOT,'data/ADM')
	folders_to_build = ['distribution','point','vector']

	list_refs = []
	REF_DIR = os.path.join(ADM_DIR,'ref_data')
	for dir_name in os.listdir(REF_DIR):
		if not '.' in dir_name:
			list_refs.append(dir_name)

	list_models = []
	MOD_DIR = os.path.join(ADM_DIR,'models')
	for dir_name in os.listdir(MOD_DIR):
		if not '.' in dir_name:
			list_models.append(dir_name)

	for ref_name in list_refs:
		for folder_name in folders_to_build:
			os.mkdir(os.path.join(REF_DIR,ref_name+'/'+folder_name))
			for model in list_models:
				os.mkdir(os.path.join(MOD_DIR,model+'/'+ref_name+'/'+folder_name))
