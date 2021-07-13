import torch
import torch.onnx
# import biomechdata_orig as biomechdata
import biomechdata
import argparse
import subprocess

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from rtmodels import ModelRT
from os import listdir, getcwd


def from_menu(m_dir=None):
	file_names = [f for f in listdir(m_dir) if '.tar' in f]

	print()
	for i, file_name in enumerate(file_names):
		print(f"[{i}] {file_name}")

	while 1:
		file_select = int(input('Please select model to convert: '))
		if file_select < len(file_names): break

	m_file = m_dir + '/' + file_names[file_select] if m_dir else file_names[file_select]

	return m_file

def load_model(t, m):
	device = torch.device('cpu')
	model_dict = torch.load(m, map_location=device)
	state_dict = model_dict['state_dict']
	del model_dict['state_dict']
	model = getattr(biomechdata, t)(**model_dict)
	model.load_state_dict(state_dict)
	model.eval()
	return model, model_dict

def to_trt(t='', m='', m_dir=''):
	m_dir = m_dir if any(m_dir) else getcwd() + '/models'
	if not any(m):
		m = from_menu(m_dir)
	m_onnx = m.replace('.tar', '.onnx')
	m_trt = m.replace('.tar', '.trt')

	print(f'Loading {m} as {t}...')
	model, model_dict = load_model(t, m)
	input_shape = (1, model_dict['input_size'], model_dict['eff_hist']+1)
	print('Done.')

	print(f'Converting {m} to {m_onnx}...')
	model_input = torch.randn(input_shape)
	torch.onnx.export(model, model_input, m.replace('.tar', '.onnx'))
	print('Done.')

	print(f'Converting {m_onnx} to {m_trt}...')
	cmd = ['trtexec', '--onnx=' + m_onnx, '--saveEngine=' + m_trt, '--explicitBatch']
	subprocess.call(cmd)
	print('Done.')

	print(f'Loading {m_trt}...')
	model = ModelRT(m_trt)
	print('Done.')

	print(f'Testing {m_trt}...')
	model.test_model(num_tests=10, verbose=True)
	print('Done.')

if __name__=="__main__":
	# Input argument parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', type=str, default='TCN')
	parser.add_argument('-m', type=str, default='')

	# Parse input argument
	args = parser.parse_args()
	args_attributes = [att for att in dir(args) if '__' not in att and '_' != att[0]]
	args_dict = {att: getattr(args, att) for att in args_attributes}

	# Convert model file to .trt
	to_trt(**args_dict)
