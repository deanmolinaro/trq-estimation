import torch
from rtmodels import ModelRT
# from biomechdata_orig import TCN
from biomechdata import TCN
from copy import deepcopy
import numpy as np
import time
import pandas as pd

def main():
	device = torch.device('cpu')
	cpu = torch.device('cpu')

	print('Loading trt model.')
	model_trt = ModelRT(m_dir='models')

	m_file_trt = model_trt.m_filepath
	m_file_torch = m_file_trt.replace('.trt', '.tar')

	# m_file_torch = 'R_S1-14-187_AB05.tar'

	print('Loading torch model.')
	model_dict = torch.load(m_file_torch, map_location=device)
	state_dict = deepcopy(model_dict['state_dict'])
	del model_dict['state_dict']
	model_torch = TCN(**model_dict)
	model_torch.load_state_dict(state_dict)
	model_torch.eval()

	# input_size = model_dict['input_size']
	input_size = model_trt.input_shape
	eff_hist = model_dict['eff_hist']
	ws = eff_hist + 1

	start_time = time.perf_counter()
	model_torch(torch.randn(input_size, device=device))
	print(time.perf_counter() - start_time)

	# d_file = 'Data/AB05_LG_BT_1_3_r_1.csv'
	d_file = 'data/test9_step.csv'
	label_name = ['hip_flexion_r_moment']
	print(f'Loading {d_file}.')
	col_headers = ['thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z', 'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z',\
		'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', 'd_hip_sagittal_r_filt', 'hip_sagittal_r']
	# col_headers = ['thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z', 'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z',\
	# 	'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', 'hip_sagittal_r', 'd_hip_sagittal_r_filt']
	data_pd = pd.read_csv(d_file, index_col=None)

	# data_pd = pd.read_csv('log/test8_input.csv', index_col=None, header=None, skiprows=1)
	# print(data_pd.shape)
	# print(data_pd.iloc[:, 1:].shape)
	# print(data_pd.iloc[:,1:].as_matrix().shape)
	# exit()
	# data_trt = np.ascontiguousarray(data_pd.iloc[:, 1:].as_matrix().transpose().reshape(1, 14, 187), dtype=np.float32)
	# print(data_trt.shape)
	# print(model_trt.predict(data_trt))

	# # print(data_trt[0, 1, :])
	# print(data_pd.iloc[0, 1:])

	# exit()

	label = data_pd[label_name].as_matrix().astype('float32').reshape(-1, len(label_name))
	# start_idx = np.where(~np.isnan(label))[0][0]
	# end_idx = np.where(~np.isnan(label))[0][-1]

	start_idx = ws-1
	end_idx = label.shape[0] - 1

	# if start_idx < ws-1:
	# 	start_idx = ws-1

	label = label[start_idx:end_idx+1].reshape(-1, len(label_name))

	data_pd = data_pd[col_headers]
	# data_trt = data_pd.as_matrix().astype('float32')
	data_trt = data_pd.as_matrix().astype('float32').transpose().reshape(1, input_size[1], -1)
	data_torch = torch.from_numpy(data_trt).to(device)

	print(f'Testing torch model: {m_file_torch}.')
	start_time = time.perf_counter()
	y_torch= np.empty((0, 1))
	with torch.no_grad():
		for i in range(start_idx+1, end_idx+2):
			# model_input = data_torch[i-ws:i, :].transpose(0, 1).reshape(1, input_size, ws)
			model_input = data_torch[:, :, i-ws:i].reshape(input_size) # Use reshape here to make sure to keep the first dimension if batch size == 1
			# print(i-ws, ws, data_torch[i-ws:i, :].shape, model_input.shape)
			y_out = model_torch(model_input).to(cpu)
			y_torch= np.append(y_torch, y_out.numpy().reshape(1, 1), axis=0)
	print(f'Done: {time.perf_counter() - start_time} s')

	# exit()

	print(f'Testing trt model: {m_file_trt}.')
	model_trt.test_model(num_tests=5, verbose=True)
	start_time = time.perf_counter()
	y_trt = np.empty((0,1))
	for i in range(start_idx+1, end_idx+2):
		# model_input = np.ascontiguousarray(data_trt[i-ws:i, :].transpose().reshape(1, input_size, ws), dtype=np.float32)
		model_input = np.ascontiguousarray(data_trt[:, :, i-ws:i].reshape(input_size), dtype=np.float32)
		y_out = model_trt.predict(model_input)
		y_trt = np.append(y_trt, y_out.reshape(1, 1), axis=0)
	print(f'Done: {time.perf_counter() - start_time} s')

	print(label.shape)
	print(y_torch.shape)
	print(y_trt.shape)

	y = np.concatenate((label, y_torch, y_trt), axis=1)
	print(y)

	rmse_torch = np.sqrt(np.mean((label-y_torch)**2))
	rmse_trt = np.sqrt(np.mean((label-y_trt)**2))

	print(f'Torch RMSE={rmse_torch}, TRT RMSE={rmse_trt}')
	
	print('Done')

	for i in range(y.shape[0]):
		print(f"{y[i,0]},{y[i,1]},{y[i,2]}")

	print(y[1475:1477, :])

if __name__=="__main__":
	main()