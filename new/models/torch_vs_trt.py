import torch
from rtmodels import ModelRT
from biomechdata_orig import TCN
# from biomechdata import TCN
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
	# d_file = 'data/test9_step.csv'
	d_file = 'data/btest_step.csv'
	# d_file = 'data/LG_BT_1_3_r_1.csv'
	print(f'Loading {d_file}.')
	data_pd = pd.read_csv(d_file, index_col=None)
	data_pd = data_pd.drop(range(0,data_pd.shape[0]-500))  # Only using last 500 datapoints

	# # unilateral_right
	# label_name = ['hip_flexion_r_moment']
	# col_headers = ['thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z', 'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z',\
	# 	'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', 'd_hip_sagittal_r_filt', 'hip_sagittal_r']
	# data_trt = data_pd[col_headers].as_matrix().astype('float32').transpose().reshape(1, input_size[1], -1)

	# # unilateral_left
	# label_name = ['hip_flexion_l_moment']
	# col_headers = ['thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z', 'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z',\
	# 	'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', 'hip_sagittal_r', 'd_hip_sagittal_r_filt']
	# data_trt = data_pd[col_headers].as_matrix().astype('float32').transpose().reshape(1, input_size[1], -1)

	# bilateral
	label_name = ['hip_flexion_r_moment', 'hip_flexion_l_moment']
	col_headers_r = ['d_hip_sagittal_r_filt', 'hip_sagittal_r', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', 'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z',\
		'thigh_r_accel_x', 'thigh_r_accel_y','thigh_r_accel_z', 'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z']
	col_headers_l = ['d_hip_sagittal_l_filt', 'hip_sagittal_l', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', 'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z',\
		'thigh_l_accel_x', 'thigh_l_accel_y','thigh_l_accel_z', 'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']
	cols_to_flip = ['thigh_l_accel_y', 'thigh_l_gyro_x', 'thigh_l_gyro_z', 'pelvis_accel_z', 'pelvis_gyro_x', 'pelvis_gyro_y']
	data_pd_r = data_pd[col_headers_r].as_matrix().astype('float32').transpose().reshape(1, input_size[1], -1)
	data_pd[cols_to_flip] *= -1
	data_pd_l = data_pd[col_headers_l].as_matrix().astype('float32').transpose().reshape(1, input_size[1], -1)
	data_trt = np.concatenate((data_pd_r, data_pd_l), axis=0)
	# data_trt = data_pd[col_headers_r].as_matrix().astype('float32').transpose().reshape(1, input_size[1], -1)

	
	data_torch = torch.from_numpy(data_trt).to(device)

	label = data_pd[label_name].as_matrix().astype('float32').reshape(-1, len(label_name))
	start_idx = ws-1
	end_idx = label.shape[0] - 1
	label = label[start_idx:end_idx+1, :].reshape(-1, len(label_name))  # Keep the reshape to expand 1d array to 2d array if needed

	print(f'Testing torch model: {m_file_torch}.')
	start_time = time.perf_counter()
	y_torch= np.empty((0, label.shape[1]))
	with torch.no_grad():
		for i in range(start_idx+1, end_idx+2):
			model_input = data_torch[:, :, i-ws:i].reshape(input_size) # Use reshape here to make sure to keep the first dimension if batch size == 1
			# print(i-ws, ws, data_torch[i-ws:i, :].shape, model_input.shape)
			y_out = model_torch(model_input).to(cpu)
			y_torch= np.append(y_torch, y_out.numpy().reshape(1, -1), axis=0)
	print(f'Done: {time.perf_counter() - start_time} s')

	# exit()

	print(f'Testing trt model: {m_file_trt}.')
	model_trt.test_model(num_tests=5, verbose=True)
	start_time = time.perf_counter()
	y_trt = np.empty((0, label.shape[1]))
	for i in range(start_idx+1, end_idx+2):
		model_input = np.ascontiguousarray(data_trt[:, :, i-ws:i].reshape(input_size), dtype=np.float32)
		y_out = model_trt.predict(model_input)
		y_trt = np.append(y_trt, y_out.reshape(1, -1), axis=0)
	print(f'Done: {time.perf_counter() - start_time} s')

	print(label.shape)
	print(y_torch.shape)
	print(y_trt.shape)

	y = np.concatenate((label, y_torch, y_trt), axis=1)
	# print(y)

	rmse_torch = np.sqrt(np.mean((label-y_torch)**2, axis=0))
	rmse_trt = np.sqrt(np.mean((label-y_trt)**2, axis=0))

	print(f'Torch RMSE={rmse_torch}, TRT RMSE={rmse_trt}')
	
	print('Done')	
	for i in range(label.shape[0]):
		print(f"{label[i,:]},{y_torch[i,:]},{y_trt[i,:]}")

	# w = len(label_name)
	# for i in range(y.shape[0]):
	# 	print(f"{y[i,0:1*w]},{y[i,1*w:2*w]},{y[i,2*w:3*w]}")

	# print(y[1475:1477, :])

if __name__=="__main__":
	main()