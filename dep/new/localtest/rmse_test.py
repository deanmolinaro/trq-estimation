from tcpip import ClientTCP
import random
import time
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt


def test_jetson(client, to_server, label_df, output_idx):
	output = []
	for i in range(to_server.shape[0]):
		data = to_server[i, :]
		msg = "!" + ','.join([str(d) for d in data]) + ',' + str(i) + ','
		client.to_server(msg)
		msg = client.from_server_wait()
		output.append([float(d) for d in msg.replace('!', '').replace('&', '').split(',')])
		print(msg)
	output = np.array(output)
	output = output[200:-200, :] * -1

	est_rmse = np.sqrt(np.mean((label_df.loc[:, 'estimate'].values - output[:, output_idx]) ** 2))
	label_rmse = np.sqrt(np.mean((label_df.loc[:, 'label'].values - output[:, output_idx]) ** 2))
	print(f'Est<->Est RMSE = {est_rmse} | Label<->Est RMSE = {label_rmse}')

	plt.plot(label_df.loc[:, 'label'].values)
	plt.plot(label_df.loc[:, 'estimate'].values)
	plt.plot(output[:, output_idx])
	plt.title('Torque')
	plt.legend(('Label', 'Original Estimate', 'Jetson Estimate'))
	plt.show()


def get_data(data_path, label_path, left = False):
	data_df = pd.read_csv(data_path)
	label_df = pd.read_csv(label_path)

	print(data_df.columns)
	to_server_cols = ['hip_sagittal', 'hip_sagittal', 'd_hip_sagittal_filt', 'd_hip_sagittal_filt',
		'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', 
		'thigh_accel_x', 'thigh_accel_y', 'thigh_accel_z', 'thigh_gyro_x', 'thigh_gyro_y', 'thigh_gyro_z', 
		'thigh_accel_x', 'thigh_accel_y', 'thigh_accel_z', 'thigh_gyro_x', 'thigh_gyro_y', 'thigh_gyro_z']

	# for c in to_server_cols:
	# 	plt.plot(data_df.loc[:, c])
	# 	plt.title(c)
	# 	plt.show()

	for c in data_df.columns:
		if 'accel' in c:
			data_df.loc[:, c] *= ((2 ** 15) / 4)
		elif 'gyro' in c:
			data_df.loc[:, c] *= ((180. / np.pi) * ((2 ** 15) / 1000))

		if left:
			flip_cols = ['pelvis_accel_y', 'pelvis_gyro_x', 'pelvis_gyro_z', 'thigh_accel_y', 'thigh_gyro_x', 'thigh_gyro_z']
			if any([c == f for f in flip_cols]):
				data_df.loc[:, c] *= -1

	to_server = data_df.loc[:, to_server_cols].values
	return to_server, label_df

# to_server = np.arange(0, len(to_server_cols), 1.0)
# to_server = np.tile(to_server, (200, 1))
# for i in range(to_server.shape[1]):
# 	if 'accel' in to_server_cols[i]:
# 		to_server[:, i] *= ((2 ** 15) / 4)
# 	elif 'gyro' in to_server_cols[i]:
# 		to_server[:, i] *= ((180. / np.pi) * ((2 ** 15) / 1000))
# print(to_server)

# client = ClientTCP('localhost', 50050)
# client = ClientTCP('10.0.1.3', 50050)
client = ClientTCP('192.168.1.2', 50050)
print('Starting!')

print('Testing Right Side')
to_server_r, label_r_df = get_data('./data/data_r.csv', './data/labels_r.csv')
test_jetson(client, to_server_r, label_r_df, 0)

print('Testing Left Side')
to_server_l, label_l_df = get_data('./data/data_l.csv', './data/labels_l.csv', left = True)
test_jetson(client, to_server_l, label_l_df, 1)

exit()