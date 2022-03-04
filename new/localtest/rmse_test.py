from tcpip import ClientTCP
import random
import time
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt


data_r_df = pd.read_csv('./data/data_r.csv')
labels_r_df = pd.read_csv('./data/labels_r.csv')

print(data_r_df.columns)
to_server_cols = ['hip_sagittal', 'hip_sagittal', 'd_hip_sagittal_filt', 'd_hip_sagittal_filt',
	'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', 
	'thigh_accel_x', 'thigh_accel_y', 'thigh_accel_z', 'thigh_gyro_x', 'thigh_gyro_y', 'thigh_gyro_z', 
	'thigh_accel_x', 'thigh_accel_y', 'thigh_accel_z', 'thigh_gyro_x', 'thigh_gyro_y', 'thigh_gyro_z']

# for c in to_server_cols:
# 	plt.plot(data_r_df.loc[:, c])
# 	plt.title(c)
# 	plt.show()

for c in data_r_df.columns:
	if 'accel' in c:
		data_r_df.loc[:, c] *= ((2 ** 15) / 4)
	elif 'gyro' in c:
		data_r_df.loc[:, c] *= ((180. / np.pi) * ((2 ** 15) / 1000))

to_server = data_r_df.loc[:, to_server_cols].values

# to_server = np.arange(0, len(to_server_cols), 1.0)
# to_server = np.tile(to_server, (200, 1))
# for i in range(to_server.shape[1]):
# 	if 'accel' in to_server_cols[i]:
# 		to_server[:, i] *= ((2 ** 15) / 4)
# 	elif 'gyro' in to_server_cols[i]:
# 		to_server[:, i] *= ((180. / np.pi) * ((2 ** 15) / 1000))
# print(to_server)

client = ClientTCP('localhost', 50050)
# client = ClientTCP('192.168.1.2', 50050)
print('Starting!')

output = []
for i in range(to_server.shape[0]):
	data = to_server[i, :]
	msg = "!" + ','.join([str(d) for d in data]) + ',' + str(i) + ','
	client.to_server(msg)
	msg = client.from_server_wait()
	output.append([float(d) for d in msg.replace('!', '').replace('&', '').split(',')])
	print(msg)
output = np.array(output)
output = output[200:-200, :]

result = np.concatenate((labels_r_df.loc[:, 'label'].values.reshape(-1, 1), output[:, 0].reshape(-1, 1)), axis = 1)
print(result)
exit()

print(labels_r_df.shape)
print(output.shape)
exit()

plt.plot(labels_r_df.loc[:, 'label'].values)
plt.plot(output[:, 0])
plt.title('Right Torque')
plt.legend(('Label', 'Estimate'))
plt.show()

print(to_server)
print(to_server.values)

exit()

client = ClientTCP('192.168.1.2', 50050)
print('Starting!')
num_vals = 22

start_time = time.perf_counter()
for i in range(2000):
    message = '!' + ','.join([str(round(random.random(),3)) for i in range(num_vals)] + [str(time.time())]) + ','

    time_start = time.perf_counter()
    client.to_server(message)
    # msg = client.from_server()
    msg = client.from_server_wait()
    print(msg, time.perf_counter() - time_start)
    time.sleep(0.005)
# print(time.perf_counter()-start_time)