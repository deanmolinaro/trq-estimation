from biomechdata_orig import TCN
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np 
from scipy.signal import find_peaks
import torch
from copy import deepcopy


class Transform():
	def __init__(self, T):
		self.T = T
		self.R = T[:3, :3]
		self.P = T[:3, -1].reshape(-1, 1)

		self.R_inv = self.R.transpose()
		self.T_inv = np.concatenate((self.R_inv, -np.matmul(self.R_inv, self.P)), axis=1)
		self.T_inv = np.concatenate((self.T_inv, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)

	def convert_for_transform(self, x):
		if x.shape[0] == 3:
			return np.concatenate((x, np.ones((1, x.shape[1]))), axis=0), True
		return x, False

	def revert_from_transform(self, x):
		x = x[:3, :]
		if x.ndim == 1:
			x = x.reshape(-1, 1)
		return x

	def rotate(self, x):
		return np.matmul(self.R, x)

	def rotate_with_inverse(self, x):
		return np.matmul(self.R_inv, x)

	def safe_transform(func):
		def wrapper(self, x):
			x, converted = self.convert_for_transform(x)
			y = func(self, x)
			if converted:
				y = self.revert_from_transform(y)
			return y
		return wrapper

	@safe_transform
	def transform(self, x):
		return np.matmul(self.T, x)

	@safe_transform
	def transform_with_inverse(self, x):
		return np.matmul(self.T_inv, x)

	def translate_accel(self, accel, ang_vel, ang_accel):
		a_r = []
		w_w_r = []
		for i in range(accel.shape[1]):
			a_r.append(np.cross(ang_accel[:, i].reshape(-1, 1), -self.P, axis=0))
			w_w_r.append(np.cross(ang_vel[:, i].reshape(-1, 1), np.cross(ang_vel[:, i].reshape(-1, 1), -self.P, axis=0), axis=0))
		return accel + np.concatenate(a_r, axis=1) + np.concatenate(w_w_r, axis=1)

def interp_data(x_new, x, y, dim=0):
	if dim==0:
		return np.concatenate([np.interp(x_new, x, y[:, i]).reshape(-1, 1) for i in range(y.shape[1])], axis=1)
	elif dim==1:
		return np.concatenate([np.interp(x_new, x, y[i, :]) for i in range(y.shape[0])], axis=0)
	else:
		print(f'Cannot interpolate along dim={dim}. Returning original data.')
		return y

data_dir = './data'
custom_file_name = 'pilot01_zi_2.txt'
orig_file_name = 'orig_exo_1.txt'

df_custom = pd.read_csv(data_dir + '/' + custom_file_name, header=None, index_col=None)
df_custom.columns = ['hip_sagittal_l', 'hip_sagittal_r', 'd_hip_sagittal_l_raw', 'd_hip_sagittal_r_raw', \
	'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', \
	'thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z', 'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z', \
	'thigh_l_accel_x', 'thigh_l_accel_y', 'thigh_l_accel_z', 'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z', \
	'time', 'none']
df_custom.loc[:, 'time'] /= 1000
for v in df_custom.columns:
	if 'pelvis_accel' in v:
		df_custom[v] *= ((4/(2**15)) * 9.81) # m/s^2
	elif 'pelvis_gyro' in v:
		df_custom[v] *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
	elif 'thigh_r_accel' in v or 'thigh_l_accel' in v:
		df_custom[v] *= (4/(2**15)) # G's
	elif 'thigh_r_gyro' in v or 'thigh_l_gyro' in v:
		df_custom[v] *= (1000/(2**15)) # deg/s

df_orig = pd.read_csv(data_dir + '/' + orig_file_name, index_col=None)
for v in df_orig.columns:
	if 'thigh_r_accel' in v or 'thigh_l_accel' in v:
		df_orig[v] /= 9.81
	elif 'thigh_r_gyro' in v or 'thigh_l_gyro' in v:
		df_orig[v] *= (180./np.pi)

pk_loc_custom = find_peaks(df_custom.loc[:, 'hip_sagittal_r'], height=0.2, prominence=0.1)[0]
df_custom['time'].iloc[:] -= df_custom['time'].iloc[pk_loc_custom[1]]

pk_loc_orig = find_peaks(df_orig.loc[:, 'hip_sagittal_r'], height=0.2, prominence=0.1)[0]
df_orig['time'].iloc[:] -= df_orig['time'].iloc[pk_loc_orig[1]]

dt = 0.005

t_interp_custom = np.arange(0, 10, dt).reshape(-1, 1)
df_custom = pd.DataFrame(interp_data(t_interp_custom, df_custom['time'].iloc[:], df_custom.values), columns=df_custom.columns)

t_interp_orig = np.arange(0, 10, dt).reshape(-1, 1)
df_orig = pd.DataFrame(interp_data(t_interp_orig, df_orig['time'].iloc[:], df_orig.values), columns=df_orig.columns)

pelvis_T = np.array([[0.1935, 0.2979, -0.9348, 0.0270], [0.9811, -0.0637, 0.1828, -0.0027], [-0.0051, -0.9525, -0.3046, -0.1205], [0, 0, 0, 1]]) # from original (incorrect) pelvis transform
# pelvis_T = np.array([[-0.1936, 0.1375, -0.9714, 0.0153], [0.9800, -0.0197, -0.1981, -0.1116], [-0.0463, -0.9903, -0.1309, -0.0269], [0, 0, 0, 1]])
# thigh_l_T = np.array([[-0.2107, 0.9750, 0.0708, -0.0393], [0.0249, -0.0670, 0.9974, 0.1443], [0.9772, 0.2119, -0.0101, -0.0935], [0, 0, 0, 1]])
thigh_r_T = np.array([[-0.1948, -0.9749, 0.1082, -0.0173], [-0.3277, -0.0393, -0.9440, -0.1253], [0.9245, -0.2193, -0.3117, -0.1241], [0, 0, 0, 1]])
thigh_l_T = np.array([[-0.1948, 0.9749, 0.1082, -0.0173], [0.3277, -0.0393, 0.9440, -0.1253], [0.9245, 0.2193, -0.3117, -0.1241], [0, 0, 0, 1]]) # adapted from thigh_r_T

pelvis_transform = Transform(pelvis_T)
thigh_r_transform = Transform(thigh_r_T)
thigh_l_transform = Transform(thigh_l_T)

pelvis_gyro = df_custom[['pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z']].values
pelvis_accel = df_custom[['pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z']].values
pelvis_gyro = pelvis_transform.rotate(pelvis_gyro.transpose())
pelvis_accel = pelvis_transform.rotate(pelvis_accel.transpose())
pelvis_ang_accel = np.diff(pelvis_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
pelvis_ang_accel = np.concatenate((pelvis_ang_accel[:, 0].reshape(-1, 1), pelvis_ang_accel), axis=1) # Repeat first value so arrays are the same length
pelvis_accel = pelvis_transform.translate_accel(pelvis_accel, pelvis_gyro, pelvis_ang_accel)
pelvis_gyro = pelvis_gyro.transpose()
pelvis_accel = pelvis_accel.transpose()
df_custom['pelvis_gyro_x'] = pelvis_gyro[:, 0]
df_custom['pelvis_gyro_y'] = pelvis_gyro[:, 1]
df_custom['pelvis_gyro_z'] = pelvis_gyro[:, 2]
df_custom['pelvis_accel_x'] = pelvis_accel[:, 0]
df_custom['pelvis_accel_y'] = pelvis_accel[:, 1]
df_custom['pelvis_accel_z'] = pelvis_accel[:, 2]

# plt.plot(df_orig['thigh_r_gyro_x'])
# plt.plot(df_orig['thigh_l_gyro_x'])
# plt.plot(df_custom['thigh_r_gyro_y'])
# plt.plot(df_custom['thigh_l_gyro_y'])

# plt.plot(df_orig['thigh_r_gyro_y'])
# plt.plot(df_orig['thigh_l_gyro_y'])
# plt.plot(df_custom['thigh_r_gyro_z'])
# plt.plot(df_custom['thigh_l_gyro_z'])

# plt.plot(df_custom['thigh_r_gyro_z'])
# plt.plot(df_custom['thigh_l_gyro_z'])
# plt.plot(df_custom['thigh_r_gyro_x'])
# plt.plot(df_custom['thigh_l_gyro_x'])
# plt.show()

# # plt.plot(df_orig['thigh_r_gyro_z'])
# plt.plot(df_orig['thigh_l_gyro_z'])
# # plt.plot(df_custom['thigh_r_gyro_x'])
# plt.plot(df_custom['thigh_l_gyro_x'])

# test_2 = df_custom[['thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']].values * (np.pi/180.)
# # test_2 = np.matmul(thigh_r_T[2, :3].reshape(1, -1)*np.array([1, 1, 1]).reshape(1, -1), test_2.transpose())[0] * (180./np.pi)
# test_2 = np.matmul(np.array([0.9245, 0.2193, -0.3117]).reshape(1, -1), test_2.transpose())[0] * (180./np.pi)
# test_1 = df_custom[['thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']].values * (np.pi/180.)
# test_1 = np.matmul(thigh_l_T[2, :3].reshape(1, -1), test_1.transpose())[0] * (180./np.pi)

# plt.plot(test_1)
# plt.plot(test_2)
# plt.show()

# plt.plot(df_orig['thigh_r_accel_x'])
# plt.plot(df_orig['thigh_l_accel_x'])
# plt.plot(df_custom['thigh_r_accel_y'])
# plt.plot(df_custom['thigh_l_accel_y'])

# plt.plot(df_orig['thigh_r_accel_y'])
# plt.plot(df_orig['thigh_l_accel_y'])
# plt.plot(df_custom['thigh_r_accel_z'])
# plt.plot(df_custom['thigh_l_accel_z'])

# plt.plot(df_orig['thigh_r_accel_z'])
# plt.plot(df_orig['thigh_l_accel_z'])
# plt.plot(df_custom['thigh_r_accel_x'])
# plt.plot(df_custom['thigh_l_accel_x'])

# plt.plot(df_orig['hip_sagittal_r'])
# plt.plot(df_orig['hip_sagittal_l'])

# plt.show()
# exit()

thigh_l_gyro = df_custom[['thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']].values * (np.pi/180.)
thigh_l_accel = df_custom[['thigh_l_accel_x', 'thigh_l_accel_y', 'thigh_l_accel_z']].values * 9.81
thigh_l_gyro = thigh_l_transform.rotate(thigh_l_gyro.transpose())
thigh_l_accel = thigh_l_transform.rotate(thigh_l_accel.transpose())
thigh_l_ang_accel = np.diff(thigh_l_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
thigh_l_ang_accel = np.concatenate((thigh_l_ang_accel[:, 0].reshape(-1, 1), thigh_l_ang_accel), axis=1) # Repeat first value so arrays are the same length
thigh_l_accel = thigh_l_transform.translate_accel(thigh_l_accel, thigh_l_gyro, thigh_l_ang_accel)
thigh_l_gyro = thigh_l_gyro.transpose() * (180./np.pi)
thigh_l_accel = thigh_l_accel.transpose() / 9.81
df_custom['thigh_l_gyro_x'] = thigh_l_gyro[:, 0]
df_custom['thigh_l_gyro_y'] = thigh_l_gyro[:, 1]
df_custom['thigh_l_gyro_z'] = thigh_l_gyro[:, 2]
df_custom['thigh_l_accel_x'] = thigh_l_accel[:, 0]
df_custom['thigh_l_accel_y'] = thigh_l_accel[:, 1]
df_custom['thigh_l_accel_z'] = thigh_l_accel[:, 2]

# thigh_r_gyro = df_custom[['thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']].values * (np.pi/180.)
thigh_r_gyro = df_custom[['thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z']].values * (np.pi/180.)
thigh_r_accel = df_custom[['thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z']].values * 9.81
thigh_r_gyro = thigh_r_transform.rotate(thigh_r_gyro.transpose())
thigh_r_accel = thigh_r_transform.rotate(thigh_r_accel.transpose())
thigh_r_ang_accel = np.diff(thigh_r_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
thigh_r_ang_accel = np.concatenate((thigh_r_ang_accel[:, 0].reshape(-1, 1), thigh_r_ang_accel), axis=1) # Repeat first value so arrays are the same length
thigh_r_accel = thigh_r_transform.translate_accel(thigh_r_accel, thigh_r_gyro, thigh_r_ang_accel)
thigh_r_gyro = thigh_r_gyro.transpose() * (180./np.pi)
thigh_r_accel = thigh_r_accel.transpose() / 9.81
df_custom['thigh_r_gyro_x'] = thigh_r_gyro[:, 0]
df_custom['thigh_r_gyro_y'] = thigh_r_gyro[:, 1]
df_custom['thigh_r_gyro_z'] = thigh_r_gyro[:, 2]
df_custom['thigh_r_accel_x'] = thigh_r_accel[:, 0]
df_custom['thigh_r_accel_y'] = thigh_r_accel[:, 1]
df_custom['thigh_r_accel_z'] = thigh_r_accel[:, 2]

# # plt.plot(df_custom['thigh_r_gyro_z'])
# plt.plot(df_custom['thigh_l_gyro_z'])
# plt.show()

plot_vars = ['hip_sagittal_l', 'hip_sagittal_r', 'd_hip_sagittal_l_raw', 'd_hip_sagittal_r_raw', \
	'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', 'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', \
	'thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z', 'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z', \
	'thigh_l_accel_x', 'thigh_l_accel_y', 'thigh_l_accel_z', 'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']
for v in plot_vars:
	plt.figure()
	plt.plot(df_orig['time'].iloc[:], df_orig[v].iloc[:])
	plt.plot(df_custom['time'].iloc[:], df_custom[v].iloc[:])
	plt.title(v)
	plt.show()
# exit()

# plt.plot(df_orig['time'], df_orig['thigh_r_gyro_z'])
# plt.plot(df_orig['time'], df_orig['thigh_l_gyro_z'])
# plt.plot(df_custom['time'], df_custom['thigh_r_gyro_z'])
# plt.plot(df_custom['time'], df_custom['thigh_l_gyro_z'])
# plt.legend(['orig-r', 'orig-l', 'custom-r', 'custom-l'])
# plt.show()

# Test model
model_dir = './models'
model_file_name = 'B_S2-14-187_AB05_2.tar'
model_file_path = model_dir + '/' + model_file_name

print('Loading torch model.')
device = torch.device('cpu')
model_dict = torch.load(model_file_path, map_location=device)
state_dict = deepcopy(model_dict['state_dict'])
del model_dict['state_dict']
model = TCN(**model_dict)
model.load_state_dict(state_dict)
model.eval()
print('Torch model loaded.')

data_r_orig = torch.tensor(df_orig[['d_hip_sagittal_r_raw', 'hip_sagittal_r', \
'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', \
'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', \
'thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z', \
'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z']].values).float()
num_feat = data_r_orig.shape[1]
data_r_orig = data_r_orig.T.reshape(1, num_feat, -1)

data_l_orig = deepcopy(df_orig[['d_hip_sagittal_l_raw', 'hip_sagittal_l', \
'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', \
'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', \
'thigh_l_accel_x', 'thigh_l_accel_y', 'thigh_l_accel_z', \
'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']])
data_l_orig['thigh_l_accel_y'] *= -1
data_l_orig['thigh_l_gyro_x'] *= -1
data_l_orig['thigh_l_gyro_z'] *= -1
data_l_orig['pelvis_accel_z'] *= -1
data_l_orig['pelvis_gyro_x'] *= -1
data_l_orig['pelvis_gyro_y'] *= -1
data_l_orig = torch.tensor(data_l_orig.values).float()
data_l_orig = data_l_orig.T.reshape(1, num_feat, -1)

with torch.no_grad():
	trq_r_orig = model(data_r_orig).numpy()
	trq_l_orig = model(data_l_orig).numpy()

plt.plot(trq_r_orig)
plt.plot(trq_l_orig)
# plt.show()

data_r_custom = torch.tensor(df_custom[['d_hip_sagittal_r_raw', 'hip_sagittal_r', \
'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', \
'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', \
'thigh_r_accel_x', 'thigh_r_accel_y', 'thigh_r_accel_z', \
'thigh_r_gyro_x', 'thigh_r_gyro_y', 'thigh_r_gyro_z']].values).float()
num_feat = data_r_custom.shape[1]
data_r_custom = data_r_custom.T.reshape(1, num_feat, -1)

data_l_custom = deepcopy(df_custom[['d_hip_sagittal_l_raw', 'hip_sagittal_l', \
'pelvis_accel_x', 'pelvis_accel_y', 'pelvis_accel_z', \
'pelvis_gyro_x', 'pelvis_gyro_y', 'pelvis_gyro_z', \
'thigh_l_accel_x', 'thigh_l_accel_y', 'thigh_l_accel_z', \
'thigh_l_gyro_x', 'thigh_l_gyro_y', 'thigh_l_gyro_z']])
data_l_custom['thigh_l_accel_y'] *= -1
data_l_custom['thigh_l_gyro_x'] *= -1
data_l_custom['thigh_l_gyro_z'] *= -1
data_l_custom['pelvis_accel_z'] *= -1
data_l_custom['pelvis_gyro_x'] *= -1
data_l_custom['pelvis_gyro_y'] *= -1
data_l_custom = torch.tensor(data_l_custom.values).float()
data_l_custom = data_l_custom.T.reshape(1, num_feat, -1)

with torch.no_grad():
	trq_r_custom = model(data_r_custom).numpy()
	trq_l_custom = model(data_l_custom).numpy()

plt.plot(trq_r_custom)
plt.plot(trq_l_custom)
plt.show()