import numpy as np


class HipV4(object):
	def __init__(self, h, w):
		self.data = np.zeros((h, w))
		self.height = h

		self.S_COPROC_TIME = slice(0, 1)
		self.S_EXO_TIME = slice(23, 24)
		self.S_HIP_SAGITTAL_L = slice(3, 4)
		self.S_HIP_SAGITTAL_R = slice(1, 2)
		self.S_D_HIP_SAGITTAL_L = slice(4, 5)
		self.S_D_HIP_SAGITTAL_R = slice(2, 3)
		self.S_PELVIS_ACCEL = slice(8, 11)
		self.S_PELVIS_GYRO = slice(5, 8)
		self.S_THIGH_L_ACCEL = slice(17, 20)
		self.S_THIGH_L_GYRO = slice(20, 23)
		self.S_THIGH_R_ACCEL = slice(11, 14)
		self.S_THIGH_R_GYRO = slice(14, 17)

	def update(self, new_data):
		if new_data.shape[1] != self.data.shape[1]:
			print('New data wrong size.')
			return -1

		self.data = np.append(self.data, new_data, axis=0)
		if self.data.shape[0] > self.height:
			print('Fell behind.')
			self.data = self.data[-self.height:, :]
		return 1

	def get_data(self, s):
		return self.data[:, s]

	def get_thigh_r_accel(self):
		return self.get_data(self.S_THIGH_R_ACCEL)

	def get_thigh_r_gyro(self):
		return self.data[:]