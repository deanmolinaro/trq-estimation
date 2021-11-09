import numpy as np
from utils.coprocutils import interp_data


class DataManager():
	def __init__(self, data_param_dict, data_shape):
		self.data = np.zeros(data_shape)
		self.h = data_shape[0]
		self.w = data_shape[1]
		self.data_param_dict = data_param_dict

	def fifo_append(self, data_new, axis=0):
		self.data = np.append((self.data, data_new), axis=axis)[-self.h:, -self.w:]

	def get_data(self, col):
		return data[:, data_param_dict[col]['S']]

	def resample_data(self, t_name, t_total, t_len):
		t_orig = self.data[:, data_param_dict[t_name]['S']]
		t_end = t_orig[-1]
		t_start = t_end - t_total
		t_interp = np.linspace(t_start, t_end, t_len)
		return interp_data(t_interp, t_orig, self.data, dim=0)

	# def convert_units(self, data, s, c):
	# 	data[:, s] *= c

class MidLevelController():
	def __init__(self):
		pass

	def get_cmd(self):
		raise NotImplementedError

class FilterAndDelayCmd(MidLevelController):
	def __init__(self, order=2, f_cut=6, delay=3):
		self.order = order
		self.f_cut = f_cut
		self.delay = delay