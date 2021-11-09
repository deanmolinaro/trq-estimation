import numpy as np
from utils.coprocutils import interp_data
from utils.filters import Butterworth


class DataManager():
	def __init__(self, data_param_dict, data_shape):
		self.data = np.zeros(data_shape)
		self.h = data_shape[0]
		self.w = data_shape[1]
		self.data_param_dict = self.expand_param_dict(data_param_dict)

	def expand_param_dict(self, data_param_dict):
		param_dict = {}
		for n in data_param_dict.keys():
			if 'EXP' in data_param_dict[n]:
				for i, e in enumerate(data_param_dict[n]['EXP']):
					param_dict[n+e] = {'S': data_param_dict[n]['S'], 'C': data_param_dict[n]['C'].start + i}
			else:
				param_dict[n] = data_param_dict[n]
		return param_dict

	def fifo_append(self, data_new, axis=0):
		self.data = np.append((self.data, data_new), axis=axis)[-self.h:, -self.w:]

	def get_data(self, col):
		if ~isinstance(col, list):
			col = [col]
		return np.concatenate([self.data[:, data_param_dict[c]['S']] for c in col], axis=1)
		# return data[:, data_param_dict[col]['S']]

	def replace_data(self, data):
		if data.shape[1] != self.w:
			print('Warning - New data has incorrect width!')
		self.data = data
		self.h = data.shape[0]
		self.w = data.shape[1]

	def resample_data(self, t_name, t_total, t_len):
		t_orig = self.data[:, data_param_dict[t_name]['S']]
		t_end = t_orig[-1]
		t_start = t_end - t_total
		t_interp = np.linspace(t_start, t_end, t_len)
		return interp_data(t_interp, t_orig, self.data, dim=0)


class MidLevelController():
	def __init__(self):
		pass

	def get_cmd(self):
		raise NotImplementedError


class FilterAndDelayCmd(MidLevelController):
	def __init__(self, order=2, f_cut=6, delay=3, fs=100, num_acts=2):
		self.order = order
		self.f_cut = f_cut
		self.delay = delay
		self.num_acts = num_acts
		self.filter = Butterworth(2, 6, fs=100, n_cols=num_acts)
		self.buf_len = 50
		self.buf = np.zeros((self.buf_len, num_acts))

	def update(self, data):
		data = self.filter(data, axis=0)
		self.buf = np.concatenate((self.buf, data), axis=0)[-self.buf_len:, self.num_acts]

	def update_delay(self, delay):
		self.delay = delay

	def get_cmd(self):
		return self.buf[-(self.delay+1), :].reshape(-1, self.num_acts)
