from os import path, listdir, makedirs, getcwd
import torch
import itertools
import numpy as np


def build_exp(exp_dict):
	keys, values = zip(*exp_dict.items())
	return [dict(zip(keys, v)) for v in itertools.product(*values)]

def get_exp_name(exp_dict, rename=[]):
	keys = np.sort(list(exp_dict.keys()))
	if any(rename):
		keys = [k for k in keys if k not in list(dict(rename).keys())]
		exp_name = ('_').join([('').join([r[1], str(exp_dict[r[0]])]) for r in rename])
		exp_name += '_' + ('_').join([('').join([k, str(exp_dict[k])]) for k in keys])
	else:
		exp_name = ('_').join([('').join([k, str(exp_dict[k])]) for k in keys])
	return exp_name

def get_file_names(input_dir, sub_dir='', ext=''):
	if not any(sub_dir):
		file_names = listdir(input_dir)
	else:
		if not isinstance(sub_dir, list):
			sub_dir = [sub_dir]
		file_names = [d + '/' + n for d in sub_dir for n in listdir(input_dir + '/' + d)]
	if any(ext):
		file_names = [n for n in file_names if ext in n]
	return file_names

def update_rel_dir(d, mkdir=False):
	d = [d if ':' in d or '/' == d[0] else getcwd()+'/'+d][0]
	if mkdir and not path.exists(d):
		makedirs(d)
	return d

def write_to_file(file_name, msg, write_type='a', print_msg=True):
	with open(file_name, write_type) as f:
		if isinstance(msg, list):
			for i,packet in enumerate(msg):
				f.write(str(packet))
				if i < len(msg)-1:
					f.write(',')
		else:
			f.write(str(msg))
		f.write('\n')
	if print_msg:
		print(msg)
	return True

class DeviceManager():
	def __init__(self, use_cuda=True):
		num_devices = torch.cuda.device_count()
		if num_devices == 0:
			num_devices = 1
		self.device_jobs = [0]*num_devices
		self.num_completed_jobs = 0
		self.max_device_jobs = 8
		self.use_cuda = use_cuda

	def device_available(self):
		next_device = np.argmin(self.device_jobs)
		if self.device_jobs[next_device] < self.max_device_jobs:
			return True
		else:
			return False

	def get_next_device(self):
		if self.device_available():
			next_device = np.argmin(self.device_jobs)
			self.device_jobs[next_device] += 1
			if torch.cuda.is_available() and self.use_cuda:
				return 'cuda:' + str(next_device)
			else:
				return 'cpu'
		else:
			return None

	def update_devices(self, results):
		if len(results) > self.num_completed_jobs:
			new_completed_jobs = len(results) - self.num_completed_jobs
			self.num_completed_jobs += new_completed_jobs
			for i in range(1, new_completed_jobs+1):
				if results[-i] >= 0:
					self.device_jobs[results[-i]] -= 1
					return True
				else:
					return False
		return True
