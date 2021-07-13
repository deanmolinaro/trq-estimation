import multiprocessing
import numpy as np
import time


def get_queue_data(q, q_width, block=False):
	q_data = np.empty((0, q_width))
	if not block:
		while not q.empty():
			q_data = np.append(q_data, q.get(block=block), axis=0)
	else:
		q_data = np.append(q_data, q.get(block=block), axis=0)
	return q_data

def interp_data(x_new, x, y, dim=0):
	if dim==0:
		return np.concatenate([np.interp(x_new, x, y[:, i]).reshape(-1, 1) for i in range(y.shape[1])], axis=1)
	elif dim==1:
		return np.concatenate([np.interp(x_new, x, y[i, :]) for i in range(y.shape[0])], axis=0)
	else:
		print(f'Cannot interpolate along dim={dim}. Returning original data.')
		return y

def parse_exo_msg(exo_msg, msg_len):
	return np.array([[float(value) for value in msg.split(',')[:-1]] for msg in exo_msg.decode().split('!')[1:] if len(msg.split(',')[:-1])==msg_len]) # Ignore anything before the first ! (this should just be empty)
	# return np.array([[float(value) for value in msg.split(',')] for msg in exo_msg.split('!')[1:] if len(msg.split(','))==msg_len]) # Ignore anything before the first ! (this should just be empty)

class Stamp(object):
	def __init__(self, s_time):
		self.time_start = time.perf_counter()
		self.s_time = s_time

	def timestamp_data(self, data):
		time_curr = time.perf_counter() - self.time_start
		timestamp = np.array([time_curr + i * self.s_time for i in range(1-data.shape[0], 1)]).reshape(-1, 1)
		return np.concatenate((timestamp, data), axis=1)

class DataLogger(object):
	def __init__(self, header):
		self.header = header
		self.f_name = ''

	def init_file(self, f_name):
		self.f_name = f_name
		self.write_to_file(self.header, write_type='w', verbose=False)

	def write_to_file(self, msg, write_type='a', verbose=False):
		if not any(self.f_name):
			print('File not initialized. Call init_file before writing.')
			return

		if type(msg).__module__ == np.__name__:
			with open(self.f_name, write_type) as f:
				np.savetxt(f, msg, delimiter=',', fmt='%.8f')
		else:
			with open(self.f_name, write_type) as f:
				if isinstance(msg, list):
					for i,packet in enumerate(msg):
						f.write(str(packet))
						if i < len(msg)-1:
							f.write(',')
				else:
					f.write(str(msg))
				f.write('\n')
		if verbose:
			print(msg)
		return True


# @dataclass
# class DataContainerIMU:
# 	acc_x: float = 0
# 	acc_y: float = 0
# 	acc_z: float = 0
# 	gyro_x: float = 0
# 	gyro_y: float = 0
# 	gyro_z: float = 0

# @dataclass
# class DataContainerTrq:
# 	trq: float = 0
# 	t: float = 0