# TODO: Add logging
# TODO: Add correct model - done
# TODO: Set up server - done
# TODO: Check model with saved data
# TODO: Update with asyncio
# TODO: Add filter. Also need to add something to figure out output frequency. Clean up import.
# TODO: Update for blocking

import traceback
import multiprocessing as mp
from config import config_util
from models import rtmodels
from tcpip.tcpip import ServerTCP
from control import midlevel
import numpy as np
from typing import Callable
import time
import filters


class FifoBuffer():
	def __init__(self, h, w):
		self.h = h
		self.w = w
		self.data = np.zeros((self.h, self.w))

	def add_row_zeros(self):
		self.add_row(np.zeros((1, self.w)))

	def add_row(self, new_row):
		self.roll_rows()
		self.data[-1, :] = new_row

	def roll_rows(self):
		self.data = np.roll(self.data, -1, axis=0)

	def get_col(self, idx):
		return self.data[:, idx]

	def get_last_col(self):
		return self.get_col(-1)

	def get_row(self, idx):
		return self.data[idx, :]

	def get_last_row(self):
		return self.get_row(-1)


class ExoData(FifoBuffer):
	def __init__(self, h, w=0):
		super(ExoData, self).__init__(h, w)
		self.d_idx = {}

	def update_last_row(self, val, name):
		if name not in self.d_idx.keys():
			print('Adding {0} to ExoData!'.format(name))
			self.data = np.concatenate((self.data, np.zeros((self.h, 1))), axis=1)
			self.d_idx[name] = self.w
			self.w += 1

		idx = self.d_idx[name]
		self.data[-1, idx] = val

	def update_last_row_from_dict(self, d, add_row=False):
		if add_row:
			self.roll_rows()
		for name in d.keys():
			self.update_last_row(d[name], name)

	def update_last_row_from_msg(self, msg, add_row=False):
		if msg:
			self.update_last_row_from_dict(proto_flat_map.to_dict(msg), add_row=add_row)

	def update_from_queue(self, q, block=False):
		if block:
			self.update_last_row_from_msg(q.get(block=block), add_row=True)
		while not q.empty():
			self.update_last_row_from_msg(q.get(block=block), add_row=True)

	def get_tcn_inputs_by_name(self, col_names, nested=False):  # TODO: Update this to handle input shapes for multiple model types with muxer
		if nested:
			data = np.concatenate([self.get_tcn_inputs_by_name(c, nested=False) for c in col_names], axis=0)
		else:
			data = self.get_cols_by_name(col_names).transpose().reshape(1, len(col_names), -1)
		return np.ascontiguousarray(data).astype('float32')

	def get_tcn_inputs_by_dict(self, col_dict, nested=False):
		sorted_keys = self.get_sorted_keys(col_dict)

		if nested:
			data = np.concatenate([self.get_tcn_inputs_by_dict(col_dict[k]['INPUTS'], nested=False) for k in sorted_keys], axis=0)
		else:
			data = self.get_cols_by_dict(col_dict, sorted_keys = sorted_keys).transpose().reshape(1, len(col_dict), -1)
		return np.ascontiguousarray(data).astype('float32')

	def get_cols_by_name(self, col_names):
		idx = [self.d_idx[c] for c in col_names]
		return self.data[:, idx]

	def get_cols_by_dict(self, col_dict, sorted_keys = None):
		if not sorted_keys:
			sorted_keys = self.get_sorted_keys(col_dict)
		data = np.concatenate([self.data[:, self.d_idx[k]] * col_dict[k]['CONV'] for k in sorted_keys])
		return data

	def get_last_vals_by_name(self, col_names):
		return self.get_cols_by_name(col_names)[-1, :]

	def get_sorted_keys(self, d):
		sorted_keys = [''] * len(d)
		for k in d.keys():
			sorted_keys[d[k]['IDX']] = k
		return sorted_keys

	def is_col(self, name):
		if name in self.d_idx.keys():
			return True
		else:
			return False


class Estimator():
	def __init__(self, config: config_util.ConfigurableConstants, load_model = True):
		self.config = config

		num_outputs = len(config.MODEL_INPUTS_OUTPUTS)
		self.exo_data = ExoData(h = config.BUF_LEN)
		self.output_data = FifoBuffer(config.BUF_LEN, num_outputs + 1)  # Last column is for timestamp

		if load_model:
			self.model = rtmodels.ModelRT(m_file = config.M_FILE, m_dir = config.M_DIR)
			self.model.test_model(num_tests = 5, verbose = True)
		else:
			self.model = None

		if config.FILT:
			self.filt = filters.Butterworth(config.ORDER, config.F_CUT, fs = config.EXO_FREQ, n_cols = num_outputs)
		else:
			self.filt = None

	def update(self, request: dict) -> None:
		# Update exo data instance with latest message
		for i in range(request.shape[0]):
			parsed_request = self.parse_request(request[i, :])
			self.exo_data.update_last_row_from_dict(parsed_request, add_row = True)

	def step(self):
		# Run inference then update output_data
		model_in, timestamp = self.get_model_input()
		model_out = self.model.predict(model_in)[-1]  # TODO: Make sure model output is flattened if multiple outputs for single input
		self.update_output(model_out, timestamp)
		return (model_out, timestamp)

	def get_response(self, request: dict) -> list:
		response = []
		for i in range(len(self.config.MODEL_INPUTS_OUTPUTS)):
			response.append(self.get_cmd(i))
		response.append(self.exo_data.get_last_vals_by_name(('exo_time',))[0])  # adding latest exo timestamp to message for debugging
		return response

	def get_model_input(self) -> tuple:
		# Query data for model inference
		model_in = self.exo_data.get_tcn_inputs_by_dict(self.config.MODEL_INPUTS_OUTPUTS, nested = True)
		timestamp = self.exo_data.get_last_vals_by_name(('exo_time',))[0]
		return (model_in, timestamp)

	def update_output(self,
		model_out: np.ndarray,
		timestamp: float
		) -> None:

		# model_out = filter.filter(model_out)  # TODO: Filter torque estimates based on filter params in config
		if self.config.FILT:
			model_out = self.filt.filter(model_out.reshape(1, -1)).reshape(-1)

		# Add any final conversions to the torque estimate (e.g., flip flexion/extension)
		for k in config.MODEL_INPUTS_OUTPUTS.keys():
			idx = config.MODEL_INPUTS_OUTPUTS[k]['IDX']
			conv = config.MODEL_INPUTS_OUTPUTS[k]['CONV']
			model_out[idx] = model_out[idx] * conv

		model_out = np.concatenate((model_out, (timestamp,))).reshape(1, -1)  # Need to concatenate because this is returned as a tuple of (model estimates, last data timestamp)
		self.output_data.add_row(model_out)

	def get_cmd(self, idx: int) -> float:

		if not self.exo_data.is_col('control'):
			# trq = self.output_data.get_col(idx)
			# scale = 1
			# delay = 40
			# t = self.output_data.get_last_col()
			# t_des = self.exo_data.get_last_vals_by_name(('exo_time',))[0] - delay
			# cmd = midlevel.delay_scale(trq, t, t_des, scale)
			cmd = self.output_data.get_last_row()[idx]
			return cmd

			# return self.output_data.get_last_row()[idx]

		controller = self.exo_data.get_last_vals_by_name(('control',))[0]
		if controller == 1:
			trq = self.output_data.get_col(idx)
			scale = self.exo_data.get_last_vals_by_name(('scale',))[0]
			delay = self.exo_data.get_last_vals_by_name(('delay',))[0]
			t = self.output_data.get_last_col()
			t_des = self.exo_data.get_last_vals_by_name(('exo_time',))[0] - delay
			cmd = midlevel.delay_scale(trq, t, t_des, scale)

		return cmd

	async def spin_async(self, exit_func: Callable[[], bool] = lambda: False) -> None:
		await self.model.predict_rand_async(exit_func = exit_func)

	def parse_request(self, request):
		if len(self.config.EXO_INPUTS) != len(request): 
			print('Request length error.')
			return None

		parsed_request = {}
		for k in self.config.EXO_INPUTS.keys():
			idx = self.config.EXO_INPUTS[k]['IDX']
			conv = self.config.EXO_INPUTS[k]['CONV']
			parsed_request[k] = request[idx] * conv

		return parsed_request

	def update_output_from_q(self, q, block = False):
		if block:
			q_data = q.get(block = True)
			self.update_output(q_data[0], q_data[1])

		while not q.empty():
			q_data = q.get(block = False)
			self.update_output(q_data[0], q_data[1])


def parse_msg(msg):
	if msg:
		print('Received', msg)
		return np.array([[float(value) for value in m.split(',')[:-1]] for m in msg.split('!')[1:]]) # Ignore anything before the first ! (this should just be empty)
	else:
		return None


def package_msg(msg):
	print('Sending', msg)
	pkg_msg = ["{:.3f}".format(m) for m in msg]
	return "!" + ",".join(pkg_msg) + "&"


def parse_q(q, block = False):
	q_data = []
	if block:
		q_data.append(q.get(block = True))

	while not q.empty():
		q_data.append(q.get(block = False))

	return q_data


def run_server(config, q_exo_inf, q_trq_inf):
	print('Initializing estimator.')
	estimator = Estimator(config, load_model = False)

	print('Initializing server.')
	server = ServerTCP('', config.PORT)
	server.start_server()

	try:

		while True:
			# Read and parse any incoming data
			request = parse_msg(server.from_client())

			if request is not None:
				estimator.update(request)

				# Make sure output q is empty before blocking for newest estimate
				if config.BLOCK:
					estimator.update_output_from_q(q_trq_inf, block = False)

				q_exo_inf.put_nowait(estimator.get_model_input())
				estimator.update_output_from_q(q_trq_inf, block = config.BLOCK)
				response = estimator.get_response(request)
				server.to_client(package_msg(response))

			estimator.update_output_from_q(q_trq_inf, block = False)  # TODO: Check how blocking works. Check if sending the same message twice.

	except:
		# Close server
		print('Closing server.')
		server.close()

def main(config):
	print('Initializing queues.')
	q_exo_inf = mp.Queue()
	q_trq_inf = mp.Queue()

	print('Loading model.')
	model = rtmodels.ModelRT(m_file = config.M_FILE, m_dir = config.M_DIR)
	model.test_model(num_tests = 5, verbose = True)

	try:
		print('Staring server.')
		processes = []
		server_process = mp.Process(target=run_server, args=(config, q_exo_inf, q_trq_inf))
		processes.append(server_process)

		[p.start() for p in processes]

		while True:

			if not q_exo_inf.empty():
				q_data = parse_q(q_exo_inf, block = False)
				model_in = q_data[-1][0]
				timestamp = q_data[-1][1]
				model_out = model.predict(model_in)[-1]  # TODO: Make sure model output is flattened if multiple outputs for single input
				q_trq_inf.put_nowait((model_out, timestamp))

			else:
				model.predict_rand_breakable(exit_func = lambda: not q_exo_inf.empty)  # TODO: Update with breakable stream
	except:
		# Print traceback
		traceback.print_exc()

		# Close processes
		[p.join() for p in processes]

		print('Exiting!')

		return

if __name__ == '__main__':
	print(f'Running {__file__}.')
	config = config_util.load_config_from_args()
	config.BUF_LEN = rtmodels.ModelRT.get_shape_from_name(config.M_FILE)[-1]
	config.MODEL_INPUTS_OUTPUTS = config.MODEL_INPUTS_OUTPUTS[0]
	config.EXO_INPUTS = config.EXO_INPUTS[0]
	main(config)

	print('Exiting.')
	exit()