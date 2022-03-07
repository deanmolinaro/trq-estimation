# TODO: Add logging
# TODO: Add correct model - done
# TODO: Set up server - done
# TODO: Check model with saved data
# TODO: Update with asyncio
# TODO: Add filter. Also need to add something to figure out output frequency.
# TODO: Update for blocking

from config import config_util
from models import rtmodels
from tcpip.tcpip import ServerTCP
from control import midlevel
import numpy as np
from typing import Callable
import time


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
	def __init__(self, config: config_util.ConfigurableConstants):
		self.config = config

		num_outputs = len(config.MODEL_INPUTS_OUTPUTS)
		self.exo_data = ExoData(h = config.BUF_LEN)
		self.output_data = FifoBuffer(config.BUF_LEN, num_outputs + 1)  # Last column is for timestamp

		self.model = rtmodels.ModelRT(m_file = config.M_FILE, m_dir = config.M_DIR)
		self.model.test_model(num_tests = 5, verbose = True)

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

		# Add any final conversions to the torque estimate (e.g., flip flexion/extension)
		for k in config.MODEL_INPUTS_OUTPUTS.keys():
			idx = config.MODEL_INPUTS_OUTPUTS[k]['IDX']
			conv = config.MODEL_INPUTS_OUTPUTS[k]['CONV']
			model_out[idx] = model_out[idx] * conv

		model_out = np.concatenate((model_out, (timestamp,))).reshape(1, -1)  # Need to concatenate because this is returned as a tuple of (model estimates, last data timestamp)
		self.output_data.add_row(model_out)

	def get_cmd(self, idx: int) -> float:

		if not self.exo_data.is_col('control'):
			return self.output_data.get_last_row()[idx]

		controller = self.exo_data.get_last_vals_by_name(('control',))[0]
		if controller == 1:
			trq = self.output_data.get_col(idx)
			scale = self.exo_data.get_last_vals_by_name(('scale',))[0]
			delay = self.exo_data.get_last_vals_by_name(('delay',))[0]
			t = self.output_data.get_last_col()
			t_des = self.exo_data.get_last_vals_by_name(('exo_time',))[0] - delay
			cmd = midlevel.delay_scale(trq, t, t_des, scale)



		# if joint.controller.controller == snapshot_pb2.ControllerType.DEFAULT:
		# 	cmd = self.output_data.get_last_row()[idx]

		# elif joint.controller.controller == snapshot_pb2.ControllerType.DELAY_SCALE:
		# 	trq = self.output_data.get_col(idx)
		# 	t = self.output_data.get_last_col()
		# 	scale = joint.controller.delay_scale.scale
		# 	delay = joint.controller.delay_scale.delay
		# 	t_des = self.exo_data.get_last_vals_by_name(('timestamp',))[0] - delay
		# 	cmd = midlevel.delay_scale(trq, t, t_des, scale)

		# else:
		# 	warnings.warn(f"{self.get_enum_as_str(snapshot_pb2.ControllerType, joint.controller.controller)} not implemented on server!")
		# 	cmd = self.output_data.get_last_row()[idx]

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


def parse_msg(msg):
	if msg:
		return np.array([[float(value) for value in m.split(',')[:-1]] for m in msg.split('!')[1:]]) # Ignore anything before the first ! (this should just be empty)
	else:
		return None


def package_msg(msg):
	pkg_msg = ["{:.5f}".format(m) for m in msg]
	return "!" + ",".join(pkg_msg) + "&\r\n"
	

def main(config):
	print('Initializing estimator.')
	estimator = Estimator(config)

	print('Initializing server.')
	server = ServerTCP('', config.PORT)
	server.start_server()

	while True:
		# Read and parse any incoming data
		request = parse_msg(server.from_client())

		if not request is None:
			if request[-1, -1] < 1000:
				print(request[-1, -1])
			time.sleep(0.001)
			# server.to_client(f"!0,0,{request[-1,-1]}&\r\n")
			server.to_client(package_msg([0, 0, request[-1, -1]]))
			continue

			estimator.update(request)
			estimator.step()

			response = estimator.get_response(request)
			server.to_client(package_msg(response))

		else:
                        continue
                        estimator.model.predict_rand()  # TODO: Update this with asyncio


if __name__ == '__main__':
	print(f'Running {__file__}.')
	config = config_util.load_config_from_args()
	config.BUF_LEN = rtmodels.ModelRT.get_shape_from_name(config.M_FILE)[-1]
	config.MODEL_INPUTS_OUTPUTS = config.MODEL_INPUTS_OUTPUTS[0]
	config.EXO_INPUTS = config.EXO_INPUTS[0]
	main(config)

	print('Exiting.')
	exit()
