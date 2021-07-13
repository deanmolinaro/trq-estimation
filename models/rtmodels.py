import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from os import listdir, getcwd

class ModelRT(object):
	def __init__(self, m_file='', m_dir='', input_shape=None):
		self.m_dir = m_dir if any(m_dir) else None
		self.m_file = m_file if any(m_file) else self.choose_model()
		self.m_filepath = self.m_dir + '/' + self.m_file if self.m_dir else self.m_file
		print(f'Loading {self.m_file}.')

		self.input_shape = input_shape if input_shape else self.get_shape_from_name()
		if not self.input_shape:
			print('Could not determine model input shape.')
			return

		self.init_model()

	def choose_model(self):
		m_files = [f for f in listdir(self.m_dir) if '.trt' in f]

		print()
		for i, m_file in enumerate(m_files):
			print(f"[{i}] {m_file}")

		while 1:
			file_select = int(input('Please select a model from the menu: '))
			if file_select < len(m_files): break

		m_file = m_files[file_select]
		return m_file

	def get_shape_from_name(self):
		if '_S' in self.m_file:
			return tuple([int(s.split('_')[0]) for s in self.m_file.split('_S')[-1].split('-')])
		else:
			return None

	def init_model(self):
		# Load model and set up engine
		f = open(self.m_filepath, "rb")
		self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
		self.engine = self.runtime.deserialize_cuda_engine(f.read())
		self.context = self.engine.create_execution_context()

		# Allocate device memory
		model_input = np.ones(self.input_shape, dtype=np.float32)
		self.output = np.empty([1, 1], dtype=np.float32)
		self.d_input = cuda.mem_alloc(1 * model_input.nbytes)
		self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
		self.bindings = [int(self.d_input), int(self.d_output)]

		# Create stream to transfer data between cpu and gpu
		self.stream = cuda.Stream()

	def predict(self, model_input):
		# transfer input data to device
		cuda.memcpy_htod_async(self.d_input, model_input, self.stream)

		# execute model
		self.context.execute_async_v2(self.bindings, self.stream.handle, None)

		# transfer predictions back
		cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)

		# syncronize threads
		self.stream.synchronize()

		return self.output

	def test_model(self, num_tests=1, verbose=True):
		o_all = []
		t_all = []

		for i in range(1, num_tests+1):
			o = []
			t = []
			time_s = time.perf_counter()

			for j in range(100):
				model_input = np.random.uniform(0.0, 1.0, size=self.input_shape).astype('float32')
				output = self.predict(model_input)
				o.append(output[0])
				time_e = time.perf_counter()
				t.append(time_e - time_s)
				time_s = time_e

			if verbose: print(f'Test {i}: Avg Time: {np.mean(t) * 1000} ms +/- {np.std(t) * 1000} ms')
			t_all.append(np.mean(t))
			o_all.append(o)

		return o_all, t_all

if __name__=="__main__":
	model = ModelRT()
	model.test_model(num_tests=10, verbose=True)
