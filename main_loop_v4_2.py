import multiprocessing as mp
import time
import traceback
import numpy as np
from utils.coprocutils import get_queue_data, parse_exo_msg, interp_data, Stamp, DataLogger
from imu.imu import ExoImu, Side
from models.rtmodels import ModelRT
from server.ipserver import ServerTCP
import board
import digitalio
from os import path, makedirs, getcwd
from imu.transforms import Transform
from utils.filters import Butterworth
import config.config_util as config_util
from data_manager import DataManager


def run_loggers(loggers):
	try:
		while True:
			for log_tuple in loggers:
				logger = log_tuple[0]
				q = log_tuple[1]
				q_size = log_tuple[2]
				if not q.empty():
					data = get_queue_data(q, q_size)
					logger.write_to_file(data)

			# if not q_trq.empty():
			# 	trq_data = get_queue_data(q_trq, q_trq_size)
			# 	trq_logger.write_to_file(trq_data)

	except:
		print('Logger failed.')
		print(traceback.print_exc())
		raise

def run_server(config, m_dir, m_file, q_cmd_save, d, trq_est):
	try:
		print('Initializing torque estimator.')
		# trq_est = ModelRT(m_dir=m_dir, m_file=m_file)
		trq_est.test_model(num_tests=5, verbose=True)
		input_seq_len = trq_est.input_shape[2]
		input_seq_time = (input_seq_len - 1) / config.TARGET_FREQ_EXO
		input_cols = sorted(['HIP_SAGITTAL_*', 'D_HIP_SAGITTAL_*', 
			'PELVIS_ACCEL_X', 'PELVIS_ACCEL_Y', 'PELVIS_ACCEL_Z', 
			'PELVIS_GYRO_X', 'PELVIS_GYRO_Y', 'PELVIS_GYRO_Z', 
			'THIGH_*_ACCEL_X', 'THIGH_*_ACCEL_Y', 'THIGH_*_ACCEL_Z', 
			'THIGH_*_GYRO_X', 'THIGH_*_GYRO_Y', 'THIGH_*_GYRO_Z'])
		sides = ['L', 'R']

		print('Initializing server.')
		server = ServerTCP('', 8080)
		server.start_server()

		print('Intiializing data managers.')
		exo_data = DataManager(config.DATA_PARAMS, (int(1/config.TARGET_FREQ_EXO), config.Q_EXO_INF_SIZE))  # Store ~1 second of exo data in buffer
		exo_data_interp = DataManager(config.DATA_PARAMS, (input_seq_len, config.Q_EXO_INF_SIZE))  # Store resampled data to input to the model

		print('Intiailizing mid-level controller.')
		delay = d.value
		midlevel = FilterAndDelayCmd(order=config.ORDER, f_cut=config.F_CUT, delay=delay, fs=config.TARGET_FREQ_COPROC)

		stamp = Stamp(confiig.TARGET_FREQ_EXO)

		while True:
			# Read and parse any incoming data
			exo_msg = server.from_client()
			# exo_msg = '!1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0!1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,15.0'
			if any(exo_msg):
				exo_data_new = parse_exo_msg(exo_msg, config.EXO_MSG_SIZE)
				if not exo_data_new.any(): continue
				exo_data_new = stamp.timestamp_data(exo_data_new)
				
				for k in config.DATA_PARAMS.keys():
					s = config.DATA_PARAMS[k]['S']
					c = config.DATA_PARAMS[k]['C']
					exo_data_new[:, s] *= c

				exo_data.fifo_append(exo_data_new)
				exo_data_interp_new = exo_data.interp('COPROC_TIME', input_seq_time)
				exo_data_interp.replace_data(exo_data_interp_new)
				
				input_arr = []

				for s in sides:
					cols = [c.replace('*', s) for c in input_cols]
					input_data = exo_data_interp.get_data(cols)

					if s == 'L':
						input_data[:, cols.index('THIGH_L_ACCEL_Y')] *= -1
						input_data[:, cols.index('THIGH_L_GYRO_X')] *= -1
						input_data[:, cols.index('THIGH_L_GYRO_Z')] *= -1
						input_data[:, cols.index('PELVIS_ACCEL_Y')] *= -1
						input_data[:, cols.index('PELVIS_GYRO_X')] *= -1
						input_data[:, cols.index('PELVIS_GYRO_Z')] *= -1

					input_data = input_data.transpose()
					input_arr.append(input_data.reshape(1, input_data.shape[0], input_data.shape[1]))

				input_data = np.ascontiguousarray(np.concatenate(input_arr, axis=0)).astype('float32')
				trq = trq_est.predict(input_data)
				
				midlevel.update(trq)
				cmd = midlevel.get_cmd()

				# Convert torque estimate to message template
				send_msg = "!" + "{:.5f}".format(cmd[-1, 0]*-1) + "," + "{:.5f}".format(cmd[-1, 1]*-1) # flip l/r and mult by -1
				server.to_client(send_msg)

				q_trq_save.put_nowait(np.array((exo_data.get_data('COPROC_TIME')[-1], trq[0, 0], trq[0, 1], exo_data.get_data('EXO_TIME')[-1]), axis=1).reshape(1, -1))

				if delay != d.value:
					delay = d.value
					midlevel.update_delay(delay)
					print(f'Updated delay to {delay}!')

	except:
		print('Server failed.')
		print(traceback.print_exc())
		raise

def main():
	try:
		config = config_util.load_config_from_args()

		m_dir = getcwd() + '/models/models'
		# m_file = ModelRT.choose_model(m_dir)
		m_file = ''
		trq_est = ModelRT(m_dir=m_dir)

		# TODO: Add delay, clean up code, remove real-time transforms, check ending comma from exo message
		if not path.exists('log'): makedirs('log')
		f_name = input('Please enter log file name: ')

		print('Initializing torque estimator logger.')
		trq_logger = DataLogger('coproc_time, trq_l, trq_r, exo_time')
		trq_logger.init_file('log/' + f_name + '_trq.txt')

		print('Initializing shared queues.')
		q_trq_save = mp.Queue()
		d = mp.Value('i', config.DELAY)

		processes = []

		print('Starting logging process.')
		trq_log_tuple = (trq_logger, q_trq_save, config.Q_TRQ_SAVE_SIZE)
		loggers = (trq_log_tuple,)
		# log_process = mp.Process(target=run_loggers, args=(((trq_log_tuple,),)))
		log_process = mp.Process(target=run_loggers, args=(loggers,))
		processes.append(log_process)

		print('Staring server.')
		server_process = mp.Process(target=run_server, args=(config, m_dir, m_file, q_trq_save, d, trq_est))
		processes.append(server_process)

		[p.start() for p in processes]

		while True:
			d.value = input('Update delay (max delay of 49): ')

	except:
			# Print traceback
			traceback.print_exc()

			# Close processes
			[p.join() for p in processes]

			# Close server
			server.close()

			print('Exiting!')

			return


if __name__=="__main__":
	main()