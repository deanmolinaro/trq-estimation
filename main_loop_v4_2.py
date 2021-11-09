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

# DES_S_TIME = 0.005
# Q_IMU_INF_SIZE = 13  # l/r 6-axis IMU + timestamp
# Q_IMU_SAVE_SIZE = Q_IMU_INF_SIZE + 1  # adds sync output
# 
# # EXO_MSG_SIZE = 11  # l/r enc pos/vel, pelvis 6-axis IMU, exo timestamp
# EXO_MSG_SIZE = 23 # l/r enc pos/vel, pelvis 6-axis IMU, l/r thigh 6-axis IMU, exo timestamp
# Q_EXO_INF_SIZE = EXO_MSG_SIZE + 1 # adds coproc timestamp
# Q_TRQ_SAVE_SIZE = 4 # coproc timestamp, l/r trq, exo timestamp
# # gc.enable()


def run_loggers(logger_tuple):
	try:
		while True:
			for logger_tuple in loggers:
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

def run_server(confiig, q_cmd_save):
	try:
		print('Initializing torque estimator.')
		trq_est = ModelRT(m_dir = getcwd() + '/models/models')
		trq_est.test_model(num_tests=5, verbose=True)
		input_seq_len = trq_est.input_shape[2]
		input_seq_time = (input_seq_len - 1) / config.TARGET_FREQ_EXO

		print('Initializing server.')
		server = ServerTCP('', 8080)
		server.start_server()

		print('Intiializing data manager.')
		exo_data = DataManager((config.Q_EXO_INF_SIZE, int(1/config.TARGET_FREQ_EXO)))  # Store ~1 second of exo data in buffer

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
				exo_data_interp = exo_data.interp('COPROC_TIME', input_seq_time)

				# Now get data for estimator in correct order
				# Run estimator
				# Run midlevel controller
				# Send cmd
				# Save cmd (or just raw torque est)
				# Add input to update filter params/delay
				# Consider filtering hip velocity

				coproc_time = exo_data.get_data('COPROC_TIME', )

				t_end = exo_data[-1, 0] # timestamp assigned by coprocessor NOT exo
				t_start = t_end - ((interp_len - 1) * DES_S_TIME)
				t_interp = np.linspace(t_start, t_end, interp_len)

				trq = get_queue_data(q_trq, 2, block=True)

				# Convert torque estimate to message template
				send_msg = "!" + "{:.5f}".format(trq[-1, 0]*-1) + "," + "{:.5f}".format(trq[-1, 1]*-1) # flip l/r and mult by -1
				# print(send_msg)
				server.to_client(send_msg)

				# Check that trt model makes same estimates as pytorch version
				# print(send_msg, time.perf_counter()-time_loop)

	except:
		print('Server failed.')
		print(traceback.print_exc())
		raise

def main():
	try:
		config = config_util.load_config_from_args()

		# TODO: Add delay, clean up code, remove real-time transforms, check ending comma from exo message
		if not path.exists('log'): makedirs('log')
		f_name = input('Please enter log file name: ')

		print('Initializing torque estimator logger.')
		trq_logger = DataLogger('coproc_time, trq_l, trq_r, exo_time')
		trq_logger.init_file('log/' + f_name + '_trq.txt')

		print('Initializing queues.')
		# q_exo_inf = mp.Queue()
		# q_trq_inf = mp.Queue()
		q_trq_save = mp.Queue()

		processes = []

		print('Starting logging process.')
		trq_log_tuple = (trq_logger, q_trq_save, Q_TRQ_SAVE_SIZE)
		log_process = mp.Process(target=run_loggers, args=((trq_log_tuple,)))
		processes.append(log_process)

		print('Staring server.')
		server_process = mp.Process(target=run_server, args=(config, q_trq_save,))
		processes.append(server_process)

		[p.start() for p in processes]

		# trq_filter = Butterworth(1, 10, fs=100, n_cols=2)
		trq_filter = Butterworth(2, 6, fs=100, n_cols=2)

		buf_len = int(1/DES_S_TIME) # set up data buffers to hold ~1 second of data
		exo_data = np.zeros((buf_len, Q_EXO_INF_SIZE))
		# interp_len = trq_est_r.input_shape[2]
		interp_len = trq_est.input_shape[2]

		delay = 3
		trq_d = np.zeros((delay, 2)).reshape(-1, 2)

		# count = 0
		while True:
			# count += 1
			# if count % 1000 == 0:
			# 	print(time.perf_counter()-stamp.time_start)

			if not q_exo_inf.empty():
				exo_data_new = get_queue_data(q_exo_inf, Q_EXO_INF_SIZE)

				S_COPROC_TIME = slice(0, 1)
				S_EXO_TIME = slice(23, 24)
				S_HIP_SAGITTAL_L = slice(1, 2)
				S_HIP_SAGITTAL_R = slice(2, 3)
				S_D_HIP_SAGITTAL_L = slice(3, 4)
				S_D_HIP_SAGITTAL_R = slice(4, 5)
				S_PELVIS_ACCEL = slice(8, 11)
				S_PELVIS_GYRO = slice(5, 8)
				S_THIGH_L_ACCEL = slice(17, 20)
				S_THIGH_L_GYRO = slice(20, 23)
				S_THIGH_R_ACCEL = slice(11, 14)
				S_THIGH_R_GYRO = slice(14, 17)

				exo_data_new[:, S_PELVIS_ACCEL] *= ((4/(2**15)) * 9.81)  # m/s2
				exo_data_new[:, S_PELVIS_GYRO] *= ((1000/(2**15)) * (np.pi / 180.))  # rad/s
				exo_data_new[:, S_THIGH_R_ACCEL] *= (4/(2**15))  # G's
				exo_data_new[:, S_THIGH_R_GYRO] *= (1000/(2**15))  # deg/s
				exo_data_new[:, S_THIGH_L_ACCEL] *= (4/(2**15))  # G's
				exo_data_new[:, S_THIGH_L_GYRO] *= (1000/(2**15))  # deg/s

				# # Transform IMU data to original frame
				# exo_data_new = np.concatenate((exo_data[-1, :].reshape(1, -1), exo_data_new), axis=0)
				# pelvis_accel = exo_data_new[:, S_PELVIS_ACCEL].copy()  # x, y, z
				# pelvis_gyro = exo_data_new[:, S_PELVIS_GYRO].copy()  # x, y, z
				# thigh_l_accel = exo_data_new[:, S_THIGH_L_ACCEL].copy() # x, y, z
				# thigh_l_gyro = exo_data_new[:, S_THIGH_L_GYRO].copy() # x, y, z
				# thigh_r_accel = exo_data_new[:, S_THIGH_R_ACCEL].copy() # x, y, z
				# thigh_r_gyro = exo_data_new[:, S_THIGH_R_GYRO].copy() # x, y, z

				# # print(exo_data_new[-1, S_COPROC_TIME], exo_data_new[-1, S_EXO_TIME], exo_data_new[-1, S_THIGH_L_GYRO])

				# # t = time.perf_counter()
				# # Convert units
				# pelvis_gyro *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
				# thigh_l_gyro *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
				# thigh_r_gyro *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
				# pelvis_accel *= ((4/(2**15)) * 9.81) # m/s2
				# thigh_l_accel *= ((4/(2**15)) * 9.81) # m/s2
				# thigh_r_accel *= ((4/(2**15)) * 9.81) # m/s2

				# # Transform IMU data to original frame
				# pelvis_gyro = pelvis_transform.rotate(pelvis_gyro.transpose())
				# pelvis_accel = pelvis_transform.rotate(pelvis_accel.transpose())
				# pelvis_ang_accel = np.diff(pelvis_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
				# pelvis_ang_accel = np.concatenate((pelvis_ang_accel[:, 0].reshape(-1, 1), pelvis_ang_accel), axis=1) # Repeat first value so arrays are the same length
				# pelvis_accel = pelvis_transform.translate_accel(pelvis_accel, pelvis_gyro, pelvis_ang_accel)
				# pelvis_gyro = pelvis_gyro.transpose()
				# pelvis_accel = pelvis_accel.transpose()

				# thigh_l_gyro = thigh_l_transform.rotate(thigh_l_gyro.transpose())
				# thigh_l_accel = thigh_l_transform.rotate(thigh_l_accel.transpose())
				# thigh_l_ang_accel = np.diff(thigh_l_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
				# thigh_l_ang_accel = np.concatenate((thigh_l_ang_accel[:, 0].reshape(-1, 1), thigh_l_ang_accel), axis=1) # Repeat first value so arrays are the same length
				# thigh_l_accel = thigh_l_transform.translate_accel(thigh_l_accel, thigh_l_gyro, thigh_l_ang_accel)
				# thigh_l_gyro = thigh_l_gyro.transpose()
				# thigh_l_accel = thigh_l_accel.transpose()

				# thigh_r_gyro = thigh_r_transform.rotate(thigh_r_gyro.transpose())
				# thigh_r_accel = thigh_r_transform.rotate(thigh_r_accel.transpose())
				# thigh_r_ang_accel = np.diff(thigh_r_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
				# thigh_r_ang_accel = np.concatenate((thigh_r_ang_accel[:, 0].reshape(-1, 1), thigh_r_ang_accel), axis=1) # Repeat first value so arrays are the same length
				# thigh_r_accel = thigh_r_transform.translate_accel(thigh_r_accel, thigh_r_gyro, thigh_r_ang_accel)
				# thigh_r_gyro = thigh_r_gyro.transpose()
				# thigh_r_accel = thigh_r_accel.transpose()

				# exo_data_new[:, S_PELVIS_ACCEL] = pelvis_accel  # x, y, z
				# exo_data_new[:, S_PELVIS_GYRO] = pelvis_gyro  # x, y, z
				# exo_data_new[:, S_THIGH_L_ACCEL] = thigh_l_accel  # x, y, z
				# exo_data_new[:, S_THIGH_L_GYRO] = thigh_l_gyro  # x, y, z
				# exo_data_new[:, S_THIGH_R_ACCEL] = thigh_r_accel  # x, y, z
				# exo_data_new[:, S_THIGH_R_GYRO] = thigh_r_gyro  # x, y, z
				# exo_data_new = exo_data_new[1:, :]
				# # print(time.perf_counter() - t)

				# exo_data = np.delete(exo_data, slice(exo_data_new.shape[0]), axis=0)
				exo_data = np.append(exo_data, exo_data_new, axis=0)[-buf_len:, :]

				t_end = exo_data[-1, 0] # timestamp assigned by coprocessor NOT exo
				t_start = t_end - ((interp_len - 1) * DES_S_TIME)
				t_interp = np.linspace(t_start, t_end, interp_len)

				exo_data_interp = interp_data(t_interp, exo_data[:, 0], exo_data[:, :], dim=0)

				d_hip_sagittal_l_filt = exo_data_interp[:, S_D_HIP_SAGITTAL_L].reshape(-1, 1)
				d_hip_sagittal_r_filt = exo_data_interp[:, S_D_HIP_SAGITTAL_R].reshape(-1, 1)
				hip_sagittal_l = exo_data_interp[:, S_HIP_SAGITTAL_L].reshape(-1, 1)
				hip_sagittal_r  = exo_data_interp[:, S_HIP_SAGITTAL_R].reshape(-1, 1)
				pelvis_accel = exo_data_interp[:, S_PELVIS_ACCEL]  # x, y, z
				pelvis_gyro = exo_data_interp[:, S_PELVIS_GYRO]  # x, y, z
				thigh_l_accel = exo_data_interp[:, S_THIGH_L_ACCEL]# / 9.81  # x, y, z
				thigh_l_gyro = exo_data_interp[:, S_THIGH_L_GYRO]# * (180./np.pi)  # x, y, z
				thigh_r_accel = exo_data_interp[:, S_THIGH_R_ACCEL]# / 9.81  # x, y, z
				thigh_r_gyro = exo_data_interp[:, S_THIGH_R_GYRO]# * (180./np.pi)  # x, y, z

				# print(d_hip_sagittal_l_filt[-1, :], hip_sagittal_l[-1, :], pelvis_accel[-1, :], pelvis_gyro[-1, :], thigh_l_accel[-1, :], thigh_l_gyro[-1, :])

				# Convert units
				# t = time.perf_counter()
				# pelvis_gyro *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
				# thigh_l_gyro *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
				# thigh_r_gyro *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
				# pelvis_accel *= ((4/(2**15)) * 9.81) # m/s2
				# thigh_l_accel *= ((4/(2**15)) * 9.81) # m/s2
				# thigh_r_accel *= ((4/(2**15)) * 9.81) # m/s2

				# # Transform IMU data to original frame
				# pelvis_gyro = pelvis_transform.rotate(pelvis_gyro.transpose())
				# pelvis_accel = pelvis_transform.rotate(pelvis_accel.transpose())
				# pelvis_ang_accel = np.diff(pelvis_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
				# pelvis_ang_accel = np.concatenate((pelvis_ang_accel[:, 0].reshape(-1, 1), pelvis_ang_accel), axis=1) # Repeat first value so arrays are the same length
				# pelvis_accel = pelvis_transform.translate_accel(pelvis_accel, pelvis_gyro, pelvis_ang_accel)
				# pelvis_gyro = pelvis_gyro.transpose()
				# pelvis_accel = pelvis_accel.transpose()

				# thigh_l_gyro = thigh_l_transform.rotate(thigh_l_gyro.transpose())
				# thigh_l_accel = thigh_l_transform.rotate(thigh_l_accel.transpose())
				# thigh_l_ang_accel = np.diff(thigh_l_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
				# thigh_l_ang_accel = np.concatenate((thigh_l_ang_accel[:, 0].reshape(-1, 1), thigh_l_ang_accel), axis=1) # Repeat first value so arrays are the same length
				# thigh_l_accel = thigh_l_transform.translate_accel(thigh_l_accel, thigh_l_gyro, thigh_l_ang_accel)
				# thigh_l_gyro = thigh_l_gyro.transpose()
				# thigh_l_accel = thigh_l_accel.transpose()

				# thigh_r_gyro = thigh_r_transform.rotate(thigh_r_gyro.transpose())
				# thigh_r_accel = thigh_r_transform.rotate(thigh_r_accel.transpose())
				# thigh_r_ang_accel = np.diff(thigh_r_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
				# thigh_r_ang_accel = np.concatenate((thigh_r_ang_accel[:, 0].reshape(-1, 1), thigh_r_ang_accel), axis=1) # Repeat first value so arrays are the same length
				# thigh_r_accel = thigh_r_transform.translate_accel(thigh_r_accel, thigh_r_gyro, thigh_r_ang_accel)
				# thigh_r_gyro = thigh_r_gyro.transpose()
				# thigh_r_accel = thigh_r_accel.transpose()

				# print(time.perf_counter() - t)
				# Sorted Input Order: d_hip_sagittal_filt, hip_sagittal, pelvis_accel_x, pelvis_accel_y, pelvis_accel_z, pelvis_gyro_x, pelvis_gyro_y, pelvis_gyro_z, thigh_accel_x, thigh_accel_y, thigh_accel_z, thigh_gyro_x, thigh_gyro_y, thigh_gyro_z
				model_input_r = np.concatenate((d_hip_sagittal_r_filt, hip_sagittal_r, pelvis_accel, pelvis_gyro, thigh_r_accel, thigh_r_gyro), axis=1).transpose()
								
				# Flip on Left Side: thigh_l_accel_y, thigh_l_gyro_x, thigh_l_gyro_z, pelvis_accel_z, pelvis_gyro_x, pelvis_gyro_y
				# thigh_l_accel[:, 1] *= -1  # ay
				# thigh_l_gyro[:, 0] *= -1  # gx
				# thigh_l_gyro[:, 2] *= -1 # gz
				# pelvis_accel[:, 2] *= -1  # az
				# pelvis_gyro[:, 0:2] *= -1  # gx, gy

				# Flips for hip v4 convention
				thigh_l_accel[:, 1] *= -1  # ay
				thigh_l_gyro[:, 0] *= -1  # gx
				thigh_l_gyro[:, 2] *= -1 # gz
				pelvis_accel[:, 1] *= -1  # ay
				pelvis_gyro[:, 0] *= -1  # gx
				pelvis_gyro[:, 2] *= -1 # gz
				model_input_l = np.concatenate((d_hip_sagittal_l_filt, hip_sagittal_l, pelvis_accel, pelvis_gyro, thigh_l_accel, thigh_l_gyro), axis=1).transpose()

				model_input_r = model_input_r.reshape(1, model_input_r.shape[0], model_input_r.shape[1])
				model_input_l = model_input_l.reshape(1, model_input_l.shape[0], model_input_l.shape[1])
				model_input = np.ascontiguousarray(np.concatenate((model_input_r, model_input_l), axis=0)).astype('float32')

				trq = trq_est.predict(model_input)
				trq = trq_filter.filter(trq, axis=0)  # filter torque estimates and scale to body mass and assistance percentage
				trq_r = trq[0, 0]
				trq_l = trq[0, 1]

				trq_d = np.concatenate((trq_d, np.array([trq_r, trq_l]).reshape(-1, 2)), axis=0)
				trq_d = trq_d[-delay:, :].reshape(-1, 2)

				q_trq_inf.put_nowait(np.array((trq_d[0, 0], trq_d[0, 1])).reshape(1, -1))
				# q_trq_inf.put_nowait(np.array((trq_r, trq_l)).reshape(1, -1))
				q_trq_save.put_nowait(np.array((exo_data[-1, 0], trq_l, trq_r, exo_data[-1, -1])).reshape(1, -1))

				# count += 1
				# if count == 1000:
				# 	t = np.zeros((14, 1)).reshape(-1, 1)
				# 	t[-1, 0] = exo_data[-1, 0]
				# 	q_input_save.put_nowait(np.concatenate((t, model_input_r[0, :, :]), axis=1))
				# 	print('Saving!')
				# # q_input_save.put_nowait(np.concatenate((np.array(t_end).reshape(1, 1), model_input_r[0, :, -1].reshape(1, -1)), axis=1))

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
	# x = np.array([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10], [3, 6, 9, 12, 15]])
	# t = np.array([1, 1.5, 2, 2.5, 3]).reshape(-1, 1)
	# print(x)
	# print(interp_data(t, x[:, 0], x[:, 1:], dim=0))