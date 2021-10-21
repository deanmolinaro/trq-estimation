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

DES_S_TIME = 0.005
Q_IMU_INF_SIZE = 13  # l/r 6-axis IMU + timestamp
Q_IMU_SAVE_SIZE = Q_IMU_INF_SIZE + 1  # adds sync output

# EXO_MSG_SIZE = 11  # l/r enc pos/vel, pelvis 6-axis IMU, exo timestamp
EXO_MSG_SIZE = 23 # l/r enc pos/vel, pelvis 6-axis IMU, l/r thigh 6-axis IMU, exo timestamp
Q_EXO_INF_SIZE = EXO_MSG_SIZE + 1 # adds coproc timestamp
Q_TRQ_SAVE_SIZE = 4 # coproc timestamp, l/r trq, exo timestamp
# gc.enable()


def run_imus(imu_l, imu_r, q_imu_inf, q_imu_save, stamp, sync_pin):
	sync_prev = 0
	try:
		loop_start = stamp.time_start
		while True:
			if time.perf_counter()-loop_start >= stamp.s_time:
					loop_start = time.perf_counter()

					try:
						imu_data_l = np.array(imu_l.get_imu_data()).reshape(1, -1) # TODO: Using threading here
					except KeyboardInterrupt:
						raise KeyboardInterrupt
					except:
						reboot_start = time.perf_counter()
						print('Rebooting left IMU.')
						imu_l.reboot(block=True)
						print(f'Left IMU down for {time.perf_counter()-reboot_start} s')
						continue

					try:
						imu_data_r = np.array(imu_r.get_imu_data()).reshape(1, -1) # TODO: Using threading here
					except KeyboardInterrupt:
						raise KeyboardInterrupt
					except:
						reboot_start = time.perf_counter()
						print('Rebooting right IMU')
						imu_r.reboot(block=True)
						print(f'Right IMU down for {time.perf_counter()-reboot_start} s')
						continue

					# timestamp = np.array([loop_start - time_start]).reshape(1, -1)
					# imu_data = np.concatenate((timestamp, imu_data_l, imu_data_r), axis=1)
					imu_data = stamp.timestamp_data(np.concatenate((imu_data_l, imu_data_r), axis=1))
					q_imu_inf.put_nowait(imu_data)
						
					sync = int(sync_pin.value)
					if sync != sync_prev:
						print('Last Sync = ' + str(sync))
						sync_prev = sync
					q_imu_save.put_nowait(np.concatenate((imu_data, np.array(sync).reshape(1,1)), axis=1))
	except:
		print('Imu logging failed.')
		traceback.print_exc()
		raise

def run_loggers(imu_logger, trq_logger, q_imu, q_trq, q_imu_size, q_trq_size):
	try:
		while True:
			if not q_imu.empty():
				imu_data = get_queue_data(q_imu, q_imu_size)
				imu_logger.write_to_file(imu_data)

			if not q_trq.empty():
				trq_data = get_queue_data(q_trq, q_trq_size)
				trq_logger.write_to_file(trq_data)

	except:
		print('Logger failed.')
		print(traceback.print_exc())
		raise

def run_server(server, q_exo, q_trq, exo_msg_len, stamp):
	try:
		while True:
			# time_loop = time.perf_counter()
			# Read and parse any incoming data
			exo_msg = server.from_client()
			# exo_msg = '!1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0!1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,15.0'
			if any(exo_msg):
				exo_data = parse_exo_msg(exo_msg, exo_msg_len)
				if not exo_data.any(): continue
				exo_data = stamp.timestamp_data(exo_data)
				q_exo.put_nowait(exo_data)

				trq = get_queue_data(q_trq, 2, block=True)

				# Convert torque estimate to message template
				send_msg = "!" + "{:.5f}".format(trq[-1, 1]*-1) + "," + "{:.5f}".format(trq[-1, 0]*-1) # flip l/r and mult by -1
				server.to_client(send_msg)

				# Check that trt model makes same estimates as pytorch version
				# print(send_msg, time.perf_counter()-time_loop)

	except:
		print('Server failed.')
		print(traceback.print_exc())
		raise

def main():
	try:
		print('Initializing torque estimator.')
		trq_est = ModelRT(m_dir = getcwd() + '/models/models')
		trq_est.test_model(num_tests=5, verbose=True)

		time_start = time.perf_counter()
		if not path.exists('log'): makedirs('log')
		f_name = input('Please enter log file name: ')

		# print('Initializing IMU logger.')
		# imu_logger = DataLogger('imu_time,thigh_l_accel_x,thigh_l_accel_y,thigh_l_accel_z,thigh_l_gyro_x,thigh_l_gyro_y,thigh_l_gyro_z\
		# 	,thigh_r_accel_x,thigh_r_accel_y,thigh_r_accel_z,thigh_r_gyro_x,thigh_r_gyro_y,thigh_r_gyro_z,sync')
		# imu_logger.init_file('log/' + f_name + '_imu.txt')

		print('Initializing torque estimator logger.')
		trq_logger = DataLogger('imu_time, trq_r, trq_l, exo_time')
		trq_logger.init_file('log/' + f_name + '_trq.txt')

		# print('Initializing sync pin.')
		# # Set up sync
		# sync_pin = digitalio.DigitalInOut(board.D18) # Was D4
		# sync_pin.direction = digitalio.Direction.INPUT
		# sync_pin.pull = digitalio.Pull.DOWN

		print('Initializing server.')
		server = ServerTCP('', 8080)
		server.start_server()

		print('Initializing queues.')
		# q_imu_inf = mp.Queue()
		q_exo_inf = mp.Queue()
		q_trq_inf = mp.Queue()
		# q_imu_save = mp.Queue()
		q_trq_save = mp.Queue()

		processes = []
		stamp = Stamp(DES_S_TIME)

		# print('Starting IMU process.')
		# imu_process = mp.Process(target=run_imus, args=(imu_l, imu_r, q_imu_inf, q_imu_save, stamp, sync_pin,))
		# processes.append(imu_process)

		print('Starting logging process.')
		log_process = mp.Process(target=run_loggers, args=(imu_logger, trq_logger, q_imu_save, q_trq_save, Q_IMU_SAVE_SIZE, Q_TRQ_SAVE_SIZE))
		processes.append(log_process)

		print('Staring server.')
		server_process = mp.Process(target=run_server, args=(server, q_exo_inf, q_trq_inf, EXO_MSG_SIZE, stamp,))
		processes.append(server_process)

		[p.start() for p in processes]

		# Transforms from v4 exo IMU frame to original frame
		pelvis_T = np.identity(4)
		thigh_r_T = np.identity(4)
		thigh_l_T = np.identity(4)
		# T = np.array([[0.1935, 0.2979, -0.9348, 0.0270], [0.9811, -0.0637, 0.1828, -0.0027], [-0.0051, -0.9525, -0.3046, -0.1205], [0, 0, 0, 1]])
		self.pelvis_transform = Transform(pelvis_T)
		self.thigh_r_transform = Transform(thigh_r_T)
		self.thigh_l_transform = Transform(thigh_l_T)

		buf_len = int(1/DES_S_TIME) # set up data buffers to hold ~1 second of data
		exo_data = np.zeros((buf_len, Q_EXO_INF_SIZE))
		# interp_len = trq_est_r.input_shape[2]
		interp_len = trq_est.input_shape[2]

		# count = 0
		while True:
			# count += 1
			# if count % 1000 == 0:
			# 	print(time.perf_counter()-stamp.time_start)

			if not q_exo_inf.empty():
				exo_data_new = get_queue_data(q_exo_inf, Q_EXO_INF_SIZE)
				exo_data = np.delete(exo_data, slice(exo_data_new.shape[0]), axis=0)
				exo_data = np.append(exo_data, exo_data_new, axis=0)

				t_exo = exo_data[:, 0]
				t_end = t_imu[-1]
				t_start = t_end - ((interp_len - 1) * DES_S_TIME)
				t_interp = np.linspace(t_start, t_end, interp_len)

				exo_data_interp = interp_data(t_interp, exo_data[:, 0], exo_data[:, 1:], dim=0)

				d_hip_sagittal_l_filt = exo_data_interp[:, 2].reshape(-1, 1)
				d_hip_sagittal_r_filt = exo_data_interp[:, 3].reshape(-1, 1)
				hip_sagittal_l = exo_data_interp[:, 0].reshape(-1, 1)
				hip_sagittal_r  = exo_data_interp[:, 1].reshape(-1, 1)
				pelvis_accel = exo_data_interp[:, 7:10]  # x, y, z
				pelvis_gyro = exo_data_interp[:, 4:7]  # x, y, z
				thigh_l_accel = exo_data_interp[:, 10:13] # x, y, z
				thigh_l_gyro = exo_data_interp[:, 13:16] # x, y, z
				thigh_r_accel = exo_data_interp[:, 16:19] # x, y, z
				thigh_r_gyro = exo_data_interp[:, 19:22] # x, y, z

				# Convert units
				pelvis_gyro *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
				thigh_l_gyro *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
				thigh_r_gyro *= ((1000/(2**15)) * (np.pi / 180.)) # rad/s
				pelvis_accel *= ((4/(2**15)) * 9.81) # m/s2
				thigh_l_accel *= ((4/(2**15)) * 9.81) # m/s2
				thigh_r_accel *= ((4/(2**15)) * 9.81) # m/s2

				# Transform IMU data to original frame
				pelvis_gyro = pelvis_transform.rotate(pelvis_gyro.transpose())
				pelvis_accel = pelvis_transform.rotate(pelvis_accel.transpose())
				pelvis_ang_accel = np.diff(pelvis_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
				pelvis_ang_accel = np.concatenate((pelvis_ang_accel[:, 0].reshape(-1, 1), pelvis_ang_accel), axis=1) # Repeat first value so arrays are the same length
				pelvis_accel = pelvis_transform.translate_accel(pelvis_accel, pelvis_gyro, pelvis_ang_accel)
				pelvis_gyro = pelvis_gyro.transpose()
				pelvis_accel = pelvis_accel.transpose()

				thigh_l_gyro = thigh_l_transform.rotate(thigh_l_gyro.transpose())
				thigh_l_accel = thigh_l_transform.rotate(thigh_l_accel.transpose())
				thigh_l_ang_accel = np.diff(thigh_l_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
				thigh_l_ang_accel = np.concatenate((thigh_l_ang_accel[:, 0].reshape(-1, 1), thigh_l_ang_accel), axis=1) # Repeat first value so arrays are the same length
				thigh_l_accel = thigh_l_transform.translate_accel(thigh_l_accel, thigh_l_gyro, thigh_l_ang_accel)
				thigh_l_gyro = thigh_l_gyro.transpose()
				thigh_l_accel = thigh_l_accel.transpose()

				thigh_r_gyro = thigh_r_transform.rotate(thigh_r_gyro.transpose())
				thigh_r_accel = thigh_r_transform.rotate(thigh_r_accel.transpose())
				thigh_r_ang_accel = np.diff(thigh_r_gyro, axis=1) / 0.005 # Assuming data is at 200 Hz
				thigh_r_ang_accel = np.concatenate((thigh_r_ang_accel[:, 0].reshape(-1, 1), thigh_r_ang_accel), axis=1) # Repeat first value so arrays are the same length
				thigh_r_accel = thigh_r_transform.translate_accel(thigh_r_accel, thigh_r_gyro, thigh_r_ang_accel)
				thigh_r_gyro = thigh_r_gyro.transpose()
				thigh_r_accel = thigh_r_accel.transpose()

				# Sorted Input Order: d_hip_sagittal_filt, hip_sagittal, pelvis_accel_x, pelvis_accel_y, pelvis_accel_z, pelvis_gyro_x, pelvis_gyro_y, pelvis_gyro_z, thigh_accel_x, thigh_accel_y, thigh_accel_z, thigh_gyro_x, thigh_gyro_y, thigh_gyro_z
				model_input_r = np.concatenate((d_hip_sagittal_r_filt, hip_sagittal_r, pelvis_accel, pelvis_gyro, thigh_r_accel, thigh_r_gyro), axis=1).transpose()
								
				# Flip on Left Side: thigh_l_accel_y, thigh_l_gyro_x, thigh_l_gyro_z, pelvis_accel_z, pelvis_gyro_x, pelvis_gyro_y
				thigh_l_accel[:, 1] *= -1  # ay
				thigh_l_gyro[:, 0] *= -1  # gx
				thigh_l_gyro[:, 2] *= -1 # gz
				pelvis_accel[:, 2] *= -1  # az
				pelvis_gyro[:, 0:2] *= -1  # gx, gy
				model_input_l = np.concatenate((d_hip_sagittal_l_filt, hip_sagittal_l, pelvis_accel, pelvis_gyro, thigh_l_accel, thigh_l_gyro), axis=1).transpose()

				model_input_r = model_input_r.reshape(1, model_input_r.shape[0], model_input_r.shape[1])
				model_input_l = model_input_l.reshape(1, model_input_l.shape[0], model_input_l.shape[1])
				model_input = np.ascontiguousarray(np.concatenate((model_input_r, model_input_l), axis=0)).astype('float32')

				trq = trq_est.predict(model_input)
				trq_r = trq[0, 0]
				trq_l = trq[0, 1]

				q_trq_inf.put_nowait(np.array((trq_r, trq_l)).reshape(1, -1))
				q_trq_save.put_nowait(np.array((exo_data[-1, 0], trq_r, trq_l, exo_data[-1, -1])).reshape(1, -1))

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