import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MPU9250 import MPU9250
import board
import digitalio
from enum import Enum
import time

class Side(Enum):
	LEFT = 0
	RIGHT = 1

class ExoImu(MPU9250):
	def __init__(self, side):
		if side==Side.LEFT:
			self.scl = board.SCL
			self.sda = board.SDA 
		elif side==Side.RIGHT:
			self.scl = board.SCL_1
			self.sda = board.SDA_1
		else:
			print('Unknown side. IMU unitialized.')
			return

		# Initialize IMUs
		self.init_imu()

	def init_imu(self):
		super().__init__(scl=self.scl, sda=self.sda)
		time.sleep(0.1)

		if self.scl==board.SCL and self.sda==board.SDA:
			# imu.set_accel_offset_x(-3120)
			# imu.set_accel_offset_y(2902)
			# imu.set_accel_offset_z(4096)
			# imu.set_gyro_offset_x(573)
			# imu.set_gyro_offset_y(-78)
			# imu.set_gyro_offset_z(-25)

			# imu.set_accel_offset_x(-3800) # IMU1 settings (updated before start of pilots)
			# imu.set_accel_offset_y(3500)
			# imu.set_accel_offset_z(3650)
			# imu.set_gyro_offset_x(-20)
			# imu.set_gyro_offset_y(50)
			# imu.set_gyro_offset_z(-10)

			self.set_accel_offset_x(-1550) # IMU3 settings (Updated before start of pilots)
			self.set_accel_offset_y(-2150)
			self.set_accel_offset_z(5000)
			self.set_gyro_offset_x(30)
			self.set_gyro_offset_y(5)
			self.set_gyro_offset_z(-50)

		elif self.scl==board.SCL_1 and self.sda==board.SDA_1:
			# imu.set_accel_offset_x(-3150) # IMU2 settings (old, before collection of AB02)
			# imu.set_accel_offset_y(2900)
			# imu.set_accel_offset_z(4100)
			# imu.set_gyro_offset_x(575)
			# imu.set_gyro_offset_y(-75)
			# imu.set_gyro_offset_z(-20)
			self.set_accel_offset_x(-3817) # IMU2 settings (Updated 12102020, after collection of AB02)
			self.set_accel_offset_y(3567)
			self.set_accel_offset_z(3625)
			self.set_gyro_offset_x(-20)
			self.set_gyro_offset_y(50)
			self.set_gyro_offset_z(-20)

		self.set_accel_scale(4)
		self.set_gyro_scale(1000)

	def reboot(block=False):
		if block:
			while True:
				try:
					self.reboot(block=False)
					break
				except KeyboardInterrupt:
					raise KeyboardInterrupt
				except:
					pass
		else:
			try:
				return self.init_imu()
			except:
				raise
		return


if __name__=="__main__":
	import numpy as np

	imu_l = ExoImu(Side.LEFT)
	imu_r = ExoImu(Side.RIGHT)
	t = []
	t_start = time.perf_counter()
	for i in range(1000):
		x = imu_l.get_imu_data()
		y = imu_r.get_imu_data()
		t_start_new = time.perf_counter()
		t.append(t_start_new - t_start)
		t_start = t_start_new

	print(f'Avg Time: {np.mean(t) * 1000} ms +/- {np.std(t) * 1000} ms')