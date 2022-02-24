import numpy as np
import warnings


def delay_scale(trq_arr, t_arr, t_des, scale):
	if t_des > t_arr[-1]:
		warnings.warn(f"Desired time = {t_des} s but latest estimate is from {t_arr[-1]} s.")
		return trq_arr[-1] * scale

	elif t_des < t_arr[0]:
		warnings.warn(f"Desired time = {t_des} s but oldest estimate is from {t_arr[0]} s.")
		return trq_arr[0] * scale

	else:
		return np.interp(t_des, t_arr, trq_arr) * scale