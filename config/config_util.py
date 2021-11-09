from typing import Type
from dataclasses import dataclass
import time
import csv
import sys
import importlib
import argparse
import numpy as np


def exo_data_params():
    return {
        'COPROC_TIME': {'S': slice(0, 1), 'C': 1.0},  # timestamp from coprocessor
        'EXO_TIME': {'S': slice(23, 24), 'C': 1.0},   # timestamp from exo
        'HIP_SAGITTAL_L': {'S': slice(1, 2), 'C': 1.0}   # left hip encoder (rad)
        'HIP_SAGITTAL_R': {'S': slice(2, 3), 'C': 1.0}   # right hip encoder (rad)
        'D_HIP_SAGITTAL_L': {'S': slice(3, 4), 'C': 1.0}   # left hip encoder velocity (rad)
        'D_HIP_SAGITTAL_R': {'S': slice(4, 5), 'C': 1.0}   # right hip encoder velocity (rad)
        'PELVIS_ACCEL': {'S': slice(8, 11), 'C': ((4/(2**15)) * 9.81)}  # pelvis accelerometer x, y, z (m/s^2)
        'PELVIS_GYRO': {'S': slice(5, 8), 'C': ((1000/(2**15)) * (np.pi / 180.))}  # pelvis gyroscope x, y, z (rad/s)
        'THIGH_L_ACCEL': {'S': slice(17, 20), 'C': (4/(2**15))}  # left thigh accelerometer x, y, z (G's)
        'THIGH_L_GYRO': {'S': slice(20, 23), 'C': (1000/(2**15))}  # left thigh gyroscope x, y, z (deg/s)
        'THIGH_R_ACCEL': {'S': slice(11, 14), 'C': (4/(2**15))}  # right thigh accelerometer x, y, z (G's)
        'THIGH_R_GYRO': {'S': slice(14, 17), 'C': (1000/(2**15))}  # right thigh gyroscope x, y, z (deg/s)
        }


@dataclass
class ConfigurableConstants():
    '''Class that stores configuration-related constants.
    These variables serve to allow 1) loadable configurations from files in /custom_constants/, 
    2) online updating of device behavior via parameter_passers.py, and 3) to store calibration 
    details. Below are the default config constants. DO NOT MODIFY DEFAULTS. Write your own short
    script in /custom_constants/ (see default_config.py for example).
    (see )  '''
    
    # Loop Rate
    TARGET_FREQ_COPROC: int = 100  # Hz
    TARGET_FREQ_EXO: int = 200  # Hz

    # I/O
    EXO_MSG_SIZE: int = 23  # l/r enc pos/vel, pelvis 6-axis IMU, l/r thigh 6-axis IMU, exo timestamp
    Q_EXO_INF_SIZE: int = 24  # adds coproc timestamp to exo msg data
    Q_TRQ_SAVE_SIZE: int = 4  # coproc timestamp, l/r trq, exo timestamp

    # Torque Mid-Level Control Parameters
    ORDER: int = 2
    F_CUT: int = 6  # Hz
    DELAY: int = 3  # Index (~10 ms per value)

    # Data Storage Parameters
    DATA_PARAMS = exo_data_params()


class ConfigSaver():
    def __init__(self, file_ID: str, config: Type[ConfigurableConstants]):
        '''file_ID is used as a custom file identifier after date.'''
        self.file_ID = file_ID
        self.config = config
        subfolder_name = 'exo_data/'
        filename = subfolder_name + \
            time.strftime("%Y%m%d_%H%M_") + file_ID + \
            '_CONFIG' + '.csv'
        self.my_file = open(filename, 'w', newline='')
        self.writer = csv.DictWriter(
            self.my_file, fieldnames=self.config.__dict__.keys())
        self.writer.writeheader()

    def write_data(self, loop_time):
        '''Writes new row of Config data to Config file.'''
        self.config.loop_time = loop_time
        self.config.actual_time = time.time()
        self.writer.writerow(self.config.__dict__)

    def close_file(self):
        if self.file_ID is not None:
            self.my_file.close()


def load_config(config_filename) -> Type[ConfigurableConstants]:
    try:
        # strip extra parts off
        config_filename = config_filename.lower()
        if config_filename.endswith('_config'):
            config_filename = config_filename[:-7]
        elif config_filename.endswith('_config.py'):
            config_filename = config_filename[:-11]
        elif config_filename.endswith('.py'):
            config_filename = config_filename[:-4]
        config_filename = config_filename + '_config'
        module = importlib.import_module('.' + config_filename,
                                         package='custom_configs')
    except:
        error_str = 'Unable to find config file: ' + \
            config_filename + ' in custom_constants'
        raise ValueError(error_str)
    config = module.config
    print('Using ConfigurableConstants from: ', config_filename)
    return config


def parse_args():
    # Create the parser
    my_parser = argparse.ArgumentParser()
    # Add the arguments
    my_parser.add_argument('-c', '--config', action='store',
                           type=str, required=False, default='default_config')
    # Execute the parse_args() method
    args = my_parser.parse_args()
    return args


def load_config_from_args():
    args = parse_args()
    config = load_config(config_filename=args.config)
    return config