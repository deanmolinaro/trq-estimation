from typing import Type
from dataclasses import dataclass
import time
import csv
import sys
import importlib
import argparse
import numpy as np


FROM_15BIT_TO_4G = 4 / (2 ** 15)
FROM_15BIT_TO_1000DEGS = 1000 / (2 ** 15)
FROM_G_TO_MS2 = 9.81
FROM_DEGS_TO_RADS = np.pi / 180.


def exo_inputs_default():
    return \
        {
            'hip_sagittal_l': {'IDX': 0, 'CONV': 1},
            'hip_sagittal_r': {'IDX': 1, 'CONV': 1},

            'd_hip_sagittal_l': {'IDX': 2, 'CONV': 1},
            'd_hip_sagittal_r': {'IDX': 3, 'CONV': 1},

            'pelvis_gyro_x': {'IDX': 4, 'CONV': FROM_15BIT_TO_1000DEGS * FROM_DEGS_TO_RADS},
            'pelvis_gyro_y': {'IDX': 5, 'CONV': FROM_15BIT_TO_1000DEGS * FROM_DEGS_TO_RADS},
            'pelvis_gyro_z': {'IDX': 6, 'CONV': FROM_15BIT_TO_1000DEGS * FROM_DEGS_TO_RADS},

            'pelvis_accel_x': {'IDX': 7, 'CONV': FROM_15BIT_TO_4G * FROM_G_TO_MS2},
            'pelvis_accel_y': {'IDX': 8, 'CONV': FROM_15BIT_TO_4G * FROM_G_TO_MS2},
            'pelvis_accel_z': {'IDX': 9, 'CONV': FROM_15BIT_TO_4G * FROM_G_TO_MS2},

            'thigh_r_accel_x': {'IDX': 10, 'CONV': FROM_15BIT_TO_4G * FROM_G_TO_MS2},
            'thigh_r_accel_y': {'IDX': 11, 'CONV': FROM_15BIT_TO_4G * FROM_G_TO_MS2},
            'thigh_r_accel_z': {'IDX': 12, 'CONV': FROM_15BIT_TO_4G * FROM_G_TO_MS2},

            'thigh_r_gyro_x': {'IDX': 13, 'CONV': FROM_15BIT_TO_1000DEGS * FROM_DEGS_TO_RADS},
            'thigh_r_gyro_y': {'IDX': 14, 'CONV': FROM_15BIT_TO_1000DEGS * FROM_DEGS_TO_RADS},
            'thigh_r_gyro_z': {'IDX': 15, 'CONV': FROM_15BIT_TO_1000DEGS * FROM_DEGS_TO_RADS},

            'thigh_l_accel_x': {'IDX': 16, 'CONV': FROM_15BIT_TO_4G * FROM_G_TO_MS2},
            'thigh_l_accel_y': {'IDX': 17, 'CONV': FROM_15BIT_TO_4G * FROM_G_TO_MS2},
            'thigh_l_accel_z': {'IDX': 18, 'CONV': FROM_15BIT_TO_4G * FROM_G_TO_MS2},

            'thigh_l_gyro_x': {'IDX': 19, 'CONV': FROM_15BIT_TO_1000DEGS * FROM_DEGS_TO_RADS},
            'thigh_l_gyro_y': {'IDX': 20, 'CONV': FROM_15BIT_TO_1000DEGS * FROM_DEGS_TO_RADS},
            'thigh_l_gyro_z': {'IDX': 21, 'CONV': FROM_15BIT_TO_1000DEGS * FROM_DEGS_TO_RADS},

            'exo_time': {'IDX': 22, 'CONV': 1}
        }


def model_inputs_outputs_default():
    return \
        {
            'trq_l': \
            {
                'IDX': 1,
                'CONV': -1,
                'INPUTS':
                {
                    'd_hip_sagittal_l': {'IDX': 0, 'CONV': 1},
                    'hip_sagittal_l': {'IDX': 1, 'CONV': 1},
                    'pelvis_accel_x': {'IDX': 2, 'CONV': 1},
                    'pelvis_accel_y': {'IDX': 3, 'CONV': -1},
                    'pelvis_accel_z': {'IDX': 4, 'CONV': 1},
                    'pelvis_gyro_x': {'IDX': 5, 'CONV': -1},
                    'pelvis_gyro_y': {'IDX': 6, 'CONV': 1},
                    'pelvis_gyro_z': {'IDX': 7, 'CONV': -1},
                    'thigh_l_accel_x': {'IDX': 8, 'CONV': 1},
                    'thigh_l_accel_y': {'IDX': 9, 'CONV': -1},
                    'thigh_l_accel_z': {'IDX': 10, 'CONV': 1},
                    'thigh_l_gyro_x': {'IDX': 11, 'CONV': -1},
                    'thigh_l_gyro_y': {'IDX': 12, 'CONV': 1},
                    'thigh_l_gyro_z': {'IDX': 13, 'CONV': -1},
                }
            },

            'trq_r': \
            {
                'IDX': 0,
                'CONV': -1,
                'INPUTS':
                {
                    'd_hip_sagittal_r': {'IDX': 0, 'CONV': 1},
                    'hip_sagittal_r': {'IDX': 1, 'CONV': 1},
                    'pelvis_accel_x': {'IDX': 2, 'CONV': 1},
                    'pelvis_accel_y': {'IDX': 3, 'CONV': 1},
                    'pelvis_accel_z': {'IDX': 4, 'CONV': 1},
                    'pelvis_gyro_x': {'IDX': 5, 'CONV': 1},
                    'pelvis_gyro_y': {'IDX': 6, 'CONV': 1},
                    'pelvis_gyro_z': {'IDX': 7, 'CONV': 1},
                    'thigh_r_accel_x': {'IDX': 8, 'CONV': 1},
                    'thigh_r_accel_y': {'IDX': 9, 'CONV': 1},
                    'thigh_r_accel_z': {'IDX': 10, 'CONV': 1},
                    'thigh_r_gyro_x': {'IDX': 11, 'CONV': 1},
                    'thigh_r_gyro_y': {'IDX': 12, 'CONV': 1},
                    'thigh_r_gyro_z': {'IDX': 13, 'CONV': 1},
                }
            }
        }
        

@dataclass
class ConfigurableConstants():
    '''Class that stores configuration-related constants.
    These variables serve to allow 1) loadable configurations from files in /custom_constants/, 
    2) online updating of device behavior via parameter_passers.py, and 3) to store calibration 
    details. Below are the default config constants. DO NOT MODIFY DEFAULTS. Write your own short
    script in /custom_constants/ (see default_config.py for example).
    (see )  '''
    
    # Expected Loop Rate (TODO: to be used for filtering torque estimates)
    EXO_FREQ: int = 200 # Hz

    # Torque filter parameters (TODO: to be used for filtering torque estimates)
    ORDER: int = 2
    F_CUT: int = 6  # Hz

    # Data Storage & I/O Parameters
    EXO_INPUTS: dict = exo_inputs_default()
    MODEL_INPUTS_OUTPUTS: dict = model_inputs_outputs_default()

    # Model File Path
    M_FILE: str = None  # This will prompt user to select model file
    M_DIR: str = None

    # Input flags
    BLOCK: bool = False

    # Socket
    PORT: int = 8080


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


# Thanks Max :)
def load_config(config_filename, pkg=None, block=True) -> Type[ConfigurableConstants]:
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
                                         package=pkg)

    except:
        error_str = 'Unable to find config file: ' + \
            config_filename + '.py in ' + pkg
        raise ValueError(error_str)

    config = module.config
    config.BLOCK = block
    print('Using ConfigurableConstants from:', config_filename)
    return config


def parse_args():
    # Create the parser
    my_parser = argparse.ArgumentParser()

    # Add the arguments
    my_parser.add_argument('-c', '--config', action='store', dest='config_filename',
                           type=str, required=False, default='default')
    my_parser.add_argument('-p', '--pkg', action='store', dest='pkg',
                            type=str, required=False, default='config')
    my_parser.add_argument('-b', '--enable_blocking', action='store_true', dest='block',
                            required=False)

    # Execute the parse_args() method
    return my_parser.parse_args()


def load_config_from_args():
    args = parse_args()
    config = load_config(**vars(args))
    return config