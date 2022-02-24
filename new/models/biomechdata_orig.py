import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from os import listdir
import pandas as pd
import numpy as np
import time
# from matplotlib import pyplot as plt
from biomechutils import get_file_names

class AbleBodyDataset2(Dataset):
	def __init__(self, input_dir, label, ft_use=[], ft_ignore=[], trials_use=[], trials_ignore=[]):
		self.input_dir = input_dir
		self.label = label
		self.ft_use = ft_use
		self.ft_ignore = ft_ignore
		self.trials_use = trials_use
		self.trials_ignore = trials_ignore
		self.trial_names = []

	def __len__(self):
		return len(self.trial_names)

	def __getitem__(self, idx):
		if isinstance(idx, list):
			trial_names = [self.trial_names[i] for i in idx]
		else:
			trial_names = self.trial_names[idx]
			if not isinstance(trial_names, list):
				trial_names = [trial_names]

		input_data = [self.read_tensor(self.input_dir + '/' + name) for name in trial_names]
		label_data = tuple([data[1] for data in input_data])
		label_data = torch.cat(label_data, dim=0)
		feature_data = tuple([data[0] for data in input_data])
		feature_data = torch.cat(feature_data, dim=0)

		return {'features': feature_data, 'labels': label_data}

	def add_trials(self, sub_dir='', ext='', search=True, include=[]):
		input_dir = self.input_dir + '/' + sub_dir
		if any(include):
			search_dirs = include
		elif search:
			search_dirs = listdir(input_dir)
		else:
			search_dirs = []
		trial_names = get_file_names(input_dir, sub_dir=search_dirs, ext=ext)
		trial_names = [sub_dir + '/' + t for t in trial_names]
		if any(self.trials_use):
			trial_names = [t for t in trial_names if any([n for n in self.trials_use if n in t])]
		if any(self.trials_ignore):
			trial_names = [t for t in trial_names if not any([n for n in self.trials_ignore if n in t])]
		self.trial_names += trial_names
		return True

	def get_boundaries(self, data, eff_hist, eff_pred, trial):
		# Find start and end of valid label data (based on padding in preprocessing)
		label_start = np.where(~np.isnan(data['labels'].numpy()))[0][0]
		label_end = np.where(~np.isnan(data['labels'].numpy()))[0][-1]

		# Check if there is enough data before start of label for the given effective history and update accordingly
			# label_start = how many data points exist before the first label for this step data.
			# eff_hist = the number of data points required as input to the TCN for the first label
			# eff_pred = the number of data points between the last input and first label for the TCN (prediction)
		if eff_hist+eff_pred > label_start:
			print(f'Warning - Starting later than padded input data. Removing {(eff_hist+eff_pred)-label_start} labels from start of {trial}.')
		label_start = max(eff_hist+eff_pred, label_start)
		feat_start = label_start-(eff_hist+eff_pred)

		# Similar check as before but now we only need to make sure there is enough room for any prediction
			# Technically this only matters for negative prediction values (estimating back in time from the input data)
		if label_end > data['labels'].shape[0] + eff_pred - 1:
			print(f"Warning - Ending earlier than padded input data. Removing {(data['labels'].shape[0] + eff_pred - 1)-label_end} labels from end of {trial}.")
		label_end = min(data['labels'].shape[0]+eff_pred-1, label_end)
		feat_end = label_end-eff_pred

		return feat_start, feat_end, label_start, label_end

	def load_trial_dict(self, eff_hist, eff_pred=0, batch_size_per_trial=1, device='cpu', verbose=False):
		trial_dict = {trial: {} for trial in self.trial_names}
		trial_list = []
		for i, trial in enumerate(self.trial_names):
			if verbose:
				print('Loading ' + trial)
			
			# Get trial data from dataset
			data = self[i]
			feat_start, feat_end, label_start, label_end = self.get_boundaries(data, eff_hist, eff_pred, trial)

			# Save input and label data to training dictionary
			trial_dict[trial]['data'] = data['features'][feat_start:feat_end+1, :].float().to(device)
			trial_dict[trial]['labels'] = data['labels'][label_start:label_end+1].float().to(device)

			# Slice trial based on batch_size_per_trial and randomize order of slices
			label_length = trial_dict[trial]['labels'].shape[0]
			num_slices = int(np.ceil(label_length / batch_size_per_trial))
			slice_order = np.arange(num_slices)
			np.random.shuffle(slice_order)
			trial_dict[trial]['slice_order'] = list(slice_order)
			trial_dict[trial]['slice_count'] = 0

			# Add trial to training trial list num_slices times
			trial_list += [trial]*num_slices

			if verbose and i==0:
				print(self.features_use)

		return trial_dict, trial_list

	def load_trial_list(self, eff_hist, eff_pred=0, device='cpu', verbose=False):
		x_list = []
		y_list = []
		y_trial_count = [] # Label for each step through the timeseries data
		count = 0
		for i, trial in enumerate(self.trial_names):
			if verbose:
				print('Loading ' + trial)

			# Get trial data from dataset
			data = self[i]
			feat_start, feat_end, label_start, label_end = self.get_boundaries(data, eff_hist, eff_pred, trial)

			# Save input and label data to test list
			x_list.append(data['features'][feat_start:feat_end+1, :].float().to(device))
			y_list.append(data['labels'][label_start:label_end+1].float().to(device))

			# Update y_trial_count with the step number for this trial for each timestep in the data
			count += 1
			y_trial_count += [count]*y_list[-1].shape[0]

		return x_list, y_list, y_trial_count

	def read_tensor(self, file_path):
		df = pd.read_csv(file_path) # Read in trial data

		# Select specified features
		features_use = list(df.columns)
		if any(self.ft_use):
			features_use = [feature for feature in features_use if any([f for f in self.ft_use if f in feature])]
		if any(self.ft_ignore):
			features_use = [feature for feature in features_use if not any([f for f in self.ft_ignore if f in feature])]
		self.features_use = features_use

		missing_ft = [f for f in self.ft_use if not any([f for feature in features_use if f in feature])]
		if any(missing_ft):
			print(f"Missing {(', ').join(missing_ft)} in input data!")

		# Convert data to pytorch tensor
		input_data = torch.tensor(df[features_use].values)
		label_data = torch.tensor([df[self.label].values,]).T

		if any([any(torch.isnan(d)) for d in input_data]):
			print('Warning - NaN in input ' + file_path)

		return input_data, label_data

class AbleBodyDataset(Dataset):
	def __init__(self, input_dir, gait_modes, ft_use='all', ft_ignore=None, label='moment', subject_use='all', subject_ignore=None, trials_ignore=None, trial_names=[None]):
		self.input_dir = input_dir
		self.gait_modes = gait_modes
		self.ft_use = ft_use
		self.ft_ignore = ft_ignore
		self.label = label

		if any(trial_names):
			self.trial_names = trial_names
		else:
			self.trial_names = [[mode + '/' + name for name in listdir(input_dir + '/' + mode)] for mode in gait_modes]
			self.trial_names = [name for sublist in self.trial_names for name in sublist if '.csv' in name]
			if subject_use != 'all' and (isinstance(subject_use, list) and 'all' not in subject_use):
				if not isinstance(subject_use, list):
					subject_use = [subject_use]
				self.trial_names = [name for name in self.trial_names if any(subject in name for subject in subject_use)]
			if subject_ignore:
				if not isinstance(subject_ignore, list):
					subject_ignore = [subject_ignore]
				self.trial_names = [name for name in self.trial_names if not any(name_ignore in name for name_ignore in subject_ignore)]
			if trials_ignore:
				self.trial_names = [name for name in self.trial_names if not any(name_ignore in name for name_ignore in trials_ignore)]

			self.subject_use = np.unique([name.split('_')[0] for name in self.trial_names])
		
	def __len__(self):
		return len(self.trial_names)

	def __getitem__(self, idx):
		if isinstance(idx, list):
			trial_names = [self.trial_names[i] for i in idx]
		else:
			trial_names = self.trial_names[idx]
			if not isinstance(trial_names, list):
				trial_names = [trial_names]

		input_data = [self.read_tensor(self.input_dir + '/' + name) for name in trial_names]
		label_data = tuple([data[1] for data in input_data])
		label_data = torch.cat(label_data, dim=0)
		feature_data = tuple([data[0] for data in input_data])
		feature_data = torch.cat(feature_data, dim=0)

		return {'features': feature_data, 'labels': label_data}

	def delete(self, idx):
		if not isinstance(idx, list):
			# idx = list(idx)
			idx = [idx]
		del_trials = [self.trial_names[i] for i in idx]
		x = [self.trial_names.remove(trial) for trial in del_trials]

	def read_tensor(self, file_path):
		df = pd.read_csv(file_path) # Read in trial data

		# Select specified features
		features_use = list(df.columns)
		if 'all' not in self.ft_use:
			features_use = [feature for feature in features_use if any([f for f in self.ft_use if f in feature])]
		if self.ft_ignore:
			features_use = [feature for feature in features_use if not any([f for f in self.ft_ignore if f in feature])]
		self.features_use = features_use

		missing_ft = [f for f in self.ft_use if not any([f for feature in features_use if f in feature])]
		if any(missing_ft):
			print(f"Missing {(', ').join(missing_ft)} in input data!")

		# Convert data to pytorch tensor
		input_data = torch.tensor(df[features_use].values)
		label_data = torch.tensor([df[self.label].values,]).T

		if any([any(torch.isnan(d)) for d in input_data]):
			print('Warning - NaN in input ' + file_path)

		return input_data, label_data

def get_model_dict(model_type, h_dict):
	if model_type=='TCN':
		return get_tcn_dict(h_dict)

def get_tcn_dict(h_dict):
	m_dict = {k: h_dict[k] for k in ('ksize', 'dropout')}
	m_dict['eff_hist'] = 2*sum([(h_dict['ksize']-1)*(2**level) for level in range(h_dict['levels'])])
	m_dict['num_channels'] = [h_dict['hsize']]*h_dict['levels']
	return m_dict

def stride_features(data, ws, device_name='cpu', mean_feat=True, std_feat=True,  min_feat=True, max_feat=True, last_feat=True, pad_nans=True):
	data = data.to(torch.device('cpu')).numpy()
	feat_data_all = []
	for i in range(data.shape[1]):
		col_data = data[:, i]
		shape_des = (col_data.shape[0] - ws + 1, ws)
		strides_des = (col_data.strides[0], col_data.strides[0])
		strided_data = np.lib.stride_tricks.as_strided(col_data, shape=shape_des, strides=strides_des, writeable=False)

		feat_data = []
		if mean_feat:
			feat_data.append(np.mean(strided_data, axis=1).reshape(-1, 1))
		if std_feat:
			feat_data.append(np.std(strided_data, axis=1).reshape(-1, 1))
		if min_feat:
			feat_data.append(np.min(strided_data, axis=1).reshape(-1, 1))
		if max_feat:
			feat_data.append(np.max(strided_data, axis=1).reshape(-1, 1))
		if last_feat:
			feat_data.append(strided_data[:, -1].reshape(-1, 1))
		feat_data = np.concatenate(feat_data, axis=1)
		feat_data_all.append(feat_data)

		# mean_data_check = []
		# std_data_check = []
		# min_data_check = []
		# max_data_check = []
		# last_data_check = col_data[ws-1:]
		# for j in range(ws, data.shape[0]+1):
		# 	mean_data_check.append(np.mean(col_data[j-ws:j]))
		# 	std_data_check.append(np.std(col_data[j-ws:j]))
		# 	min_data_check.append(np.min(col_data[j-ws:j]))
		# 	max_data_check.append(np.max(col_data[j-ws:j]))
		# mean_data_check = np.array(mean_data_check)
		# std_data_check = np.array(std_data_check)
		# min_data_check = np.array(min_data_check)
		# max_data_check = np.array(max_data_check)

		# print()
		# print(feat_data.shape)
		# print(mean_data_check.shape)
		# print(std_data_check.shape)
		# print(min_data_check.shape)
		# print(max_data_check.shape)
		# print(last_data_check.shape)

		# plt.plot(feat_data[:, 0])
		# plt.plot(mean_data_check)
		# plt.title('Mean')
		# plt.show()

		# plt.plot(feat_data[:, 1])
		# plt.plot(std_data_check)
		# plt.title('Std')
		# plt.show()

		# plt.plot(feat_data[:, 2])
		# plt.plot(min_data_check)
		# plt.title('Min')
		# plt.show()

		# plt.plot(feat_data[:, 3])
		# plt.plot(max_data_check)
		# plt.title('Max')
		# plt.show()

		# plt.plot(feat_data[:, 4])
		# plt.plot(last_data_check)
		# plt.title('Last')
		# plt.show()

	feat_data_all = np.concatenate(feat_data_all, axis=1)
	feat_data_all = torch.from_numpy(feat_data_all).to(torch.device(device_name))
	if pad_nans:
		feat_data_all = torch.cat((torch.ones((ws-1, feat_data_all.shape[1]), device=torch.device(device_name))*float('NaN'), feat_data_all), dim=0)
	return feat_data_all


class NetFF(nn.Module):
	def __init__(self, input_size, output_size, num_layers=3, num_nodes=30, dropout=0.0, use_bn=False, af='relu'):
		super(NetFF, self).__init__()
		self._input_size = input_size
		self._num_layers = num_layers
		self._num_nodes = num_nodes
		self._use_bn = use_bn
		self._output_size = output_size

		if num_layers==1:
			self.model = nn.ModuleList([nn.Linear(input_size, output_size)])
		else:
			self.model = nn.ModuleList([nn.Linear(input_size, num_nodes)])
			if num_layers>2:
				self.model.extend([nn.Linear(num_nodes, num_nodes) for i in range(num_layers-2)])
			self.model.append(nn.Linear(num_nodes, output_size))

		if self._use_bn:
			self.bn = nn.ModuleList([nn.BatchNorm1d(num_nodes) for i in range(num_layers-1)])

		self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for i in range(num_layers-1)])

		if af.lower() == 'elu':
			self.af = nn.ELU
			# self.af = nn.ModuleList([nn.ELU() for i in range(num_layers)])
		elif af.lower() == 'relu':
			self.af = F.relu
			# self.af = nn.ModuleList([F.relu() for i in range(num_layers)])
		else:
			self.af = nn.Tanh()
			# self.af = nn.ModuleList([nn.Tanh() for i in range(num_layers)])

	def forward(self, x):
		if self._use_bn:
			for i in range(self._num_layers-1):
				x = self.dropout[i](self.af(self.bn[i](self.model[i](x))))
			# x = self.af[i](self.dropout[i](self.bn[i](self.model[i](x))))
		else:
			for i in range(self._num_layers-1):
				x = self.dropout[i](self.af(self.model[i](x)))
		x = self.model[-1](x)
		return x


class NetLSTM(nn.Module):
	def __init__(self, input_size, num_layers=3, num_nodes=30, pad_value=-200):
		super(NetLSTM, self).__init__()
		self.input_size_ = input_size
		self.num_layers_ = num_layers
		self.num_nodes_ = num_nodes
		self.pad_value_ = pad_value
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_nodes, num_layers=num_layers)
		self.fc_out = nn.Linear(num_nodes, 1)
		self.hidden = self.init_hidden(4)

	def forward(self, x, x_lengths, burn_in=0, reset_hidden=True):
		if reset_hidden:
			self.hidden = self.init_hidden(len(x_lengths))

		if burn_in > 0:
			x_burn_in = [x_val[:burn_in, :, :] for x_val in x]
			x_lengths_burn_in = [burn_in for length in x_lengths]
			x_burn_in_padded = self.pad_data(x_burn_in, x_lengths_burn_in)
			_, self.hidden = self.lstm(x_burn_in_padded, self.hidden)
			x_lengths = [length-burn_in for length in x_lengths]
			x = [x_val[burn_in:, :, :] for x_val in x]

		x_padded = self.pad_data(x, x_lengths)
		x_padded = nn.utils.rnn.pack_padded_sequence(x_padded, x_lengths, enforce_sorted=False)
		x_out, self.hidden = self.lstm(x_padded, self.hidden)
		x_out, _ = nn.utils.rnn.pad_packed_sequence(x_out, padding_value=self.pad_value_)
		x_out = x_out.contiguous()
		# x_out[:-1, :, :] = x_out[:-1, :, :].detach()
		x_out = torch.cat([x_out[:, i, :] for i in range(x_out.shape[1])], dim=0) # This should fix the alignment issue
		pad_comp_idx = x_out[:,0]>self.pad_value_
		x_out = self.fc_out(x_out)
		return x_out, pad_comp_idx

	def init_hidden(self, batch_size=4):
		hidden_state = torch.randn(self.num_layers_, batch_size, self.num_nodes_)
		cell_state = torch.randn(self.num_layers_, batch_size, self.num_nodes_)
		return(hidden_state, cell_state)

	def pad_data(self, x, x_lengths):
		minibatch_num = len(x)
		max_length = max(x_lengths)
		x_padded = torch.ones((max_length, minibatch_num, self.input_size_))*self.pad_value_
		for i, length in enumerate(x_lengths):
			x_padded[0:length, i, :] = x[i].view(-1, self.input_size_)
		return x_padded

class NetLSTM2(nn.Module):
	def __init__(self, input_size, num_layers=3, num_nodes=30, pad_value=-200):
		super(NetLSTM2, self).__init__()
		self.input_size_ = input_size
		self.num_layers_ = num_layers
		self.num_nodes_ = num_nodes
		self.pad_value_ = pad_value
		self.lstm = nn.LSTM(input_size=input_size, hidden_size=num_nodes, num_layers=num_layers)
		self.fc_out = nn.Linear(num_nodes, 1)
		self.hidden = self.init_hidden(4)

	def forward(self, x, x_lengths, reset_hidden=True):
		if reset_hidden:
			self.hidden = self.init_hidden(len(x))

		x = self.pad_data(x, x_lengths)

		# for i in range(x.shape[0]-1):
		# 	_, self.hidden = self.lstm(x[i, :, :], self.hidden)

		_, self.hidden = self.lstm(x[:-1, :, :], self.hidden)
		x_out, self.hidden = self.lstm(x[-1, :, :].view(1, -1, self.input_size_), self.hidden)
		x_out = x_out.contiguous()
		x_out = torch.cat([x_out[0, i, :].view(1, -1) for i in range(x_out.shape[1])], dim=0)
		x_out = self.fc_out(x_out)
		return x_out

	def forward_test(self, x, x_lengths, reset_hidden=True):
		if reset_hidden:
			self.hidden = self.init_hidden(len(x))
		x_padded = self.pad_data(x, x_lengths)
		x_padded = nn.utils.rnn.pack_padded_sequence(x_padded, x_lengths, enforce_sorted=False)
		x_out, self.hidden = self.lstm(x_padded, self.hidden)
		x_out, _ = nn.utils.rnn.pad_packed_sequence(x_out, padding_value=self.pad_value_)
		x_out = x_out.contiguous()
		x_out = torch.cat([x_out[:, i, :] for i in range(x_out.shape[1])], dim=0) # This should fix the alignment issue
		pad_comp_idx = x_out[:,0]>self.pad_value_
		x_out = self.fc_out(x_out)
		return x_out, pad_comp_idx

	def init_hidden(self, batch_size=4):
		hidden_state = torch.randn(self.num_layers_, batch_size, self.num_nodes_)
		cell_state = torch.randn(self.num_layers_, batch_size, self.num_nodes_)
		return(hidden_state, cell_state)

	def pad_data(self, x, x_lengths):
		minibatch_num = len(x)
		max_length = max(x_lengths)
		x_padded = torch.ones((max_length, minibatch_num, self.input_size_))*self.pad_value_
		for i, length in enumerate(x_lengths):
			x_padded[0:length, i, :] = x[i].view(-1, self.input_size_)
		return x_padded


class NetLSTM3(nn.Module):
	def __init__(self, input_size, output_size, lstm_layers, lstm_nodes, fc_layers, fc_nodes, lstm_dropout=0.0, fc_dropout=0.0, pad_value=-200, device=torch.device('cpu')):
		super(NetLSTM3, self).__init__()
		self.input_size = input_size
		self.lstm_layers = lstm_layers
		self.lstm_nodes = lstm_nodes
		self.fc_layers = fc_layers
		self.fc_nodes = fc_nodes
		self.lstm_dropout = lstm_dropout
		self.fc_dropout = fc_dropout
		self.pad_value = pad_value
		self.device = device

		self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_nodes, num_layers=lstm_layers, dropout=lstm_dropout)

		if fc_layers==1:
			self.fc = nn.ModuleList([nn.Linear(lstm_nodes, 1)])
		else:
			self.fc = nn.ModuleList([nn.Linear(lstm_nodes, fc_nodes)])
			self.fc.extend([nn.Linear(fc_nodes, fc_nodes) for i in range(fc_layers-2)])
			self.fc.append(nn.Linear(fc_nodes, output_size))
			self.relu = nn.ModuleList([nn.ReLU() for i in range(fc_layers-1)])
			self.dropout = nn.ModuleList([nn.Dropout(fc_dropout) for i in range(fc_layers-1)])

		self.hidden = self.init_hidden(batch_size=1, device=device)

		# Add in relu activation (Done)
		# Also make model from sequential and then compare forward pass
		# Also need to init model weights (Done)
		# Also need to init lstm cell and hidden states (Done)
		# Need to try out burn in w/ detach(). I also need to think about if we want to include this maybe include this question for Matthew. I think I should just schedule a meeting with Matthew.
			# Bascially how should I choose window size for training and how should I handle burn in
		# Need to implement dropout for lstm and fc layers
			# Check that dropout=0 for fc works
		# Pad data
		# Default input is seq_len x batch x input size

	def forward(self, x, x_lengths, eff_hist=0, burn_in=0, reset_hidden=True, debug=False):
		time1 = time.time()
		if reset_hidden:
			self.hidden = self.init_hidden(x.shape[1], device=self.device)

		if burn_in > 0:
			x_burn_in = x[:burn_in, :, :]
			_, self.hidden = self.lstm(x_burn_in, self.hidden)
			# x_lengths = [length-burn_in for length in x_lengths]
			x = x[burn_in:, :, :]
			eff_hist -= burn_in

			# x_burn_in = [x_val[:burn_in, :, :] for x_val in x]
			# x_lengths_burn_in = [burn_in for length in x_lengths]
			# x_burn_in_padded = self.pad_data(x_burn_in, x_lengths_burn_in)
			# _, self.hidden = self.lstm(x_burn_in_padded, self.hidden)
			# x_lengths = [length-burn_in for length in x_lengths]
			# x = [x_val[burn_in:, :, :] for x_val in x]

		x = x.detach() # This uses the original memory of x but removes the gradient of previous operations
		x_lengths_pad = [length+eff_hist for length in x_lengths]

		time1 = time.time() - time1
		time2 = time.time()

		# Need to detach here for truncated BPTT I think
		# Stoped here: need to shift pad_adta to ourside then pack the sequence in here
		# x_padded = self.pad_data(x, x_lengths) # Move this outside of forward
		time4 = time.time()
		x_padded = nn.utils.rnn.pack_padded_sequence(x, x_lengths_pad, enforce_sorted=False)
		time4 = time.time() - time4
		time5 = time.time()	
		x_out, self.hidden = self.lstm(x_padded, self.hidden)
		# x_out, self.hidden = self.lstm(x, self.hidden)
		time5 = time.time() - time5
		time6 = time.time()
		x_out, _ = nn.utils.rnn.pad_packed_sequence(x_out, padding_value=self.pad_value)
		time6 = time.time() - time6

		time2 = time.time() - time2
		time3 = time.time()

		x_out = torch.cat([x_out[eff_hist:eff_hist+x_lengths[i], i, :].contiguous() for i in range(x_out.shape[1])], dim=0).contiguous()
		# if any(x_lengths):
		# 	x_out = torch.cat([x_out[eff_hist:eff_hist+x_lengths[i], i, :].contiguous() for i in range(x_out.shape[1])], dim=0).contiguous()
		# else:
		# 	x_out = torch.cat([x_out[eff_hist:, i, :].contiguous() for i in range(x_out.shape[1])], dim=0).contiguous()

		for i in range(len(self.fc)-1):
			x_out = self.dropout[i](self.relu[i](self.fc[i](x_out)))
		x_out = self.fc[-1](x_out)

		time3 = time.time() - time3
		
		if debug:
			return x_out, (time1, time2, time3, time4, time5, time6)
		else:
			return x_out


	# def pad_data(self, x, x_lengths):
	# 	minibatch_num = len(x)
	# 	max_length = max(x_lengths)
	# 	x_padded = torch.ones((max_length, minibatch_num, self.input_size_))*self.pad_value_
	# 	for i, length in enumerate(x_lengths):
	# 		x_padded[0:length, i, :] = x[i].view(-1, self.input_size_)
	# 	return x_padded

	def init_hidden(self, batch_size=4, device=torch.device('cpu')):
		hidden_state = torch.randn(self.lstm_layers, batch_size, self.lstm_nodes).to(device)
		cell_state = torch.randn(self.lstm_layers, batch_size, self.lstm_nodes).to(device)
		return(hidden_state, cell_state)

	# def forward(self, x, x_lengths, reset_hidden=True):
	# 	if reset_hidden:
	# 		self.hidden = self.init_hidden(len(x))

	# 	x = self.pad_data(x, x_lengths)

	# 	# for i in range(x.shape[0]-1):
	# 	# 	_, self.hidden = self.lstm(x[i, :, :], self.hidden)

	# 	_, self.hidden = self.lstm(x[:-1, :, :], self.hidden)
	# 	x_out, self.hidden = self.lstm(x[-1, :, :].view(1, -1, self.input_size_), self.hidden)
	# 	x_out = x_out.contiguous()
	# 	x_out = torch.cat([x_out[0, i, :].view(1, -1) for i in range(x_out.shape[1])], dim=0)
	# 	x_out = self.fc_out(x_out)
	# 	return x_out

	# def forward(self, x, x_lengths, burn_in=0, reset_hidden=True):
	# 	if reset_hidden:
	# 		self.hidden = self.init_hidden(len(x_lengths))

	# 	if burn_in > 0:
	# 		x_burn_in = [x_val[:burn_in, :, :] for x_val in x]
	# 		x_lengths_burn_in = [burn_in for length in x_lengths]
	# 		x_burn_in_padded = self.pad_data(x_burn_in, x_lengths_burn_in)
	# 		_, self.hidden = self.lstm(x_burn_in_padded, self.hidden)
	# 		x_lengths = [length-burn_in for length in x_lengths]
	# 		x = [x_val[burn_in:, :, :] for x_val in x]

	# 	x_padded = self.pad_data(x, x_lengths)
	# 	x_padded = nn.utils.rnn.pack_padded_sequence(x_padded, x_lengths, enforce_sorted=False)
	# 	x_out, self.hidden = self.lstm(x_padded, self.hidden)
	# 	x_out, _ = nn.utils.rnn.pad_packed_sequence(x_out, padding_value=self.pad_value_)
	# 	x_out = x_out.contiguous()
	# 	# x_out[:-1, :, :] = x_out[:-1, :, :].detach()
	# 	x_out = torch.cat([x_out[:, i, :] for i in range(x_out.shape[1])], dim=0) # This should fix the alignment issue
	# 	pad_comp_idx = x_out[:,0]>self.pad_value_
	# 	x_out = self.fc_out(x_out)
	# 	return x_out, pad_comp_idx


class Chomp1d(nn.Module):
	def __init__(self, chomp_size):
		super(Chomp1d, self).__init__()
		self.chomp_size = chomp_size

	def forward(self, x):
		return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
	def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
		super(TemporalBlock, self).__init__()

		self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
			stride=stride, padding=padding, dilation=dilation))
		self.chomp1 = Chomp1d(padding)
		self.relu1 = nn.ReLU()
		self.dropout1 = nn.Dropout(dropout)

		self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
			stride=stride, padding=padding, dilation=dilation))
		self.chomp2 = Chomp1d(padding)
		self.relu2 = nn.ReLU()
		self.dropout2 = nn.Dropout(dropout)

		self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
			self.conv2, self.chomp2, self.relu2, self.dropout2)

		self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None 
		# for param in self.downsample.parameters():
		# 	print(param)
		# return
		self.relu = nn.ReLU()
		self.init_weights()

	def init_weights(self):
		self.conv1.weight.data.normal_(0, 0.01)
		self.conv2.weight.data.normal_(0, 0.01)
		if self.downsample is not None:
			self.downsample.weight.data.normal_(0, 0.01)

	def forward(self, x):
		# out = self.conv1(x)
		# print(out.shape)
		
		out = self.net(x)
		res = x if self.downsample is None else self.downsample(x)		
		return self.relu(out + res)


class TemporalConvNet(nn.Module):
	def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
		super(TemporalConvNet, self).__init__()
		layers = []
		num_levels = len(num_channels)
		for i in range(num_levels):
			dilation_size = 2 ** i 
			in_channels = num_inputs if i == 0 else num_channels[i-1]
			out_channels = num_channels[i]
			layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, 
				padding=(kernel_size-1) * dilation_size, dropout=dropout)]

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		return self.network(x)


class TCN(nn.Module):
	def __init__(self, input_size, output_size, num_channels, ksize, dropout, eff_hist):
		super(TCN, self).__init__()
		self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=ksize, dropout=dropout)
		self.linear = nn.Linear(num_channels[-1], output_size)
		self.init_weights()
		self.eff_hist = eff_hist

	def init_weights(self):
		self.linear.weight.data.normal_(0, 0.01)

	# def forward(self, x, sequence_lens=[], t=-1):
	def forward(self, x, sequence_lens=[]):
		# print(x.shape)
		y1 = self.tcn(x)
		# start_time = time.time()

		y1 = y1[:, :, -1].contiguous()

		# y1 = torch.cat([y1[i, :, self.eff_hist:].contiguous() for i in range(y1.shape[0])], dim=1)
		# y1 = y1.transpose(0, 1).contiguous()

		# if any(sequence_lens):
		# 	y1 = torch.cat([y1[i, :, self.eff_hist:self.eff_hist+sequence_lens[i]].contiguous() for i in range(y1.shape[0])], dim=1).transpose(0, 1).contiguous()	
		# else:
		# 	y1 = torch.cat([y1[i, :, self.eff_hist:].contiguous() for i in range(y1.shape[0])], dim=1).transpose(0, 1).contiguous()
		
		# t += time.time()-start_time
		# if t>=0:
		# 	return self.linear(y1), t
		return self.linear(y1)


class CNN(nn.Module):
	def __init__(self, input_size, output_size, window_size, kernel_size=[5], channel_size=[], hidden_size=[10], dropout=0.0, bkf=False, fast=True):
		super(CNN, self).__init__()

		# If channel size of conv is not specified but kernel size is, assuming channel size = input size for each conv layer
		if not any(channel_size) and any(kernel_size):
			channel_size = [input_size for i in range(len(kernel_size))]

		# Compute kernel size of final conv layer to condense timestep data into size 1 before passing to linear layers
		if any(kernel_size):
			ws_out = window_size
			for k in kernel_size:
				ws_out -= (k - 1)
			kernel_size.append(ws_out)
		else:
			kernel_size.append(window_size)

		# Set last channel size of conv layers equal to first input size of linear layers
			# Reminder: First element in hidden_size is actually the input layer of the linear layers that comes out of the conv layers. So technically the first element is not a 'hidden' layer
			# Further: The number of weight columns (layers) is equal to the length of the hidden_size list including the output layer
		if any(hidden_size):
			channel_size.append(hidden_size[0])
		else:
			channel_size.append(output_size)

		# Insert input size as first channel size for building the conv layers
		channel_size.insert(0, input_size)

		# Create conv layers and corresponding activation functions
		# self.conv = nn.ModuleList([weight_norm(nn.Conv1d(channel_size[i], channel_size[i+1], ksize)) for i, ksize in enumerate(kernel_size)]) # Consider batch norm instead of weight norm here!! Check batchnorm1d and batchnorm2d.
		self.conv = nn.ModuleList([nn.Conv1d(channel_size[i], channel_size[i+1], ksize) for i, ksize in enumerate(kernel_size)]) # Consider batch norm instead of weight norm here!! Check batchnorm1d and batchnorm2d.
		self.conv_af = nn.ModuleList(nn.ReLU() for i in range(len(kernel_size)))
		self.num_conv_layers = len(self.conv)

		# Create linear layers and corresponding activation functions
		self.linear = nn.ModuleList([nn.Linear(hidden_size[i], hidden_size[i+1]) for i in range(len(hidden_size)-1)])
		# self.linear_bn = nn.ModuleList([nn.BatchNorm1d(hidden_size[i]) for i in range(len(self.linear))]) # Currently not adding batch norm on output layer (don't think it matters since it's size 1 anyways)
		self.linear_af = nn.ModuleList([nn.ReLU() for i in range(len(self.linear))]) # Activation function for each linear layer including output
		self.num_linear_layers = len(self.linear)

		# Create output layer
		# Add Kalman Filter parameters to module if bkf is true
		self.bkf = bkf
		self.fast = fast
		if self.bkf and not self.fast:
			self.A_mat = torch.tensor([[1.0, 0.005], [0.0, 1.0]]) # Dynamics model is an integrator
			self.C_mat = torch.tensor([[1.0, 0.0], [0.0, 1.0]]) # Identify matrix
			self.Q_mat = torch.tensor([[0.1, 0.0], [0.0, 0.1]]) # Process noise matrix (might need to change to be learnable using nn.parameter.Parameter). Haarnoja et al. 2016 said they only added noise on velocity term
			self.output_layer = nn.Linear(hidden_size[-1], 5) # Override output size to output estimated tau (size 1), tau_dot (size 2), and covariance matrix (size 3)

		elif self.bkf and self.fast:
			self.output_layer = nn.Linear(hidden_size[-1], 5) # Override output size to output estimated tau (size 1), tau_dot (size 2), and covariance matrix (size 3)

		else:
			self.output_layer = nn.Linear(hidden_size[-1], output_size)

		self.init_weights()
		self.window_size = window_size

	def init_weights(self):
		for i in range(self.num_conv_layers):
			self.conv[i].weight.data.normal_(0, 0.01)

		for i in range(self.num_linear_layers):
			self.linear[i].weight.data.normal_(0, 0.01)

		self.output_layer.weight.data.normal_(0, 0.01)

	def forward(self, x, debug=False):
		# plt.plot(x[0, 0, :])
		# plt.show()

		# print(x.shape)

		# Get sequence length of input data (only used is bkf is true)
		num_sequence = x.shape[0]
		len_sequence = x.shape[-1] - self.window_size + 1

		# Pass data through conv layers
		for i in range(self.num_conv_layers):
			x = self.conv_af[i](self.conv[i](x))

		# print(x.shape)
		# plt.plot(x[0, 0, :].detach().numpy())

		# Reshape mini-batch data to align w/ expected input shape for linear layers (n x input_size)
		x = torch.cat([x[i, :, :].contiguous() for i in range(x.shape[0])], dim=1).transpose(0, 1).contiguous()

		# print(x.shape)
		# plt.plot(x[:, 0].detach().numpy())

		# Pass data through linear layers
		for i in range(self.num_linear_layers):
			# x = self.linear_af[i](self.linear_bn[i](self.linear[i](x)))

			x = self.linear_af[i](self.linear[i](x))
			# plt.plot(x[:, 0].detach().numpy())
			# if plot:
			# 	plt.show()
			# 	return

		# print(x.shape)
		

		# Pass data through output layer
		x = self.output_layer(x)

		if debug:
			x_unfilt = x
		
		# A_mat: Dynamics model is an integrator
		# C_mat: Output matrix equal to identity matrix
		# Q_mat: Process noise matrix (might need to change to be learnable using nn.parameter.Parameter). Haarnoja et al. 2016 said they only added noise on velocity term
			
		A_mat = torch.eye(2*num_sequence)
		A_mat += torch.diagflat(torch.cat([torch.ones((num_sequence, 1))*0.005, torch.zeros((num_sequence, 1))], dim=1), offset=1)[:-1, :-1]
		C_mat = torch.eye(2*num_sequence)
		Q_mat = torch.eye(2*num_sequence)*0.1
		
		x_prev = x[::len_sequence, :2].contiguous().view(-1, 1)
		sigma_prev = torch.eye(2*num_sequence)
		bkf_out = [x_prev[::2]]

		if self.bkf and self.fast:
			for i in range(1, len_sequence):
				# Compute measurement uncertainty from NN (R_mat) using Cholesky Decomposition to guarantee a positive-definite matrix
					# Square the diagonal terms to guarantee lower triangular matrix w/ positive real values in the diagonal
				L_mat = torch.diagflat(x[i::len_sequence, 2::2])**2
				L_mat += torch.diagflat(torch.cat([x[i::len_sequence, 3].view(-1, 1), torch.zeros((num_sequence, 1))], dim=1), offset=-1)[:-1, :-1]
				R_mat = torch.matmul(L_mat, L_mat.transpose(0, 1))

				# Compute dynamic model and measured (NN estimate) state
				x_model = torch.matmul(A_mat, x_prev) # Dynamic model estimate of current state. Joint torque and joint d_torque.
				x_nn = x[i::len_sequence, :2].contiguous().view(-1, 1) # Measurement (NN estimate) of current state. Joint torque and joint d_torque.

				# Compute model uncertainty w/ process noise
				sigma_model = torch.matmul(torch.matmul(A_mat, sigma_prev), A_mat.transpose(0, 1)) + Q_mat # Currently process noise is hyperparameter, but could be changed to a learnable parameter.

				# Compute Kalman Gain using dynamice model uncertainty (sigma_model) and measurement uncertainty (R_mat)
				K_mat_num = torch.matmul(sigma_model, C_mat.transpose(0, 1))
				K_mat_den = torch.matmul(torch.matmul(C_mat, sigma_model), C_mat.transpose(0, 1)) + R_mat
				K_mat = torch.matmul(K_mat_num, torch.inverse(K_mat_den))

				# Compute filtered output and model uncertainty using Kalman Gain
				x_out = x_model + torch.matmul(K_mat, (x_nn - torch.matmul(C_mat, x_model))) # Filtered output
				sigma_out = torch.matmul((torch.eye(2*num_sequence) - torch.matmul(K_mat, C_mat)), sigma_model) # Updated model uncertainty

				# Save filtered output as output of foward pass and update state and model uncertainty for next pass
				x_prev = x_out
				sigma_prev = sigma_out
				bkf_out.append(x_out[::2])

			x = torch.cat(bkf_out, dim=1).view(-1, 1).contiguous()

		elif self.bkf and not self.fast:
			z = x[:, 0:2]
			L_mat_hat = x[:, 2:]
			
			bkf_out = []
			for i in range(x.shape[0]):
				if i % len_sequence == 0: # For mini-batch training, make sure to reset Kalman Filter at the start of each sequence
					bkf_out.append(z[i, :].view(1, -1)) # Save initial KF output as the NN estimate
					x_prev = z[i, :].view(-1, 1) # Save first NN estimate as previous state for next iteration of Kalman Filter
					sigma_prev = torch.tensor([[1.0, 0.0], [0.0, 1.0]]) # Make sure autograd is true after recomputing simga_out
				else:
					# Old 1D Kalman Filter
					# model_est = bkf_out[i-1] + (bkf_out[i-1]-bkf_out[i-2]) # Constant velocity model
					# kalman_gain = model_err / (model_err + x[i, 1])
					# bkf_out.append(model_est + kalman_gain * (x[i, 0] - model_est))
					# model_err = (1 - kalman_gain) * (model_err + self.process_noise)

					# Updated Kalman Filter using integrator model as done in Haarnoja et al. 2016
					# Compute measurement uncertainty from NN (R_mat) using Cholesky Decomposition to guarantee a positive-definite matrix
					L_mat = torch.zeros((2, 2)) # Lower triangular matrix from NN output used to compute model uncertainty
					# L_mat[0, 0] = torch.exp(L_mat_hat[i, 0]) # Exponentiated diagonal guarantees L*L_T to be positive-definite and symmetric
					# L_mat[1, 0] = L_mat_hat[i, 1]
					# L_mat[1, 1] = torch.exp(L_mat_hat[i, 2]) # Exponentiated diagonal guarantees L*L_T to be positive-definite and symmetric
					L_mat[0, 0] = L_mat_hat[i, 0]**2 # Exponentiated diagonal guarantees L*L_T to be positive-definite and symmetric
					L_mat[1, 0] = L_mat_hat[i, 1]
					L_mat[1, 1] = L_mat_hat[i, 2]**2 # Exponentiated diagonal guarantees L*L_T to be positive-definite and symmetric
					R_mat = torch.matmul(L_mat, L_mat.transpose(0, 1)) # Measurement uncertainty output from NN (also called Observational Covariance Matrix)
					
					# Compute dynamic model and measured (NN estimate) state
					x_model = torch.matmul(self.A_mat, x_prev) # Dynamic model estimate of current state. Joint torque and joint d_torque.
					x_nn = z[i, :].view(-1, 1) # Measurement (NN estimate) of current state. Joint torque and joint d_torque.
					
					# Compute model uncertainty w/ process noise
					sigma_model = torch.matmul(torch.matmul(self.A_mat, sigma_prev), self.A_mat.transpose(0, 1)) + self.Q_mat # Currently process noise is hyperparameter, but could be changed to a learnable parameter.
					
					# Compute Kalman Gain using dynamice model uncertainty (sigma_model) and measurement uncertainty (R_mat)
					K_mat_num = torch.matmul(sigma_model, self.C_mat.transpose(0, 1))
					K_mat_den = torch.matmul(torch.matmul(self.C_mat, sigma_model), self.C_mat.transpose(0, 1)) + R_mat
					K_mat = torch.matmul(K_mat_num, torch.inverse(K_mat_den))

					# Compute filtered output and model uncertainty using Kalman Gain
					x_out = x_model + torch.matmul(K_mat, (x_nn - torch.matmul(self.C_mat, x_model))) # Filtered output
					sigma_out = torch.matmul((torch.eye(2) - torch.matmul(K_mat, self.C_mat)), sigma_model) # Updated model uncertainty
					
					# Save filtered output as output of foward pass and update state and model uncertainty for next pass
					x_prev = x_out
					sigma_prev = sigma_out
					bkf_out.append(x_out.transpose(0, 1))

			x = torch.cat(bkf_out, dim=0)[:, 0].view(-1, 1).contiguous() # Only output the torque estimate (removing the d_torque estimate)

		if debug:
			return x, x_unfilt
		else:
			return x


class AntTCN(nn.Module):
	def __init__(self, input_size, input_size_ant, output_size, num_channels, kernel_size, dropout, eff_hist):
		super(AntTCN, self).__init__()
		self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

		self.ant_linear1 = nn.Linear(input_size_ant, 10)
		self.ant_relu1 = nn.ReLU()
		self.ant_linear2 = nn.Linear(10, 10)
		self.ant_relu2 = nn.ReLU()
		# self.linear1 = nn.Linear(num_channels[-1]+input_size_ant, num_channels[-1])
		self.linear1 = nn.Linear(num_channels[-1]+10, num_channels[-1])
		self.relu1 = nn.ReLU()
		self.linear2 = nn.Linear(num_channels[-1], num_channels[-1])
		self.relu2 = nn.ReLU()
		self.linear3 = nn.Linear(num_channels[-1], output_size)
		self.init_weights()
		self.eff_hist = eff_hist

	def init_weights(self):
		self.linear1.weight.data.normal_(0, 0.01)
		self.linear2.weight.data.normal_(0, 0.01)
		self.linear3.weight.data.normal_(0, 0.01)

		self.ant_linear1.weight.data.normal_(0, 0.01)
		self.ant_linear2.weight.data.normal_(0, 0.01)

	def forward_pass(self, x, x_ant):
		x_ant = torch.cat([x_ant[i, :, self.eff_hist:].contiguous() for i in range(x_ant.shape[0])], dim=1).transpose(0, 1).contiguous()
		y1 = self.forward(x)

		y1_ant = self.ant_relu1(self.ant_linear1(x_ant))
		y1_ant = self.ant_relu2(self.ant_linear2(y1_ant))
		# y1 = torch.cat([y1, x_ant], dim=1).contiguous()
		y1 = torch.cat([y1, y1_ant], dim=1).contiguous()
		y1 = self.relu1(self.linear1(y1))
		y1 = self.relu2(self.linear2(y1))
		return self.linear3(y1)

	def forward(self, x):
		y1 = self.tcn(x)
		y1 = torch.cat([y1[i, :, self.eff_hist:].contiguous() for i in range(y1.shape[0])], dim=1).transpose(0, 1).contiguous()
		return y1


class LossFunctions():

	class MSELoss():
		def __init__(self, weight_importance=False):
			self.weight_importance = weight_importance

		def forward(self, y1, y2, y_var=[]):
			# if y_var.shape == y1.shape:
			# 	mse = torch.mean(((y1-y2)**2)/y_var)
			# else:
			# 	mse = torch.mean((y1-y2)**2)
			err = torch.abs(y1-y2)
			# if y_var.shape == y1.shape:
			if self.weight_importance:
				err /= y_var
			mse = torch.mean(err**2)
			return mse

	class SmoothL1Loss():
		def __init__(self, weight_importance=False):
			self.weight_importance = weight_importance

		def forward(self, y1, y2, y_var):
			err = torch.abs(y1-y2)

			# if y_var.shape == y1.shape:
			if self.weight_importance:
				err /= y_var

			l1 = err-0.5
			l2 = 0.5*(err**2)
			mask = err < 1

			huber = l1 
			huber[mask] = l2[mask]
			huber = torch.mean(huber)
			return huber
