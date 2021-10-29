import numpy as np
from scipy import signal


class Filter(object):
    '''Parent class for filters, to help with type hinting. 
    Note: for filter modularity, all child classes should have a filter() 
    function that takes only the most recent value. This way, different filters 
    can be passed to objects constructors without replacing that class's code'''

    def filter(self, new_val):
        raise ValueError('filter() not implemented for child class of Filter')


class PassThroughFilter(Filter):
    def filter(self, new_val):
        return new_val


class Butterworth(Filter):
    '''Implements a real-time Butterworth filter using second orded cascaded filters.'''

    def __init__(self, N: int, Wn: float, btype='low', fs=None, n_cols=0):
        ''' 
        N: order
        Wn: (default) normalized cutoff freq (cutoff freq / Nyquist freq). If fs is passed, cutoff is in freq.
        btype: 'low', 'high', or 'bandpass'
        fs: Optional: sample freq, Hz. If not None, Wn describes the cutoff freq in Hz
        '''
        self.N = N
        if fs is not None:
            self.Wn = Wn/(fs/2)
        else:
            self.Wn = Wn
        self.btype = btype
        self.sos = signal.butter(N=self.N, Wn=self.Wn,
                                 btype=self.btype, output='sos')
        self.zi = signal.sosfilt_zi(self.sos)
        if n_cols > 0:
            self.zi = np.repeat(self.zi, n_cols, axis=0).transpose().reshape(1, 2, -1)

    def filter(self, new_val, axis=0):
        filtered_val, self.zi = signal.sosfilt(
            sos=self.sos, x=new_val, zi=self.zi, axis=axis)
        return filtered_val

    def filter_one(self, new_val: float) -> float:
        return self.filter([new_val])[0]


class MovingAverage(Filter):
    '''Implements a real-time moving average filter.'''

    def __init__(self, window_size):
        self.deque = collections.deque([], maxlen=window_size)

    def filter(self, new_val):
        # Optimize for efficiency is window size is large
        self.deque.append(new_val)
        return np.mean(self.deque)

def is_sequence(x):
	return isinstance(x, list) or isinstance(x, np.ndarray)


if __name__=="__main__":
    order = 2
    fcut = 6
    fs = 100
    x = np.sin(np.linspace(0, 2*np.pi, 100)).reshape(-1, 1) + np.random.uniform(-0.1, 0.1, (100, 1))

    print('Testing Butterworth filter methods.')

    # Test one-shot filterting
    filt = Butterworth(order, fcut, fs=fs, n_cols=1)
    x_filt = filt.filter(x, axis=0)

    # Test filtering one-at-a-time (2 versions)
    filt = Butterworth(order, fcut, fs=fs, n_cols=1)
    x_filt_2 = np.array([filt.filter_one(i) for i in x]).reshape(-1, 1)

    filt = Butterworth(order, fcut, fs=fs, n_cols=1)
    x_filt_3 = np.array([filt.filter([i])[0] for i in x]).reshape(-1, 1)

    np.testing.assert_allclose(x_filt, x_filt_2)
    np.testing.assert_allclose(x_filt, x_filt_3)
    print('Passed!')
    
    # from matplotlib import pyplot as plt
    # plt.plot(x)
    # plt.plot(x_filt)
    # plt.plot(x_filt_2)
    # plt.plot(x_filt_3)
    # plt.show()