import scipy
from scipy import signal,stats
import librosa
from librosa import display
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

path_h5 = '/media/quanc/E/data/nanopore_basecalling/res_pytorch/data_test_rrwick_read10000/test_read500/result-albacore/workspace/0/5210_N125509_20170424_FN2002039725_MN19691_sequencing_run_klebs_033_75349_ch41_read7391_strand.fast5'
if __name__ == '__main__':
    file_h5 = h5py.File(path_h5, 'r')

    raw_data = list(file_h5['/Raw/Reads/'].values())[0]
    raw_data = raw_data['Signal'].value

    event_data = file_h5['/Analyses/Basecall_1D_000/BaseCalled_template/Events']
    test_event = event_data[1:100]
    test_signal = stats.zscore(raw_data[test_event['start'][0]:test_event['start'][-1]])
    test_time = np.arange(len(test_signal))/4000.0
    f, t , Zxx = scipy.signal.stft(test_signal, fs=4000, nperseg=50, noverlap=False)
    plt.plot(stats.zscore(test_event['mean']))
    plt.show()
    plt.plot(test_time, test_signal)
    plt.show()
    plt.pcolormesh(t, f, np.abs(Zxx), vmin=0)
    plt.colorbar()
    plt.show()

    S = librosa.feature.melspectrogram(y=test_signal,sr=4000,n_fft=512)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                             x_axis='time',y_axis='mel')
    plt.colorbar()
    plt.show()
    # mfccs = librosa.feature.mfcc(test_signal, sr=4000, n_mfcc=20)
    # librosa.display.specshow(mfccs, x_axis='time')
    # plt.colorbar()
    # plt.title('MFCC')
    # plt.tight_layout()
    # plt.show()
