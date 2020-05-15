# extact feature on test wav
import pyworld as pw
import os
import pdb
import librosa
# import scipy.io.wavfile as sp
import numpy as np
import scipy


def entropy(filename):
    y, sr = librosa.core.load(filename)
    y = y.astype(np.float64)
    f0, sp, ap = pw.wav2world(y, sr)
    # compute entropy of f0
    f0_entropy = -1
    # if sum(f0) 
    entropy = scipy.stats.entropy
    f0_entropy = entropy(np.trim_zeros(f0))
    return f0_entropy

def spectral_entropy(filename):
    y, sr = librosa.core.load(filename)
    y = y.astype(np.float64)
    f0, sp, ap = pw.wav2world(y, sr)
    # power spectral density 
    freq, psd = scipy.signal.periodogram(np.trim_zeros(f0))
    # pdb.set_trace()
    # normalized_v = v / np.sqrt(np.sum(v**2))
    # This routine will normalize pk and qk if they don’t sum to 1.
    f0_spectral_entropy = scipy.stats.entropy(psd)
    return f0_spectral_entropy

def sepctral_entropy_variance(filename):
    y, sr = librosa.core.load(filename)
    # STFT spectrogram
    D = np.abs(librosa.stft(y))
    entropy = []
    N = D.shape[1]
    for i in range(N):
        current_spec = D[:,i]
        entropy.append(scipy.stats.entropy(current_spec[:]))
    # pdb.set_trace()
    return np.var(entropy)

# def mean_spectral_variance(filename):
# a vector          


def main():
    # path = './justin_roiland/' 
    # path = './seth_macfarlane/'
    # path = '/home/yang/Python-Wrapper-for-World-Vocoder/DisguisedVoice/data/all_data_8k/'
    path = '/home/yang/ASV-attack/pipelines/simplefeatures/Entropy-git/'
    # data_A = open(path + 'data_A.txt', 'r')
    # data_A = open(path + 'dev_ASVbonafide.txt', 'r')
    # data_B = open(path + 'dev_ASVspoof.txt', 'r')
    # data_A = open(path + 'eval_ASVbonafide.txt', 'r')
    # data_B = open(path + 'eval_ASVspoof.txt', 'r')
    data_A = open(path + 'train_ASVbonafide.txt', 'r')
    data_B = open(path + 'train_ASVspoof.txt', 'r')


    Sentropy_dataA = []
    entropy_dataA = []
    for item in data_A:
        # pdb.set_trace()
        tmp0 = spectral_entropy(item[:-1])
        tmp1 = entropy(item[:-1])
        # tmp = sepctral_entropy_variance(item[:-1])
        # print(tmp)
        Sentropy_dataA.append(tmp0)
        entropy_dataA.append(tmp1)
    entropy_dataA = np.array(entropy_dataA)
    Sentropy_dataA = np.array(Sentropy_dataA)

    # data_B = open(path + 'data_B.txt', 'r')
    Sentropy_dataB = []
    entropy_dataB = []
    for item in data_B:
        # pdb.set_trace()
        tmp0 = spectral_entropy(item[:-1])
        tmp1 = entropy(item[:-1])
        # tmp = sepctral_entropy_variance(item[:-1])
        # print(tmp)
        Sentropy_dataB.append(tmp0)
        entropy_dataB.append(tmp1)

    Sentropy_dataB = np.array(entropy_dataB)
    entropy_dataB = np.array(entropy_dataB)

    # pdb.set_trace()
    # save array
    # np.save('ASV_dev_Bonafide_se', Sentropy_dataA)
    # np.save('ASV_dev_Spoof_se', Sentropy_dataB)
    # np.save('ASV_dev_Bonafide_e', entropy_dataA)
    # np.save('ASV_dev_Spoof_e', entropy_dataB) 

    # np.save('ASV_eval_Bonafide_se', Sentropy_dataA)
    # np.save('ASV_eval_Spoof_se', Sentropy_dataB)
    # np.save('ASV_eval_Bonafide_e', entropy_dataA)
    # np.save('ASV_eval_Spoof_e', entropy_dataB) 

    np.save('ASV_train_Bonafide_se', Sentropy_dataA)
    np.save('ASV_train_Spoof_se', Sentropy_dataB)
    np.save('ASV_train_Bonafide_e', entropy_dataA)
    np.save('ASV_train_Spoof_e', entropy_dataB) 

if __name__ == '__main__':
    main()

