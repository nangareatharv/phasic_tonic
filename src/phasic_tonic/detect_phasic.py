from utils import get_sequences

import numpy as np
from neurodsp.filt import filter_signal
from scipy.signal import hilbert

def detect_phasic(rem, fs, nfilt=11, thr_dur=900):

    filt = np.ones(nfilt)/nfilt

    thr1, thr2, thr3, iti, amps = _compute_thresholds(rem, fs, smooth_filt=filt)

    phREM = {rem_idx:[] for rem_idx in rem.keys()}

    for rem_idx in iti:
        rem_start, rem_end = rem_idx
        offset = rem_start * fs

        # trough indices
        tridx = iti[rem_idx]

        # smoothed trough interval
        sdiff = np.convolve

        # amplitude of the REM epoch
        eegh = amps[rem_idx]

        # get the candidates for phREM
        cand_idx = np.where(sdiff <= thr1)[0]
        cand = get_sequences(cand_idx)

        for start, end in cand:
            # Duration of the candidate in milliseconds
            dur = ( (tridx[end]-tridx[start]+1)/fs ) * 1000
            if dur < thr_dur:
                continue # Failed Threshold 1
            
            min_sdiff = np.min(sdiff[start:end])
            if min_sdiff > thr2:
                continue # Failed Threshold 2
            
            mean_amp =  np.mean(eegh[tridx[start]:tridx[end]+1])
            if mean_amp < thr3:
                continue # Failed Threshold 3
            
            t_a = tridx[start] + offset
            t_b = np.min((tridx[end] + offset, rem_end * fs))

            ph_idx = (t_a, t_b+1)
            phREM[rem_idx].append(ph_idx)
    
    return phREM

def _compute_hilbert(sig, fs, pass_type, f_range, remove_edges=False):
    """
    Applies a filter and Hilbert transform.
    Returns instantaneous amplitude and phase of the signal.
    """
    sig = filter_signal(sig, fs, pass_type=pass_type, f_range=f_range, remove_edges=remove_edges)
    sig = hilbert(sig)
    return np.abs(sig), np.angle(sig)


def _compute_thresholds(rem, fs, smooth_filt):
    trdiff_list = []
    rem_eeg = []
    eeg_seq = {}
    sdiff_seq = {}
    tridx_seq = {}

    for idx in rem:
        epoch = rem[idx]
        amp, phase = _compute_hilbert(epoch, fs, pass_type="bandpass", f_range=(5, 12), remove_edges=False)

        # amplitude of the entire REM sleep
        rem_eeg += amp

        # trough indices
        tridx = _detect_troughs(phase, -3)

        # differences between troughs
        trdiff = np.diff(tridx)

        # smoothed trough differences
        sdiff_seq[idx] = np.convolve(trdiff, smooth_filt, 'same')

        # dict of trough differences for each REM period
        tridx_seq[idx] = tridx

        eeg_seq[idx] = amp
    
        # differences between troughs
        trdiff_list += list(trdiff)

        
    rem_eeg = np.array(rem_eeg)
    trdiff = np.array(trdiff_list)
    trdiff_sm = np.convolve(trdiff, smooth_filt, 'same')

    # potential candidates for phasic REM:
    # the smoothed difference between troughs is less than
    # the 10th percentile:
    thr1 = np.percentile(trdiff_sm, 10)
    # the minimum smoothed difference in the candidate phREM is less than
    # the 5th percentile
    thr2 = np.percentile(trdiff_sm, 5)
    # the peak amplitude is larger than the mean of the amplitude
    # of the REM EEG.
    thr3 = rem_eeg.mean()

    return thr1, thr2, thr3, tridx_seq, eeg_seq

def _detect_troughs(signal, thr):
    lidx  = np.where(signal[0:-2] > signal[1:-1])[0]
    ridx  = np.where(signal[1:-1] <= signal[2:])[0]
    thidx = np.where(signal[1:-1] < thr)[0]
    sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
    return sidx