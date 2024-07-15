from .utils import *
from scipy.signal import hilbert
from neurodsp.filt import filter_signal

def detect_phasic(eeg, hypno, fs):

    rem_seq = get_sequences(np.where(hypno == 5)[0])
    rem_idx = [(start * fs, (end+1) * fs) for start, end in rem_seq]

    rem_idx = ensure_duration(rem_idx, min_dur=3)
    if len(rem_idx) == 0:
        raise ValueError("No REM epochs greater than min_dur.")

    # get REM segments
    rem_epochs = get_segments(rem_idx, eeg)

    # Combine the REM indices with the corresponding downsampled segments
    rem = {seq:seg for seq, seg in zip(rem_seq, rem_epochs)}

    w1 = 5.0
    w2 = 12.0
    nfilt = 11
    thr_dur = 900

    trdiff_list = []
    rem_eeg = np.array([])
    eeg_seq = {}
    sdiff_seq = {}
    tridx_seq = {}
    filt = np.ones((nfilt,))
    filt = filt / filt.sum()

    for idx in rem:
        start, end = idx

        epoch = rem[idx]
        epoch = filter_signal(epoch, fs, 'bandpass', (w1,w2), remove_edges=False)
        epoch = hilbert(epoch)

        inst_phase = np.angle(epoch)
        inst_amp = np.abs(epoch)

        # trough indices
        tridx = _detect_troughs(inst_phase, -3)

        # differences between troughs
        trdiff = np.diff(tridx)

        # smoothed trough differences
        sdiff_seq[idx] = np.convolve(trdiff, filt, 'same')

        # dict of trough differences for each REM period
        tridx_seq[idx] = tridx

        eeg_seq[idx] = inst_amp

        # differences between troughs
        trdiff_list += list(trdiff)

        # amplitude of the entire REM sleep
        rem_eeg = np.concatenate((rem_eeg, inst_amp)) 

    trdiff = np.array(trdiff_list)
    trdiff_sm = np.convolve(trdiff, filt, 'same')

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

    phasicREM = {rem_idx:[] for rem_idx in rem.keys()}

    for rem_idx in tridx_seq:
        rem_start, rem_end = rem_idx
        offset = rem_start * fs

        # trough indices
        tridx = tridx_seq[rem_idx]

        # smoothed trough interval
        sdiff = sdiff_seq[rem_idx]

        # amplitude of the REM epoch
        eegh = eeg_seq[rem_idx]

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
            phasicREM[rem_idx].append(ph_idx)

    return phasicREM

def _detect_troughs(signal, thr):
    lidx  = np.where(signal[0:-2] > signal[1:-1])[0]
    ridx  = np.where(signal[1:-1] <= signal[2:])[0]
    thidx = np.where(signal[1:-1] < thr)[0]
    sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
    return sidx