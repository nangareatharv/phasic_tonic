import numpy as np
import logging

from scipy.signal import hilbert
from neurodsp.filt import filter_signal

logger = logging.getLogger('runtime')

def get_sequences(x, ibreak=1):
    """
    Identifies contiguous sequences.

    Parameters:
    x (np.ndarray): 1D time series.
    ibreak (int): A threshold value for determining breaks between sequences (default is 1).

    Returns:
    list of tuples: Each tuple contains the start and end integer of each contiguous sequence.
    """
    if len(x) == 0:
        return []

    diff = np.diff(x)
    breaks = np.where(diff > ibreak)[0]

    # Append the last index to handle the end of the array
    breaks = np.append(breaks, len(x) - 1)
    
    sequences = []
    start_idx = 0
    
    for break_idx in breaks:
        end_idx = break_idx
        sequences.append((x[start_idx], x[end_idx]))
        start_idx = end_idx + 1
    
    return sequences

def get_segments(idx, signal):
    """
    Extracts segments of the signal between specified start and end time indices.

    Parameters:
    idx (list of tuples): Each tuple contains (start_time, end_time).
    signal (np.ndarray): The signal from which to extract segments.

    Returns:
    list of np.ndarray: Each element is a segment of the signal corresponding to the given time ranges.
    """
    segments = []
    for (start_time, end_time) in idx:
        if end_time > len(signal):
            end_time = len(signal) - 1
        segment = signal[start_time:end_time]
        segments.append(segment)
    
    return segments

def get_tonic(rem_start, rem_end, phasic):
  tonic_seg = []
  current_start = rem_start

  for ph_start, ph_end in phasic:
    # A gap between current start and start of a phasic episode
    if current_start < ph_start:
      tonic_seg.append((current_start, ph_start))

    # Update current start
    current_start = max(current_start, ph_end)

  # After the last phasic episode there might be a remaining tonic episode
  if current_start < rem_end:
    tonic_seg.append((current_start, rem_end))

  return tonic_seg

def create_hypnogram(phasicREM, length):
       binary_hypnogram = np.zeros(length, dtype=int)
       for start, end in phasicREM:
           binary_hypnogram[start:end] = 1
       return binary_hypnogram

# bug
def ensure_duration(rem_idx, min_dur):
    for rem_start, rem_end in rem_idx:
      if(rem_end-rem_start) < min_dur:
        logger.debug("Removing REM epoch: ({0}, {1})".format(rem_start, rem_end))
        rem_idx.remove(rem_idx)

    if len(rem_idx) == 0:
      raise ValueError("No REM epochs greater than min_dur.")
    return rem_idx

def _detect_troughs(signal, thr):
    lidx  = np.where(signal[0:-2] > signal[1:-1])[0]
    ridx  = np.where(signal[1:-1] <= signal[2:])[0]
    thidx = np.where(signal[1:-1] < thr)[0]
    sidx = np.intersect1d(lidx, np.intersect1d(ridx, thidx))+1
    return sidx

def _despine_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

def phasic_detect(rem, fs, thr_dur=900, nfilt=11):
    w1 = 5.0
    w2 = 12.0

    trdiff_list = []
    rem_eeg = np.array([])
    eeg_seq = {}
    sdiff_seq = {}
    tridx_seq = {}
    filt = np.ones((nfilt,))
    filt = filt / filt.sum()
    all_empty = True

    for idx in rem:
        start, end = idx

        epoch = rem[idx]
        
        if epoch.size == 0:  # Check if the epoch is empty
            continue

        all_empty = False
        
        epoch = filter_signal(epoch, fs, 'bandpass', (w1,w2), remove_edges=False)
        epoch = hilbert(epoch)

        inst_phase = np.angle(epoch)
        inst_amp = np.abs(epoch)

        # trough indices
        tridx = _detect_troughs(inst_phase, -3)

        # alternative version:
        #tridx = np.where(np.diff(np.sign(np.diff(eegh))))[0]+1

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
    
    if all_empty:  # Check if all epochs were empty
        logger.debug("All epochs are empty. Returning empty result.")
        return {}
    
    trdiff = np.array(trdiff_list)
    trdiff_sm = np.convolve(trdiff, filt, 'same')

    # potential candidates for phasic REM:
    # the smoothed difference between troughs is less than
    # the 10th percentile:
    thr1 = np.percentile(trdiff_sm, 10)
    # the minimum difference in the candidate phREM is less than
    # the 5th percentile
    thr2 = np.percentile(trdiff_sm, 5)
    # the peak amplitude is larger than the mean of the amplitude
    # of the REM EEG.
    thr3 = rem_eeg.mean()

    logger.debug("Thresholds: thr1 = {0:.3f}, thr2 = {1:.3f}, thr3 = {2:.3f}".format(thr1, thr2, thr3))

    phrem = {rem_idx:[] for rem_idx in rem.keys()}

    for rem_idx in tridx_seq:
        rem_start, rem_end = rem_idx
        offset = rem_start * fs

        # trough indices
        tridx = tridx_seq[rem_idx]

        # smoothed trough interval
        sdiff = sdiff_seq[rem_idx]

        # ampplitude of the REM epoch
        eegh = eeg_seq[rem_idx]

        cand_idx = np.where(sdiff <= thr1)[0]
        cand = get_sequences(cand_idx)

        logger.debug("Candidates: {0}".format(str(cand)))
        for start, end in cand:
            dur = ( (tridx[end]-tridx[start]+1)/fs ) * 1000
            if dur > thr_dur and np.min(sdiff[start:end]) < thr2 and np.mean(eegh[tridx[start]:tridx[end]+1]) > thr3:
                a = tridx[start]   + offset
                b = tridx[end]  + offset
                
                if b > (rem_end * fs):
                    b = rem_end*fs

                ph_idx = (a, b+1)
                phrem[rem_idx].append(ph_idx)
    return phrem