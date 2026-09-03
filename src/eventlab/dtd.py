import numpy as np
import pandas as pd

def get_dtd(df, bins, sample_mass):
    """
    Compute the delay-time distribution (DTD) power in user-specified bins,
    from the interval dataframe produced by get_events.

    Parameters
    ----------
    df : pd.DataFrame
        Output of get_events -- must contain 'star', 'tphys_start', 'tphys_end'.
    bins : array-like
        Monotonically increasing bin edges, e.g. [0, 6, 12, 20] -> bins
        [0,6), [6,12), [12,20).
    sample_mass : float
        Total sampled population mass used to normalize power
        (power = duration / (bin_width * sample_mass)).

    Returns
    -------
    pd.DataFrame with columns: bin_start, bin_end, primary, secondary, combined
    """
    bins = np.asarray(bins, dtype=float)
    if len(bins) < 2:
        raise ValueError('bins must have at least 2 edges')
    if np.any(np.diff(bins) <= 0):
        raise ValueError('bins must be strictly increasing')

    bins_with_edges = list(zip(bins[:-1], bins[1:]))

    primary_events = df[df['star'] == 1]
    secondary_events = df[df['star'] == 2]

    def _power_per_bin(events):
        starts = events['tphys_start'].to_numpy()
        ends = events['tphys_end'].to_numpy()

        powers = []
        for t_min, t_max in bins_with_edges:
            # clip each interval to the bin boundaries before summing duration
            overlap_start = np.maximum(starts, t_min)
            overlap_end = np.minimum(ends, t_max)
            clipped_durations = np.clip(overlap_end - overlap_start, 0, None)

            total_duration = clipped_durations.sum()
            bin_width = t_max - t_min
            power = total_duration / (bin_width * sample_mass)
            powers.append(float(power))

        return powers

    primary_dtd = _power_per_bin(primary_events)
    secondary_dtd = _power_per_bin(secondary_events)
    combined_dtd = [p + s for p, s in zip(primary_dtd, secondary_dtd)]

    return pd.DataFrame({
        'bin_start': bins[:-1],
        'bin_end': bins[1:],
        'primary': primary_dtd,
        'secondary': secondary_dtd,
        'combined': combined_dtd,
    })
