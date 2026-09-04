import numpy as np
import pandas as pd

VALID_OPS = {'greater', 'less', 'greater_equal', 'less_equal', 'equal', 'not_equal'}
VALID_STATS = {'max', 'min', 'median', 'mean'}


def build_mask(events, mask_list, mask_logic='and'):
    """
    Combine a list of (column, op, val) conditions into a single boolean mask
    over `events`, using exact column names (no renaming/stripping).

    mask_list : list of (column, op, val) tuples, e.g. ('mass_1', 'greater', 5)
    mask_logic : 'and' or 'or'

    Users can call this directly, or build their own boolean array/Series
    by hand (e.g. combining conditions with arbitrary logic) and pass that
    straight into get_events -- build_mask is just a convenience for the
    common tuple-list case.
    """
    if mask_logic not in ('and', 'or'):
        raise ValueError("mask_logic must be 'and' or 'or'")

    if not mask_list:
        return np.ones(len(events), dtype=bool)

    mask = np.ones(len(events), dtype=bool) if mask_logic == 'and' else np.zeros(len(events), dtype=bool)

    for col, op, val in mask_list:
        if col not in events.columns:
            raise KeyError(f"Column '{col}' not found in events dataframe")

        if op not in VALID_OPS:
            raise ValueError(f"Unsupported operation '{op}'. Must be one of {VALID_OPS}")

        if op == 'greater':
            cond = events[col] > val
        elif op == 'less':
            cond = events[col] < val
        elif op == 'greater_equal':
            cond = events[col] >= val
        elif op == 'less_equal':
            cond = events[col] <= val
        elif op == 'equal':
            cond = events[col] == val
        elif op == 'not_equal':
            cond = events[col] != val
        else:
            raise ValueError(f'Unsupported operation: {op}')

        mask = (mask & cond) if mask_logic == 'and' else (mask | cond)

    return np.asarray(mask)


def _validate_stats_dict(stats_dict, available_columns):
    """Check that a colname -> [stats] dict only references known columns/stats."""
    if not stats_dict:
        return {}

    missing_cols = set(stats_dict) - set(available_columns)
    if missing_cols:
        raise KeyError(f"Columns not found in bcm: {sorted(missing_cols)}")

    for col, stat_list in stats_dict.items():
        invalid = set(stat_list) - VALID_STATS
        if invalid:
            raise ValueError(f"Unsupported stats {invalid} for column '{col}'. Must be from {VALID_STATS}")

    return stats_dict


def _summarize_block(block, stats_dict):
    """Build one interval record: start/end time plus requested column stats."""
    record = {
        'tphys_start': float(block['tphys'].iloc[0]),
        'tphys_end': float(block['tphys'].iloc[-1]),
    }
    for col, stat_list in stats_dict.items():
        values = block[col]
        for stat in stat_list:
            if stat == 'max':
                record[f'{col}_max'] = float(values.max())
            elif stat == 'min':
                record[f'{col}_min'] = float(values.min())
            elif stat == 'median':
                record[f'{col}_median'] = float(values.median())
            elif stat == 'mean':
                record[f'{col}_mean'] = float(values.mean())
    return record


def intervals_for_bin_num(group, stats_dict=None):
    """
    Find contiguous stretches (in sorted tphys order) where group['_mask']
    is True, for a single binary system. group must already contain a
    boolean '_mask' column. Returns a list of interval dicts.
    """
    if len(group) == 0:
        return []

    stats_dict = stats_dict or {}

    group = group.sort_values('tphys').reset_index(drop=True)
    selected_positions = np.flatnonzero(group['_mask'].to_numpy())

    if len(selected_positions) == 0:
        return []

    intervals = []
    start_pos = selected_positions[0]
    prev_pos = selected_positions[0]

    for pos in selected_positions[1:]:
        if pos != prev_pos + 1:
            block = group.iloc[start_pos:prev_pos + 1]
            intervals.append(_summarize_block(block, stats_dict))
            start_pos = pos
        prev_pos = pos

    block = group.iloc[start_pos:prev_pos + 1]
    intervals.append(_summarize_block(block, stats_dict))

    return intervals


def intervals_for_star(bcm, star, mask, stats_dict=None):
    """
    Compute intervals for one star (1 or 2) across every binary in bcm.

    mask       : boolean array/Series, same length as bcm, aligned by
                 position (e.g. the output of build_mask(bcm, ...), or any
                 boolean array/Series the user constructs themselves).
    star       : label (1 or 2) attached to output rows.
    stats_dict : dict of {column: [stat, ...]} -- exact bcm column names,
                 e.g. {'teff_1': ['min', 'max'], 'lum_1': ['mean']}.
    """
    if star not in (1, 2):
        raise ValueError('star must be 1 or 2')

    mask = np.asarray(mask)
    if len(mask) != len(bcm):
        raise ValueError(f"mask length ({len(mask)}) does not match bcm length ({len(bcm)})")

    stats_dict = _validate_stats_dict(stats_dict, bcm.columns)

    required_cols = set(stats_dict) | {'tphys', 'bin_num'}
    missing = required_cols - set(bcm.columns)
    if missing:
        raise KeyError(f"Columns not found in bcm: {sorted(missing)}")

    events = bcm[list(required_cols)].copy()
    events['_mask'] = mask

    results = []
    for bin_num, group in events.groupby('bin_num', sort=False):
        intervals = intervals_for_bin_num(group, stats_dict=stats_dict)
        for interval in intervals:
            results.append({'bin_num': bin_num, 'star': star, **interval})

    return results


def get_events(bcm, primary_mask=None, secondary_mask=None,
                primary_stats=None, secondary_stats=None):
    """
    Generate an event dataframe with intervals for each star and optional statistics.

    Parameters
    ----------
    primary_mask: pd.Series
        boolean series -- same length as bcm -- defines
        event windows for star 1. Build with build_mask(...)
        or construct by hand for arbitrary custom logic.
        None/omitted to skip star 1.
    secondary_mask: pd.Series
        Same as primary_mask, but for star 2. None/omitted
        to skip star 2.
    primary_stats: dict
        Dictionary of {column: [stat, ...]} for star 1 intervals,
        e.g. {'teff_1': ['min', 'max'], 'lum_1': ['mean']}.
        stat options: 'max', 'min', 'median', 'mean'.
    secondary_stats : dict
        Same as primary_stats for star 2 intervals, e.g. {'teff_2': [...]}.

    Returns
    -------
    event_df: pd.DataFrame
        Event dataframe with columns: bin_num, star, tphys_start, tphys_end,
        plus any stats dict columns in the format *colname*_*stat*.
    """
    records = []
    if primary_mask is not None:
        records.extend(intervals_for_star(
            bcm, star=1, mask=primary_mask, stats_dict=primary_stats,
        ))
    if secondary_mask is not None:
        records.extend(intervals_for_star(
            bcm, star=2, mask=secondary_mask, stats_dict=secondary_stats,
        ))

    base_cols = ['bin_num', 'star', 'tphys_start', 'tphys_end']
    if not records:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(records)
    other_cols = sorted(c for c in df.columns if c not in base_cols)
    df = df[base_cols + other_cols]

    return df.sort_values(['bin_num', 'star', 'tphys_start']).reset_index(drop=True)