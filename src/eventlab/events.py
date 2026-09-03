import numpy as np
import pandas as pd

VALID_OPS = {'greater', 'less', 'greater_equal', 'less_equal', 'equal', 'not_equal'}
VALID_STATS = {'max', 'min', 'median', 'mean'}

def create_mask(column, op, val):
    """
    Build (and validate) a single mask condition tuple: (column, op, val).
    Purely a convenience/validation wrapper -- users can just write the
    tuple by hand if they prefer.

    Example
    -------
    create_mask('mass_1', 'greater', 5)  -> ('mass_1', 'greater', 5)
    """
    if op not in VALID_OPS:
        raise ValueError(f"Unsupported operation '{op}'. Must be one of {sorted(VALID_OPS)}")
    return (column, op, val)


def build_mask(events, mask_list, mask_logic='and'):
    """
    Combine a list of (column, op, val) conditions into a single boolean mask
    over `events`, using exact column names (no renaming/stripping).
    """
    if mask_logic not in ('and', 'or'):
        raise ValueError("mask_logic must be 'and' or 'or'")

    if not mask_list:
        # no conditions -> everything passes ('and') 
        return np.ones(len(events), dtype=bool)

    mask = np.ones(len(events), dtype=bool) if mask_logic == 'and' else np.zeros(len(events), dtype=bool)

    for col, op, val in mask_list:
        if col not in events.columns:
            raise KeyError(f"Column '{col}' not found in events dataframe")

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

    return mask


def _summarize_block(block, stat_columns, stats):
    """Build one interval record: start/end time plus requested column stats."""
    record = {
        'tphys_start': float(block['tphys'].iloc[0]),
        'tphys_end': float(block['tphys'].iloc[-1]),
    }
    for col in stat_columns:
        values = block[col]
        for stat in stats:
            if stat == 'max':
                record[f'{col}_max'] = float(values.max())
            elif stat == 'min':
                record[f'{col}_min'] = float(values.min())
            elif stat == 'median':
                record[f'{col}_median'] = float(values.median())
            elif stat == 'mean':
                record[f'{col}_mean'] = float(values.mean())
    return record


def intervals_for_bin_num(events, mask_list, mask_logic='and', stat_columns=None, stats=None):
    """
    Find contiguous stretches (in sorted tphys order) where the mask holds,
    for a single binary system. Returns a list of interval dicts.
    """
    if len(events) == 0:
        return []

    stat_columns = stat_columns or []
    stats = stats or []

    events = events.sort_values('tphys').reset_index(drop=True)
    mask = build_mask(events, mask_list, mask_logic=mask_logic)
    selected_positions = np.flatnonzero(mask)

    if len(selected_positions) == 0:
        return []

    intervals = []
    start_pos = selected_positions[0]
    prev_pos = selected_positions[0]

    for pos in selected_positions[1:]:
        if pos != prev_pos + 1:
            block = events.iloc[start_pos:prev_pos + 1]
            intervals.append(_summarize_block(block, stat_columns, stats))
            start_pos = pos
        prev_pos = pos

    block = events.iloc[start_pos:prev_pos + 1]
    intervals.append(_summarize_block(block, stat_columns, stats))

    return intervals


def intervals_for_star(bcm, star, mask_list, mask_logic='and', stats=None):
    """
    Compute intervals for one star (1 or 2) across every binary in bcm.
    mask_list must use exact bcm column names (e.g. 'mass_1', 'kstar_2').
    `star` is just the label attached to output rows -- it does not force
    mask_list to reference _1 or _2 columns, so use it deliberately.
    """
    if star not in (1, 2):
        raise ValueError('star must be 1 or 2')
    if not mask_list:
        return []

    stats = stats or []

    # every column referenced in the mask gets its own summary stats
    stat_columns = sorted({col for col, _, _ in mask_list})

    required_cols = set(stat_columns) | {'tphys', 'bin_num'}
    missing = required_cols - set(bcm.columns)
    if missing:
        raise KeyError(f"Columns not found in bcm: {sorted(missing)}")

    events = bcm[list(required_cols)].copy()

    results = []
    for bin_num, group in events.groupby('bin_num', sort=False):
        intervals = intervals_for_bin_num(
            group,
            mask_list=mask_list,
            mask_logic=mask_logic,
            stat_columns=stat_columns,
            stats=stats,
        )
        for interval in intervals:
            results.append({'bin_num': bin_num, 'star': star, **interval})

    return results


def get_events(bcm, primary_mask=None, secondary_mask=None,
                primary_logic='and', secondary_logic='and',
                stats=None):
    """
    User-facing entry point.

    primary_mask   : list of (column, op, val) -- defines event windows for star 1.
    secondary_mask : list of (column, op, val) -- defines event windows for star 2.
                     (Either can be None/[] to skip that star entirely.)
    stats          : optional subset of ['max', 'min', 'median', 'mean'].
                      Duplicates are silently ignored. When given, adds
                      <col>_<stat> columns for every column referenced in
                      the mask that produced each row.

    Returns
    -------
    pd.DataFrame with columns: bin_num, star, tphys_start, tphys_end,
    plus any requested <col>_<stat> columns.
    """
    if stats:
        invalid = set(stats) - VALID_STATS
        if invalid:
            raise ValueError(f"Unsupported stats {invalid}. Must be from {VALID_STATS}")
        stats = list(dict.fromkeys(stats))  # de-dupe, keep order
    else:
        stats = []

    records = []
    if primary_mask:
        records.extend(intervals_for_star(bcm, star=1, mask_list=primary_mask,
                                           mask_logic=primary_logic, stats=stats))
    if secondary_mask:
        records.extend(intervals_for_star(bcm, star=2, mask_list=secondary_mask,
                                           mask_logic=secondary_logic, stats=stats))

    base_cols = ['bin_num', 'star', 'tphys_start', 'tphys_end']
    if not records:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(records)
    other_cols = sorted(c for c in df.columns if c not in base_cols)
    df = df[base_cols + other_cols]

    return df.sort_values(['bin_num', 'star', 'tphys_start']).reset_index(drop=True)