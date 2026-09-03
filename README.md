# eventlab

A small toolkit to build event catalogs and DTDs from COSMIC output

## Install

```bash
pip install .
```

## Contents

- `eventlab.events` - define events with user provided boolean conditions over `bcm` (or `bcm`-like) columns and extract contiguous time intervals per binary/star where the mask holds (`build_mask`, `get_events`).

- `eventlab.dtd` - bin those intervals into a DTD, correctly splitting duration across bin edges and weighting by bin size and population mass (`get_event_bin_counts`).

- `demo/` - example notebooks showing usage end-to-end.
