# eventlab

A small toolkit to build event catalogs and delay time distributions (DTDs) from COSMIC output

## Install

```bash
pip install .
```

## Contents

- `eventlab.events` - Define events over a `bcm` (or `bcm`-like) dataframe via a boolean mask created with the helper `build_mask` or your own custom mask. Extract contiguous time intervals per binary/star where the mask holds and select summary statistics to report on columns of interest during the events with `get_events`.

- `eventlab.dtd` - Properly bin those event intervals into a delay time distribution (DTD) with `get_event_bin_counts`. Intervals that span bins are split and their duration in each bin is treated independently. DTD power with units of $M_\odot^{-1}$ in bin i ($\Psi_i$) is defined as: 

$$
\Psi_i = \sum_{\text{events} \in \text{bin } i} \frac{\text{event duration}}{\text{duration of bin } i \times M_{\text{sample}}}
$$

- `demo/` - Example notebooks showing usage end-to-end.
