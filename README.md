# OwnDPDLab — Open-Source Digital Pre-Distortion Testbed

A hardware-in-the-loop DPD evaluation framework using a Xilinx RFSoC 4×2 software-defined radio. The testbed measures, trains, and evaluates polynomial and neural-network DPD models directly on real PA hardware.

## Overview

The main notebook `experiment.ipynb` covers the full workflow:

1. **RF front-end configuration** — sets Nyquist zone, NCO frequency, and output power via TCP.
2. **Signal generation** — creates a wideband OFDM test signal (up to 256-QAM).
3. **Baseline measurement** — captures the PA output without pre-distortion.
4. **DPD training** — fits a Generalized Memory Polynomial (GMP) model via least squares.
5. **Pre-distorted measurement** — applies DPD and re-measures the PA output.
6. **Publication figures** — generates AM-AM/AM-PM, PSD, and constellation plots.
7. **Metrics export** — computes NMSE, ACPR, EVM, and OOB-NMSE into a CSV table.

## Repository structure

```
experiment.ipynb           Main notebook (run this)
DPD.py                    GMP and polynomial DPD models
TransferFunction.py       RFSoC TCP interface and MLS-synchronized acquisition
SignalSource.py           OFDM baseband signal generator
cfr.py                    Crest Factor Reduction (hard/soft clipping)
Metrics.py                NMSE, EVM, ACPR, OOB-NMSE
messdaten_zone1.npz       Measurement data — Nyquist Zone 1
messdaten_zone2.npz       Measurement data — Nyquist Zone 2
Figures_Paper/            Output directory for publication-ready SVG figures
RFSoC/pynq_driver.ipynb  PYNQ driver running on the RFSoC board
RFSoC/*.hwh, *.bit        Hardware hand-off and bitstream files (upload pending)
```

## Requirements

```
pip install -r requirements.txt
```

- Python ≥ 3.10
- numpy, scipy, matplotlib

## Hardware

- **RFSoC 4×2** (Xilinx/AMD) running the companion PYNQ driver (`RFSoC/pynq_driver.ipynb`).
- The board must be reachable at `192.168.3.1:5555` (configurable in the notebook).
- The Vivado bitstream (`.bit`) and hardware hand-off file (`.hwh`) will be uploaded soon.
- The measurement data files (`messdaten_zone*.npz`) allow running the plotting and metrics cells **without hardware** and optimize plots without connected hardware afterwards.

## Quick start (offline, pre-recorded data)

```bash
pip install -r requirements.txt
jupyter notebook experiment.ipynb
```

Skip Cell 1 (hardware acquisition) and run Cell 2 directly — it loads `messdaten_zone1.npz` / `messdaten_zone2.npz` and produces all figures.

## DPD models

| Class | Description |
|-------|-------------|
| `DPD.GMP` | Generalized Memory Polynomial — aligned, leading, and lagging cross-terms |

Training solvers available via the `method` keyword: `'ls'` (default), `'tikhonov'`, `'lasso'`, `'rls'`, `'lms'`, `'robust'`.

## Note on AI assistance

Parts of the code in this repository were developed with the assistance of AI tools (Claude by Anthropic). All generated code has been reviewed and validated against hardware measurements.

## License

See [LICENSE](LICENSE).
