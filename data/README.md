# FD001 data

The files in `raw/` were supplied with this project and are preserved byte-for-byte. They match the FD001 schema and counts, but these checks do not establish a historical download date or independently authenticate the supplied bytes against NASA's archive.

Source: [NASA CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data). NASA describes FD001 as simulated turbofan degradation with 100 training and 100 test trajectories, one operating condition, and high-pressure-compressor degradation.

Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08, 2008. The [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) provides dataset context and citation guidance.

## Files and schema

| File | Rows | Meaning |
|---|---:|---|
| `raw/train_FD001.txt` | 20,631 | Complete run-to-failure trajectories for 100 engines |
| `raw/test_FD001.txt` | 13,096 | Truncated histories for 100 different engines |
| `raw/RUL_FD001.txt` | 100 | Remaining cycles at each test engine's last observation, in engine-ID order |

Trajectory files contain 26 whitespace-separated columns: engine ID, cycle, three operating settings, and 21 sensor values. The parser assigns the names in `aircraft_rul.data.COLUMNS`. It rejects duplicate engine/cycle pairs, nonfinite values, and gaps in histories. RUL labels must be finite and nonnegative. Endpoint-label alignment requires engine IDs 1 through 100 for these supplied files.

Training labels are computed as an engine's final cycle minus its current cycle. Test labels are used for evaluation only; inference does not need them. RUL units are cycles, not hours.

## Acquisition and byte checks

To obtain another copy, open the NASA dataset page, follow its `CMAPSSData.zip` resource, and extract `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` into `data/raw/`.
