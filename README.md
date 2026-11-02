# CardioWave Lab

Reproducible, research-only benchmark for classifying **synthetic ECG-like waveforms**. No patient data are used. **Not a medical device and not for clinical diagnosis, monitoring, or treatment.**

## Scope
- Three simulated waveform classes: regular, early-beat pattern, irregular timing.
- Group-disjoint train/validation/test split prevents simulated-subject leakage.
- Handcrafted features and Random Forest baseline.
- Independent additive-noise stress evaluation.
- Macro F1, accuracy and confusion matrices.
- Automated tests and GitHub Actions CI.

## Quickstart
```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m cardiowave --subjects 60 --output reports/metrics.json
```

Data are generated entirely by the MIT-licensed source code in this repository. There are no external data downloads. Results are seed-dependent and describe synthetic pattern recognition only.

## Limitations
Simulated waveforms do not reflect physiological or clinical ECG variability. High scores here do not imply performance on real patients. The additive-noise stress test does not cover realistic artifacts, demographic shifts or device shifts.

## Documentation
See [Turkish learning guide](docs/TR_EGITIM.md), [data card](docs/DATA_CARD.md), and [model card](docs/MODEL_CARD.md).

## License
MIT. See LICENSE.
