# breakhis-reliability

**How much of the reported accuracy on breast cancer histopathology survives honest evaluation?**

Many models trained on the BreakHis dataset report 95–99% accuracy. But BreakHis has only 82 patients
(24 benign, 58 malignant) and ~7,900 images, so when images are split at random, the same patient's tissue
appears in both training and test sets. This project measures that effect and then builds the tools a
clinical model needs before anyone should trust it.

![tests](https://github.com/amubakekengbam/breakhis-reliability/actions/workflows/ci.yml/badge.svg)

## Roadmap

| # | Question | Status |
|---|---|---|
| 1 | **Leakage:** image-level vs patient-level splits, same model, multiple seeds | 🚧 in progress |
| 2 | **Strong baseline:** patient-grouped 5-fold CV, confidence intervals, image- and patient-level metrics | planned |
| 3 | **Magnification shift:** train at one magnification, test at another — how much does performance drop? | planned |
| 4 | **Label-free shift detection:** can we raise an alarm from model confidence / features alone? | planned |
| 5 | **Safe predictions:** calibration + abstain-when-unsure (conformal prediction) | planned |
| 6 | **Packaging:** Docker, small demo app, model card | planned |

## Results

_Coming soon — first results from experiment 1._

## Quick start (Google Colab, free GPU)

Open `notebooks/01_leakage_colab.ipynb` in Colab (File → Open notebook → GitHub → `amubakekengbam/breakhis-reliability`),
set the runtime to a T4 GPU, and run all cells.

Locally (no GPU needed for tests):

```bash
pip install -e ".[dev]"
pytest -q
```

## Design choices

- **Patient-level everything:** splits, bootstrap confidence intervals (resampling patients, not images),
  and patient-level metrics (mean probability per patient).
- **Balanced accuracy and AUROC**, not just accuracy — the classes are imbalanced (~69% malignant images).
- **Same model, same settings, only the split changes** in the leakage study, so any gap is due to the split.
- **Results are saved after every run**, so long experiments can resume.

## Project layout

```
src/breakhis/
  data.py          # parse BreakHis file names -> metadata (patient, label, magnification)
  splits.py        # image-level (leaky) and patient-level (honest) splits
  metrics.py       # image/patient metrics, patient-level bootstrap CIs
  train.py         # PyTorch dataset, ResNet18/EfficientNet-B0, training loop
  experiments/
    leakage.py     # experiment 1
notebooks/
  01_leakage_colab.ipynb
tests/
```

## Data and citation

BreakHis is provided by the VRI lab, UFPR, for non-commercial research. Download it from the
[official page](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
and cite:

> Spanhol, F., Oliveira, L. S., Petitjean, C., Heutte, L. "A Dataset for Breast Cancer Histopathological
> Image Classification." IEEE Transactions on Biomedical Engineering, 63(7), 1455–1462, 2016.

The dataset itself is **not** included in this repository.

## Limitations

Research code, not a medical device. 82 patients from a single lab is a small, narrow sample;
results will not transfer to other hospitals, scanners or staining protocols without validation.

## Author

Deepak Singh Kekengbam — MSc Artificial Intelligence, University of Aberdeen.
