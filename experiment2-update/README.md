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
| 1 | **Leakage:** image-level vs patient-level splits, same model, multiple seeds | ✅ done |
| 2 | **Honest baseline:** 5-fold patient CV at every magnification, grouped by patient ID *and* by patient number, per-tumor-type results | 🚧 running |
| 3 | **Magnification shift:** train at one magnification, test at another — how much does performance drop? | planned |
| 4 | **Label-free shift detection:** can we raise an alarm from model confidence / features alone? | planned |
| 5 | **Safe predictions:** calibration + abstain-when-unsure (conformal prediction) | planned |
| 6 | **Packaging:** Docker, small demo app, model card | planned |

## Results

### Experiment 1 — splitting by image inflates accuracy

ResNet18 (ImageNet-pretrained), 200x images (2,013 images, 81 patient IDs), 5 epochs, 3 seeds.
Same model and settings; only the train/test split changes. Mean ± std over seeds.

| Split | Accuracy | Balanced accuracy | AUROC | Patient IDs in both train and test |
|---|---|---|---|---|
| By image (common practice) | 95.1% ± 1.1 | 94.8% ± 1.1 | 0.989 ± 0.002 | 79–81 |
| By patient | **82.3% ± 4.2** | **87.3% ± 4.8** | **0.954 ± 0.041** | 0 |

When test patients are truly unseen, accuracy drops by about 13 points and varies four times as much
between seeds. Each patient-split test fold holds only 16 patients, 2–4 of them benign, so single-split
results on BreakHis are very noisy. Experiment 2 uses all five folds so that every patient is tested once.

### What we found in the data

Checked before trusting any split (`breakhis.data.id_report`):

- **81 patient IDs, not 82.** Image counts match the paper exactly (7,909: 2,480 benign, 5,429 malignant),
  but ID `14-13412` is filed under both ductal and lobular carcinoma, so it is one ID counted twice.
- **9 patient numbers are shared by several IDs**, e.g. `14-22549AB`, `14-22549CD` and `14-22549G`, all
  adenosis. Most share a tumor type, so they are probably slides from the same person. Splitting by ID then
  still puts the same number in train and test (3 shared numbers in most folds). Experiment 2 also groups
  by the number alone.
- **Small, unbalanced subgroups.** Ductal carcinoma is 38 of 57 malignant IDs (64% of malignant images);
  phyllodes tumor has 3 patient IDs and adenosis 4. Overall accuracy can hide poor results on rare types,
  so we report results per tumor type.
- **No normal tissue.** Every image is a benign or malignant *tumor*, so these models cannot screen normal biopsies.

## Quick start (Google Colab, free GPU)

Open a notebook straight in Colab, set the runtime to a T4 GPU, and run all cells:

- [Experiment 1 — leakage](https://colab.research.google.com/github/amubakekengbam/breakhis-reliability/blob/main/notebooks/01_leakage_colab.ipynb)
- [Experiment 2 — patient cross-validation](https://colab.research.google.com/github/amubakekengbam/breakhis-reliability/blob/main/notebooks/02_patient_cv_colab.ipynb)

Results are saved to Google Drive after every run, so a disconnect never loses finished work.

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
    cv.py          # experiment 2
notebooks/
  01_leakage_colab.ipynb
  02_patient_cv_colab.ipynb
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

Research code, not a medical device. About 70–81 patients from a single lab is a small, narrow sample;
results will not transfer to other hospitals, scanners or staining protocols without validation.

## Author

Deepak Singh Kekengbam — MSc Artificial Intelligence, University of Aberdeen.
