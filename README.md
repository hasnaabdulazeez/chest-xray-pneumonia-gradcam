# Chest X-ray Pneumonia Detection with Grad-CAM

A DenseNet-121 classifier that flags **pneumonia on pediatric chest X-rays**, with **Grad-CAM heatmaps** showing which regions drove each prediction, and a Streamlit app for interactive review. It uses the public [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kermany et al., 2018; about 5,800 images).

> Research and education project. Not a medical device and not for clinical decisions.

## What makes it more than a notebook

| Problem | What this repo does |
|---|---|
| Official validation split has only 16 images | Carves a **stratified 10% validation set** out of train |
| Classes imbalanced (~3:1 pneumonia : normal) | **WeightedRandomSampler** so each batch is balanced |
| A 0.5 threshold is arbitrary for screening | Picks the **threshold that reaches 95% sensitivity** on validation, then reports specificity, PPV and NPV at that operating point on test |
| Shortcut learning (text markers, devices) | **Grad-CAM** on the last dense block to check that attention falls on the lungs |
| Anatomy matters | No horizontal flips in augmentation |

Training uses AdamW, a one-cycle learning-rate schedule and mixed precision, with model selection by validation ROC AUC.

## Quick start

```bash
pip install -r requirements.txt
# download and extract the Kaggle dataset to data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
python train.py --data data/raw/chest_xray --epochs 12
streamlit run app.py
```

`train.py` writes `checkpoints/best.pt`, `threshold.json` and `report.json`. The report has test AUC plus metrics at both 0.5 and the tuned threshold, so the trade-off is visible.

## Project structure

```
xray/
  data.py      # transforms, stratified val split, class-balanced sampler
  model.py     # DenseNet-121 with a single-logit head; Grad-CAM target layer
  gradcam.py   # hook-based Grad-CAM + heatmap overlay
  metrics.py   # sensitivity/specificity/PPV/NPV, threshold for target sensitivity
train.py       # training loop (AMP, one-cycle LR), evaluation report
app.py         # Streamlit upload → probability + Grad-CAM
tests/         # metric and Grad-CAM unit tests
```

## Known limitations

- The test set comes from one hospital (Guangzhou), so performance will drop on other scanners and adult patients without re-validation.
- Grad-CAM is coarse (7×7 before upsampling). It shows *where* the model looked, not *why*.

## License

MIT. The dataset is CC BY 4.0.
