# Naming Convention for Predicted Masks

For evaluation of vegetation masks, please follow the naming convention described below.

## Ground Truth Images

Ground truth images follow the pattern:

```
img_<ddd>_<split>.png
```

- `<ddd>` — a zero-padded three-digit number (e.g., `001`, `042`)
- `<split>` — the dataset split: `train`, `val`, or `test`

## Predicted Masks

Name your predicted masks using the corresponding pattern:

```
mask_pred_<ddd>_<split>.png
```

The number and split must match the ground truth image you are predicting for.

**Example:** The prediction for `img_012_test.png` should be named `mask_pred_012_test.png`.

## Running the Evaluation

Place all predicted masks in a single folder and pass the path to that folder when calling `run_veg_eval.py`.

