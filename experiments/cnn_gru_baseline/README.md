# CNN-GRU Baseline

EXP-001 is the initial CNN-GRU experiment trained on the selected Jester
gesture classes. The model uses the same pretrained MobileNetV2 frame encoder
as CNN baseline EXP-004 and a GRU to process frame features in order.

```text
16 RGB frames (112 x 112)
  -> MobileNetV2 frame encoder (ImageNet pretrained)
  -> GRU (64 units)
  -> Dense (128 units, ReLU) -> Dropout (0.3)
  -> Dense (7 classes, softmax)
```

## EXP-001 results

| Setting or result | Value |
|---|---|
| Training / validation | 8,000 training videos per epoch / full validation split (5,946 videos) |
| Input preprocessing | Shared loader; 16 frames, RGB, 112 x 112, values 0 to 1 |
| Augmentation | Training only: brightness and small rotation, consistent within each clip |
| Optimizer / learning rate | Adam / 0.0001 |
| Batch size | 8 |
| Seed | 42 |
| Best epoch | 11, selected by lowest validation loss |
| Validation accuracy at best epoch | 92.62% |
| Validation loss at best epoch | 0.2362 |
| Epochs completed | 17; early stopping patience 6 |
| Training time | 250.6 minutes on a Google Colab GPU |
| Total parameters | 677,863 |
| Versions | TensorFlow 2.21.0, Keras 3.13.2 |

The best validation accuracy across the logged epochs was 92.97% at epoch 17,
but its validation loss (0.2525) was higher than at epoch 11. The saved best
checkpoint is therefore the epoch 11 model. Epoch numbers in `history.csv`
start at zero; the epoch numbers above start at one.

Training loss continued to decrease after epoch 11 while validation loss stayed
above its best value, indicating some overfitting in the later epochs. The
held-out test split was not used in this experiment.

CNN baseline EXP-004 reported 74.62% validation accuracy and 0.6602 validation
loss at its best-loss epoch. This is a preliminary comparison: both runs used
the same splits and data limits, but the exact training samples and random
augmentation may differ. Member 4 should verify the comparison and assess
per-class errors and efficiency.

## Files and handoff

- `EXP-001/history.csv`: per-epoch training and validation metrics.
- `EXP-001/curves.png`: loss and accuracy curves.
- `EXP-001/run_info.json`: original Colab run metadata and settings; paths in
  this file describe the original Colab session and must be adapted elsewhere.
- `EXP-000-real-smoke/run_info.json`: metadata from the one-epoch real-data
  pipeline check; its metrics are not a meaningful performance result.
- Best checkpoint: `cnn_gru_EXP-001.keras`, saved outside Git and shared
  separately. The original Colab run saved it under Google Drive's
  `GestureController/checkpoints/` folder.

The model and training code are in `src/gesture_recognition/models/cnn_gru.py`
and `src/gesture_recognition/training/train_cnn_gru.py`. Configuration is in
`configs/cnn_gru.yaml`. Set the data path and output directories for the local
environment before reproducing a run, and use a new experiment ID so existing
results and checkpoints are not overwritten.
