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

## EXP-002 results

EXP-002 increased the GRU to 128 units and used the updated training-only
brightness, rotation, contrast and saturation augmentation. The CNN and GRU
were both trainable. The remaining recorded settings match EXP-001, including
the data limits, batch size, learning rate, and early-stopping patience.

| Result | EXP-001 | EXP-002 |
|---|---:|---:|
| Best-loss epoch | 11 | 11 |
| Validation accuracy at best-loss epoch | 92.62% | 92.43% |
| Best validation loss | 0.2362 | 0.2330 |
| Epochs completed | 17 | 17 |

EXP-002 was interrupted after eight epochs and continued from the latest model
and Adam optimizer state. The random augmentation and shuffle state restarted,
so the continuation is not identical to uninterrupted training. Six epochs
after epoch 11 failed to improve validation loss, triggering early stopping.
The highest validation accuracy was 93.42% at epoch 17, but the best-loss
checkpoint is from epoch 11.

Both GRU size and augmentation changed between the two experiments, so the
effect of either change cannot be isolated. The held-out test split was not
used. The continuation's `curves.png` plots only resumed epochs and labels its
horizontal axis starting at 1; its `history.csv` records the original
zero-based epoch numbers 8 through 16.

The best EXP-002 checkpoint is `cnn_gru_EXP-002_continuation-01_best.keras`.
The `EXP-002/provenance.json` file was reconstructed from the
`original_provenance` field in the exported continuation metadata.

For a continuous view, use [the combined EXP-002 history](EXP-002/history_combined.csv)
and [combined curves](EXP-002/curves_combined.png). The CSV retains Keras's
zero-based epoch numbering and identifies the source segment for each row;
the chart uses one-based epoch numbers and marks the restart after epoch 8.

## EXP-003 results

EXP-003 used 64 GRU units with the same updated training augmentation as
EXP-002. Both the CNN and GRU were trainable. The initial attempt was
interrupted before completing an epoch; the fresh retry completed nine epochs,
then continued on Colab through epoch 16 and locally through epoch 20. Each
continuation loaded the latest checkpoint and Adam optimizer state and preserved
the previous best validation loss. Random augmentation and shuffle state
restarted, so this run is not identical to uninterrupted training.

| Result | EXP-001 | EXP-002 | EXP-003 |
|---|---:|---:|---:|
| GRU units | 64 | 128 | 64 |
| Best-loss epoch | 11 | 11 | 14 |
| Validation accuracy at best-loss epoch | 92.62% | 92.43% | 92.72% |
| Best validation loss | 0.2362 | 0.2330 | 0.2430 |
| Epochs completed | 17 | 17 | 20 |

No epoch after 14 improved on its validation loss. The local continuation
reached the 20-epoch limit after six consecutive epochs without improvement.
Epoch 20 had 93.02% validation accuracy, but its validation loss was higher;
the epoch 14 checkpoint remains the selected best-loss model.

EXP-002 and EXP-003 used the same recorded augmentation and training settings
apart from GRU size, but their training resumed at different points and random
states were not preserved. The local part of EXP-003 used an RTX 4060 Laptop
GPU, TensorFlow 2.21.0 and Keras 3.15.1. These results are preliminary; compare
per-class errors, model size and inference speed before choosing a checkpoint.
The held-out test split was not used.

The best EXP-003 checkpoint is
`cnn_gru_EXP-003-retry-01-continuation-01_best.keras`; it remains outside Git.
The local continuation's `curves.png` shows only epochs 17–20, with its
horizontal axis restarting at 1. Its history retains zero-based epoch numbers.
The [combined EXP-003 history](EXP-003/history_combined.csv) and
[combined curves](EXP-003/curves_combined.png) show all 20 completed epochs.
The chart marks restarts after epochs 9 and 16; each CSV row records its source.
The interrupted initial attempt completed no epochs and is not included.

## EXP-004 results

EXP-004 reduced the input from 16 frames to 8 while retaining 64 GRU units,
the updated augmentation, and the recorded training and validation settings
of EXP-003. The CNN and GRU were both trained. The model was trained on a
local RTX 4060 Laptop GPU using TensorFlow 2.21.0 and Keras 3.15.1.

| Result | EXP-003 | EXP-004 |
|---|---:|---:|
| Frames per clip | 16 | 8 |
| GRU units | 64 | 64 |
| Best-loss epoch | 14 | 11 |
| Validation accuracy at best-loss epoch | 92.72% | 90.65% |
| Best validation loss | 0.2430 | 0.2937 |
| Epochs completed | 20 | 17 |

EXP-004 stopped after six epochs without a new low validation loss. Its
training accuracy kept increasing while validation loss rose after epoch 11,
indicating overfitting in the later epochs. The 8-frame model was about
2.07 percentage points lower in validation accuracy at the selected
checkpoint. Training took 163.2 minutes for 17 epochs on the local GPU.
EXP-003 included Colab and local continuations, so its total training time is
not directly comparable. Input augmentation is random and not fully seeded;
the training sample order and random augmentations may also differ.

The held-out test split was not used. Webcam integration would need to supply
8-frame sequences if EXP-004 were selected. The best checkpoint is
`cnn_gru_EXP-004.keras`, kept outside Git.

## Files and handoff

- `EXP-001/history.csv`: per-epoch training and validation metrics.
- `EXP-001/curves.png`: loss and accuracy curves.
- `EXP-001/run_info.json`: original Colab run metadata and settings; paths in
  this file describe the original Colab session and must be adapted elsewhere.
- `EXP-000-real-smoke/run_info.json`: metadata from the one-epoch real-data
  pipeline check; its metrics are not a meaningful performance result.
- `EXP-002/`: history and provenance for the first eight epochs.
- `EXP-002-continuation-01/`: resumed history, curves and continuation metadata.
- `EXP-003-retry-01/`: the first nine epochs and original provenance.
- `EXP-003-retry-01-continuation-01/`: Colab continuation, epochs 10–16.
- `EXP-003-retry-01-continuation-02-local/`: local continuation, epochs 17–20.
  `resume_local.py` preserves the script used for that continuation; its paths
  assume the documented local Docker runtime and pinned code snapshot.
- `EXP-004/`: 8-frame run settings, provenance, history, curves and run info.
- `combine_histories.py`: regenerates the combined histories and curves from
  the original CSVs without modifying the source records; requires Matplotlib.
- Best checkpoint: `cnn_gru_EXP-001.keras`, saved outside Git and shared
  separately. The original Colab run saved it under Google Drive's
  `GestureController/checkpoints/` folder.

The model and training code are in `src/gesture_recognition/models/cnn_gru.py`
and `src/gesture_recognition/training/train_cnn_gru.py`. Configuration is in
`configs/cnn_gru.yaml`. Set the data path and output directories for the local
environment before reproducing a run, and use a new experiment ID so existing
results and checkpoints are not overwritten.
