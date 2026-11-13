# Live webcam test: CNN baseline EXP-004

First live test of the CNN baseline through the webcam app (`feature/webcam-integration`).

- **Date:** November 11, 2026
- **Tester:** Troy (one person, one room, laptop webcam)
- **Model:** `checkpoints/cnn_baseline_EXP-004.keras`
- **App settings:** 1.5 s window, confidence threshold 0.80, cooldown 1.5 s, re-arm 0.5 s, model input not mirrored, dry run
- **Method:** one session per gesture, 5 tries each. Hand raised from below, gesture done once, hand dropped out of frame, about 3 s rest between tries.
- **Logs:** `data/webcam_tests/logs/` on the tester's laptop (not committed). Each session summarized with `python scripts/summarize_log.py <log>`; the raw output is below.

## Results

| Gesture (5 tries) | Correct command fired | Model's top prediction |
| --- | --- | --- |
| Thumb Up | 5 / 5 | Thumb Up, up to 100% |
| Stop Sign | 4 / 5 | Stop Sign, up to 100% |
| Swipe Right | 4 / 5 | Swiping Right, up to 95% |
| Swipe Left | 0 / 5 | Swiping Right, up to 96% (fired `next_slide` 5 times) |
| Two Fingers Left | 0 / 5 | Sliding Two Fingers Left, max 79% (below the 0.80 threshold) |
| Two Fingers Right | 0 / 5 | Sliding Two Fingers Left, max 79% |
| Talking with hands, 1 minute | 0 false triggers | No gesture throughout |

- **Model time:** median 65 to 67 ms per prediction, 95th percentile 68 to 71 ms (laptop CPU). The first prediction of each session is a warm-up (about 8.5 s) and is excluded.
- **Pattern:** the model reads hand shape but not direction live: every swipe came out as Swiping Right, every two-finger slide as Sliding Two Fingers Left. This matches the EXP-004 confusion matrix (direction errors are 67% of its mistakes) and is what the CNN-GRU is meant to improve.
- **Camera orientation checked:** the preview is mirrored for the user (`mirror_preview: true`), the model sees the normal image (`mirror_model_input: false`).
- **Window length:** a 3 s window fired almost nothing in an earlier session; 1.5 s worked, since live gestures are shorter than Jester clips.

## Limits

- One person, one room, one lighting setup, 5 tries per gesture. An early check, not the final real-world test from the proposal (which uses all members and 2 to 3 minutes of natural talking).
- Tries per session were counted by the tester; the log records predictions, not intended gestures.
- Earlier sessions on Nov 11 (before re-arm was added, or with mixed gestures in one session) are not included.

## Raw summary output

```
===== 2026-11-11_153341_exp004_swipe_right_x5.csv =====
Commands fired: 4  (5.2 per minute)
  next_slide: 4
Top prediction per gesture (count, best confidence):
  No gesture                      57   best 100%
  Swiping Right                    9   best 95%
Why commands did not fire:
  no gesture: 47
  disabled: 10
  low confidence: 4
  waiting for rest: 1
Model time: median 66 ms, average 67 ms, 95th percentile 69 ms

===== 2026-11-11_153436_exp004_swipe_left_x5.csv =====
Commands fired: 5  (7.2 per minute)
  next_slide: 5
Top prediction per gesture (count, best confidence):
  No gesture                      47   best 100%
  Swiping Right                    8   best 96%
Why commands did not fire:
  no gesture: 41
  disabled: 6
  waiting for rest: 2
  low confidence: 1
Model time: median 66 ms, average 68 ms, 95th percentile 68 ms

===== 2026-11-11_153526_exp004_two_fingers_right_x5.csv =====
Commands fired: 0  (0.0 per minute)
Top prediction per gesture (count, best confidence):
  No gesture                      45   best 100%
  Sliding Two Fingers Left        13   best 79%
  Swiping Right                    3   best 50%
Why commands did not fire:
  no gesture: 40
  low confidence: 16
  disabled: 5
Model time: median 65 ms, average 67 ms, 95th percentile 68 ms

===== 2026-11-11_153614_exp004_two_fingers_left_x5.csv =====
Commands fired: 0  (0.0 per minute)
Top prediction per gesture (count, best confidence):
  No gesture                      62   best 100%
  Sliding Two Fingers Left        14   best 79%
  Swiping Right                    2   best 58%
  Thumb Up                         1   best 70%
Why commands did not fire:
  no gesture: 59
  low confidence: 17
  disabled: 3
Model time: median 66 ms, average 67 ms, 95th percentile 69 ms

===== 2026-11-11_153712_exp004_thumb_up_x5.csv =====
Commands fired: 5  (6.4 per minute)
  start_presentation: 5
Top prediction per gesture (count, best confidence):
  No gesture                      49   best 100%
  Thumb Up                        17   best 100%
Why commands did not fire:
  no gesture: 41
  waiting for rest: 10
  disabled: 8
  low confidence: 2
Model time: median 65 ms, average 67 ms, 95th percentile 69 ms

===== 2026-11-11_153807_exp004_stop_sign_x5.csv =====
Commands fired: 4  (4.7 per minute)
  pause_or_end_presentation: 4
Top prediction per gesture (count, best confidence):
  No gesture                      61   best 100%
  Stop Sign                       15   best 100%
Why commands did not fire:
  no gesture: 45
  disabled: 16
  waiting for rest: 9
  low confidence: 2
Model time: median 67 ms, average 69 ms, 95th percentile 71 ms

===== 2026-11-11_153931_exp004_talking_1min.csv =====
Length: 1.0 min, 97 predictions
Commands fired: 0  (0.0 per minute)
Top prediction per gesture (count, best confidence):
  No gesture                      97   best 100%
Why commands did not fire:
  no gesture: 94
  disabled: 3
Model time: median 66 ms, average 67 ms, 95th percentile 70 ms
```

All sessions: `model_path: checkpoints/cnn_baseline_EXP-004.keras`, `window_seconds: 1.5`, `confidence_threshold: 0.8`, `cooldown_seconds: 1.5`, `rearm_seconds: 0.5`, `mirror_model_input: False`.
