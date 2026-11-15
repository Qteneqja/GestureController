# Live webcam test: window length (EXP-004, 3.0 s vs 1.5 s)

Same-day comparison of the app's time window, to check whether the window explains the live results.

- **Date:** November 15, 2026
- **Tester:** Troy (one person, one room, laptop webcam), same seat and lighting for all sessions
- **Model:** `checkpoints/cnn_baseline_EXP-004.keras` (unchanged)
- **Only change:** `window_seconds`, set with `python scripts/run_demo.py --window 3.0` or `--window 1.5`
- **Other settings:** 16 frames, confidence threshold 0.80, cooldown 1.5 s, re-arm 0.5 s, model input not mirrored, dry run
- **Method:** one session per gesture, 5 tries each. Hand raised into the frame, gesture done once, hand dropped out of frame, about 3 s rest between tries. Directions follow the tester's own left and right.
- **Evidence:** each session's JSON log records `window_seconds`; the app overlay shows model, window and session name; 12 of the 14 sessions were screen recorded (not the talking sessions).

## Why we ran it

Training samples 16 frames across the whole Jester clip (about 3 s on average, `preprocessing.py`). The app had been using 1.5 s, based on an early, unlogged test with the older EXP-003 model where 3 s fired nothing. This retest checks that choice with EXP-004 and logged sessions.

## Results

| Gesture (5 tries) | 3.0 s correct | 3.0 s wrong commands | 1.5 s correct | 1.5 s wrong commands |
| --- | --- | --- | --- | --- |
| Swipe Right | 5 / 5 | 0 | 5 / 5 | 0 |
| Swipe Left | 0 / 5 | 5 (`next_slide`) | 0 / 5 | 5 (`next_slide`) |
| Two Fingers Right | 0 / 5 | 0 | 0 / 5 | 1 (`next_slide`) |
| Two Fingers Left | 0 / 5 | 0 | 0 / 5 | 2 (`next_slide`) |
| Thumb Up | 5 / 5 | 0 | 5 / 5 | 0 |
| Stop Sign | 5 / 5 | 1 (`start_presentation`) | 5 / 5 | 0 |
| Talking with hands, about 1 min | 0 false triggers | | 0 false triggers | |
| **Total** | **15 / 30** | **6** | **15 / 30** | **8** |

- **Model time:** median 68 to 74 ms (3.0 s) and 68 to 71 ms (1.5 s) per prediction, laptop CPU. First prediction of each session is a warm-up (about 9 s) and is excluded.
- **Swipe Left:** read as Swiping Right at both windows (best 95% at 3.0 s, 97% at 1.5 s).
- **Two-finger slides:** at 3.0 s the model flipped between Swiping Right, Two Fingers Left and Two Fingers Right, all below 0.80, so nothing fired. At 1.5 s it was more confident and sometimes passed 0.80 as Swiping Right.

## What it means

- **The window is not the cause of the live errors.** Correct commands were the same at both windows (15 / 30). The same gestures worked and the same gestures failed.
- **The earlier claim that "3 s fired almost nothing" does not hold for EXP-004.** That came from the EXP-003 test on Nov 5, where the model also differed.
- **Direction is the real problem.** Swipe Left was read as Swiping Right every time, at both windows. This matches the EXP-004 confusion matrix (67% of mistakes are left vs right) and is what the CNN-GRU is meant to fix.
- **Small difference in wrong commands** (6 at 3.0 s, 8 at 1.5 s). With 5 tries per gesture this could be chance, so we report it as no clear difference.

## Limits

- One person, one room, one lighting setup, 5 tries per gesture.
- Tries were counted by the tester; the logs record predictions, not intended gestures.
- Discarded sessions (kept in `logs/discarded/`): two 3.0 s Swipe Right sessions where commands were never turned on (all predictions `disabled`), and the first 3.0 s two-finger sessions, which were redone.
- The direction convention (tester's own left and right) has not yet been checked against Jester sample clips.

## Raw summaries

Run from the project folder:

```
Get-ChildItem data\webcam_tests\logs\*w30_*.csv, data\webcam_tests\logs\*w15_*.csv | ForEach-Object { python scripts/summarize_log.py $_.FullName }
```
