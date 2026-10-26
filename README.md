# Dynamic Hand Gesture Recognition Using CNN and GRU

**COMP-3704: Neural Networks and Deep Learning**  
**Group Project | Fall 2026**  
**Framework: TensorFlow / Keras**

## 1. Project Overview

This project explores how deep learning can be used to recognize dynamic hand gestures and translate them into computer commands without requiring a keyboard or mouse.

We will develop a gesture recognition system using a **Convolutional Neural Network (CNN)** and a **Gated Recurrent Unit (GRU)**. The CNN will extract visual features from video frames, while the GRU will learn how hand movements develop over time.

The project will begin with a CNN-only baseline model. We will then develop and optimize a CNN-GRU model to determine whether incorporating temporal information improves gesture recognition accuracy and computational efficiency.

The final goal is to demonstrate a working application that uses a regular laptop webcam to recognize gestures and execute computer commands.

### Target Functionality

The system will support two categories of computer controls:

**Presentation Controls**
- Next slide
- Previous slide
- Start presentation
- Pause or end presentation

**Multitasking Controls**
- Switch to the next browser tab
- Switch to the previous browser tab
- Switch between open applications

The exact gesture-to-command mapping will be finalized after selecting the classes available in the dataset.

The system should also learn to ignore unrelated hand movements to minimize accidental commands.

---

## 2. Project Objectives

Our main objectives are to:

1. Develop a CNN baseline model for classifying hand gestures.
2. Develop a CNN-GRU model capable of learning spatial and temporal features from short video sequences.
3. Optimize the CNN-GRU architecture through controlled experiments.
4. Compare the CNN baseline and optimized CNN-GRU model using consistent evaluation metrics.
5. Integrate the selected model with a real-time webcam application.
6. Evaluate recognition accuracy, computational efficiency, and reliability under different conditions.

### Key Challenges

- **Similar gestures:** Full-hand swipes and two-finger slides can look alike.
- **Temporal recognition:** Movement direction cannot always be determined from a single frame.
- **False triggers:** Natural hand movements should not activate computer commands.
- **Computational efficiency:** Video sequences require more processing than individual images.
- **Generalization:** Recognition performance may vary with lighting, camera angles, backgrounds, and different users.

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow / Keras |
| CNN and GRU Implementation | Keras API |
| Computer Vision | OpenCV |
| Numerical Processing | NumPy |
| Dataset Pipeline | TensorFlow `tf.data` |
| Model Optimization | KerasTuner / Manual Experiments |
| Evaluation | scikit-learn |
| Visualization | Matplotlib |
| Computer Automation | PyAutoGUI |
| Version Control | Git / GitHub |
| Development Environment | VS Code / Jupyter Notebook |

### Why TensorFlow?

We chose TensorFlow and Keras because they provide straightforward tools for developing, training, evaluating, and saving neural network models.

Keras supports CNN and recurrent layers, allowing us to implement our architecture within a single framework. TensorFlow also provides tools for building data pipelines and monitoring training experiments.

---

## 4. Dataset

**Dataset:** 20BN-Jester V1

**Source:** https://www.qualcomm.com/developer/software/jester-dataset

The Jester dataset contains short videos of people performing different hand gestures.

We will select gesture classes relevant to presentation controls and multitasking, including:

- Full-hand swipes
- Two-finger slides
- Other suitable gesture classes
- Unrelated or non-command movements, where suitable data is available

### Data Preparation

Before training, videos will be processed through a common preprocessing pipeline.

1. Select relevant gesture classes.
2. Extract or sample frames from each video.
3. Resize frames to a consistent resolution.
4. Normalize pixel values.
5. Create training, validation, and test partitions.
6. Apply appropriate data augmentation.
7. Build reusable TensorFlow dataset loaders.

**Initial proposed input configuration:**

| Setting | Initial Value |
|---|---|
| Frame Resolution | 112 × 112 |
| Color Channels | 3 (RGB) |
| Frames per Video | 16 |
| Model Input | `(batch, frames, height, width, channels)` |
| Data Type | Float32 |
| Pixel Range | 0–1 |

These are starting configurations and may change during experimentation.

Both models must use consistent preprocessing and data partitions for a fair comparison. Videos from the same original recording must not be split across training and testing. Where reliable participant identifiers are available, we will also consider participant-level separation.

### Additional Webcam Data

We will record additional gesture videos using our own webcams to evaluate performance outside the original dataset.

These tests will include variations in:

- Lighting
- Backgrounds
- Camera positions
- Gesture speeds
- Different individuals

We will also test unintended gestures and natural hand movements.

**Important:** The original Jester dataset and large processed video files should not be committed to GitHub. Dataset access and local setup instructions will be documented separately.

---

## 5. Neural Network Architecture

### Model A: CNN Baseline

The CNN baseline will extract spatial features from video frames and classify gestures without explicitly modeling the order of motion.

Proposed architecture:

```text
Input Video
    |
Sampled Frames
    |
Convolutional Layers
    |
Pooling Layers
    |
Flatten / Global Average Pooling
    |
Dense Layers
    |
Frame Predictions
    |
Prediction Aggregation
    |
Video-Level Gesture Prediction
```

For example, predictions from sampled frames can be averaged to produce one video-level classification.

This model establishes a reference point for evaluating whether recurrent layers improve performance.

### Model B: CNN-GRU

The CNN-GRU will process a sequence of video frames.

Proposed architecture:

```text
Input Video
    |
Frame Sampling / Preprocessing
    |
TimeDistributed CNN
    |
Spatial Feature Extraction
    |
GRU Layer
    |
Dense Layer
    |
Softmax Classification
    |
Predicted Gesture
```

**CNN responsibilities:**
- Recognize hand shapes.
- Extract visual features.
- Identify spatial patterns in individual frames.

**GRU responsibilities:**
- Process frame features sequentially.
- Learn movement direction.
- Identify temporal patterns.
- Distinguish gestures that appear similar in individual frames.

A GRU was selected because it can learn temporal relationships while generally using fewer parameters than a comparable LSTM configuration.

### Model Optimization

We will experiment with:

- Learning rates
- Batch sizes
- CNN layers and filters
- GRU hidden units
- Sequence lengths
- Dropout rates
- Optimizers
- Early stopping

Not every combination will necessarily be tested. Experiments will be selected based on available computational resources and initial model results.

Our objective is to balance classification performance, inference time, and computational requirements.

---

## 6. Team Responsibilities

The project is divided into five primary development workstreams.

Each member is responsible for implementing, testing, and documenting their assigned component.

Team members are expected to collaborate during integration, troubleshooting, and evaluation.

### Member 1 — Dataset and Preprocessing

**Role:** Data Engineering Lead

**Responsibilities:**
- Obtain and organize the Jester dataset.
- Select gesture classes.
- Define the shared gesture labels.
- Implement frame extraction and preprocessing.
- Develop data augmentation.
- Create reproducible dataset splits.
- Build TensorFlow dataset loaders.
- Assist with collecting additional webcam test data.

**Primary Files:**

```text
src/gesture_recognition/data/
data/splits/
scripts/prepare_data.py
configs/data.yaml
```

**Expected Deliverables:**
- Working preprocessing pipeline
- Dataset loader compatible with both models
- Gesture label mapping
- Dataset statistics and documentation

### Member 2 — CNN Baseline Development

**Role:** Baseline Model Lead

**Responsibilities:**
- Design the CNN architecture.
- Implement model training.
- Establish frame-to-video prediction aggregation.
- Train the initial CNN baseline.
- Track accuracy and loss.
- Save model checkpoints.
- Investigate baseline model weaknesses.
- Document the model architecture and results.

**Primary Files:**

```text
src/gesture_recognition/models/cnn.py
src/gesture_recognition/training/train_cnn.py
configs/cnn.yaml
experiments/cnn_baseline/
```

**Expected Deliverables:**
- Trained CNN baseline
- Reproducible training script
- Model configuration
- Baseline performance results

### Member 3 — CNN-GRU Development

**Role:** Neural Network Architecture Lead

**Responsibilities:**
- Implement the CNN feature extractor.
- Implement the GRU component.
- Combine CNN and GRU into a single model.
- Train the initial architecture.
- Experiment with architectural improvements.
- Investigate sequence lengths and recurrent units.
- Document model structure and design decisions.
- Support integration of the final model.

**Primary Files:**

```text
src/gesture_recognition/models/cnn_gru.py
src/gesture_recognition/training/train_cnn_gru.py
configs/cnn_gru.yaml
experiments/cnn_gru_baseline/
```

**Expected Deliverables:**
- Trained CNN-GRU model
- Reproducible training script
- Architecture diagram
- Initial model results
- Documented improvements

### Member 4 — Model Optimization and Evaluation

**Role:** Optimization and Evaluation Lead

**Responsibilities:**
- Develop the evaluation pipeline.
- Define consistent metrics for both models.
- Conduct controlled hyperparameter experiments.
- Compare the CNN and CNN-GRU models.
- Generate confusion matrices and performance graphs.
- Analyze classification errors.
- Track model training and inference efficiency.
- Identify the best model configuration.

**Primary Files:**

```text
src/gesture_recognition/evaluation/
scripts/run_evaluation.py
experiments/cnn_gru_optimized/
reports/figures/
```

**Expected Deliverables:**
- Experiment records
- Hyperparameter comparison tables
- Accuracy and loss curves
- Confusion matrices
- Model performance comparison
- Selected optimized model configuration

### Member 5 — Webcam Application and Integration

**Role:** Real-Time Application Lead

**Responsibilities:**
- Develop live webcam capture using OpenCV.
- Implement frame buffering and preprocessing.
- Load trained TensorFlow models.
- Generate predictions from webcam input.
- Connect gestures to computer commands.
- Implement confidence thresholds and cooldown periods.
- Add enable/disable controls.
- Test responsiveness and unintended triggers.
- Prepare the final working demonstration.

**Primary Files:**

```text
src/gesture_recognition/application/
scripts/run_demo.py
configs/inference.yaml
```

**Expected Deliverables:**
- Working webcam application
- Gesture-to-command mapping
- Safety controls to reduce accidental actions
- Performance test results
- Final project demonstration

### Shared Responsibilities

All five members will contribute to:

- GitHub code reviews
- Component integration
- Debugging
- Testing
- Mid-progress presentation
- Final report
- Final presentation
- Project demonstration

Each member should document their own development work and experimental findings.

---

## 7. Repository Structure

```text
gesture-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   ├── data.yaml
│   ├── cnn.yaml
│   ├── cnn_gru.yaml
│   └── inference.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── webcam_tests/
│
├── src/
│   └── gesture_recognition/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── preprocessing.py
│       │   ├── augmentation.py
│       │   ├── dataset_loader.py
│       │   └── gesture_labels.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── cnn.py
│       │   ├── cnn_gru.py
│       │   └── model_utils.py
│       │
│       ├── training/
│       │   ├── train_cnn.py
│       │   ├── train_cnn_gru.py
│       │   └── callbacks.py
│       │
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── evaluate.py
│       │   ├── compare_models.py
│       │   └── plots.py
│       │
│       └── application/
│           ├── webcam.py
│           ├── inference.py
│           ├── gesture_controller.py
│           └── main.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── run_training.py
│   ├── run_evaluation.py
│   └── run_demo.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_application.py
│
├── experiments/
│   ├── cnn_baseline/
│   ├── cnn_gru_baseline/
│   └── cnn_gru_optimized/
│
├── checkpoints/
│
├── reports/
│   ├── proposal/
│   ├── mid_progress/
│   ├── final/
│   └── figures/
│
└── docs/
    ├── architecture.md
    ├── dataset.md
    ├── experiments.md
    └── team_responsibilities.md
```

### Folder Organization Rules

- `configs/`: Shared experiment and application settings.
- `data/`: Local dataset files and dataset manifests.
- `src/`: Reusable project source code.
- `scripts/`: Entry points for preparing data, training, evaluation, and running the demo.
- `tests/`: Automated component tests.
- `experiments/`: Experiment configurations, results, and summaries.
- `checkpoints/`: Locally saved trained models.
- `reports/`: Assignment submissions and figures.
- `docs/`: Technical documentation.

Large datasets, model checkpoints, and generated experiment logs should generally remain outside normal Git tracking.

The repository should contain code and instructions needed to reproduce the experiments.

---

## 8. Development Timeline

### Phase 1 — Foundation and Initial Development

**October 26 – November 2**

- Member 1: Set up dataset preprocessing.
- Member 2: Develop CNN architecture and training pipeline.
- Member 3: Develop CNN-GRU architecture.
- Member 4: Implement evaluation utilities.
- Member 5: Create webcam capture and command interface.

**Milestone:** All major components have an initial implementation.

### Phase 2 — Initial Model Training

**November 3 – November 9**

- Finalize shared preprocessing and dataset splits.
- Train initial CNN baseline.
- Train initial CNN-GRU.
- Run preliminary evaluation.
- Begin testing webcam inference integration.

**Milestone:** Obtain the first comparable model results.

### Phase 3 — Mid-Progress Meeting

**November 10 – November 16**

Prepare to present:

- Work completed
- Initial experiments
- CNN baseline results
- Preliminary CNN-GRU results
- Accuracy/loss visualizations
- Challenges encountered
- Updated project timeline

**Official Deadline: November 16, 2026**

The instructor will review progress and provide feedback.

### Phase 4 — Optimization and Integration

**November 17 – November 30**

- Conduct additional hyperparameter experiments.
- Improve CNN-GRU performance.
- Evaluate computational efficiency.
- Integrate the selected model with the webcam application.
- Implement confidence thresholds and cooldowns.
- Test different gestures and environmental conditions.

**Milestone:** Integrated application with a trained gesture recognition model.

### Phase 5 — Final Testing and Documentation

**December 1 – December 7**

- Complete model optimization.
- Freeze selected model configuration.
- Perform final held-out evaluation.
- Test real-time responsiveness and false triggers.
- Collect final results and figures.
- Complete the integrated application.

**Internal Deadline: December 7, 2026**

The system should be functional by this date, leaving time for final reporting and presentation preparation.

### Phase 6 — Final Submission

**December 8 – December 14**

- Complete the formal report.
- Consolidate experimental results.
- Prepare presentation slides.
- Record or rehearse the demonstration.
- Review all code and documentation.
- Conduct final presentation rehearsal.

**Official Deadline: December 14, 2026**

---

## 9. Evaluation Strategy

Our primary research question is:

**Does adding a GRU to a CNN improve dynamic hand gesture recognition compared with a CNN-only baseline, while maintaining practical computational efficiency?**

We will evaluate both models using the same dataset partitions and compatible preprocessing procedures.

### Evaluation Metrics

| Metric | Purpose |
|---|---|
| Classification Accuracy | Overall percentage of correctly classified videos |
| Macro F1-Score | Measures classification quality across gesture classes |
| Validation Loss | Monitors learning progress and generalization |
| Confusion Matrix | Identifies which gestures are frequently confused |
| Inference Latency | Measures prediction processing time |
| Model Size | Measures storage requirements |
| False Trigger Rate | Measures unintended activations during webcam testing |

### Evaluation Process

1. Train the CNN baseline.
2. Train the initial CNN-GRU model.
3. Compare their performance using validation data.
4. Conduct optimization experiments on the CNN-GRU.
5. Select the best configuration using validation results.
6. Evaluate the selected model once on the held-out test set.
7. Compare final model performance against the CNN baseline.
8. Test the integrated application under realistic webcam conditions.

### Experiment Tracking

Each experiment should record:

- Experiment ID
- Model architecture
- Dataset and split version
- Hyperparameters
- Number of epochs
- Training and validation metrics
- Inference latency, where measured
- Observations and conclusions

Example:

| Experiment | Model | Learning Rate | GRU Units | Validation Accuracy |
|---|---|---|---|---|
| EXP-001 | CNN | 0.001 | N/A | Pending |
| EXP-002 | CNN-GRU | 0.001 | 64 | Pending |
| EXP-003 | CNN-GRU | 0.0005 | 128 | Pending |

These configurations are illustrative. Results will be filled in after training.

Experiments should be reproducible whenever possible by keeping random seeds, dataset partitions, and model configurations consistent.

---

## 10. Real-Time Application

The final application will process webcam input and convert recognized gestures into computer commands.

### Application Workflow

```text
Webcam Input
     |
Capture Video Frames
     |
Frame Buffer
     |
Preprocessing
     |
TensorFlow CNN-GRU Model
     |
Gesture Prediction
     |
Confidence Check
     |
Cooldown / Trigger Validation
     |
PyAutoGUI
     |
Computer Command
```

### Preventing False Triggers

The application should include:

**Confidence Threshold**

Only execute commands when the model's predicted confidence exceeds a configurable threshold.

**Cooldown Period**

Introduce a short delay after a command to prevent repeated activations from the same gesture.

**Gesture Confirmation**

Optionally require consistent predictions across multiple inference windows before activating a command.

**Activation Control**

Provide an option to enable or disable gesture recognition.

**Unrecognized Gestures**

When confidence is insufficient or a movement does not match a supported command, the system should take no action.

These controls reduce unintended activations but do not guarantee that every prediction will be correct. Their effectiveness will be tested during development.

---

## 11. GitHub Collaboration Workflow

All team members will work from the same GitHub repository.

### Branching Strategy

```text
main
│
├── feature/data-pipeline
├── feature/cnn-baseline
├── feature/cnn-gru
├── feature/optimization
└── feature/webcam-integration
```

Each member should work primarily in their assigned feature branch.

### Development Rules

1. Do not commit unfinished changes directly to `main`.
2. Create feature branches for new development work.
3. Commit regularly with descriptive messages.
4. Pull the latest changes from `main` before beginning new work.
5. Use pull requests to merge completed work.
6. Request at least one teammate's review before merging.
7. Avoid changing another member's component without communicating first.
8. Add or update relevant tests when implementing features.
9. Document configuration changes that affect other components.
10. Never commit private credentials or unnecessary large files.

### Example Git Workflow

Clone the repository:

```bash
git clone <repository-url>
cd gesture-recognition
```

Create your branch:

```bash
git switch -c feature/cnn-baseline
```

Make changes and commit:

```bash
git add .
git commit -m "Implement initial CNN architecture"
```

Push your branch:

```bash
git push -u origin feature/cnn-baseline
```

Then open a pull request on GitHub.

### Commit Message Examples

```text
feat: add Jester dataset loader
feat: implement CNN baseline
feat: add GRU temporal classifier
feat: implement webcam capture
test: add preprocessing unit tests
fix: correct input tensor dimensions
docs: update architecture documentation
```

---

## 12. Shared Integration Standards

To ensure everyone's code works together, we will agree on common interfaces before training begins.

### Input Format

The CNN-GRU will initially receive tensors shaped as:

```python
(batch_size, 16, 112, 112, 3)
```

The dimensions represent:

- Batch size
- Number of frames
- Image height
- Image width
- RGB channels

The CNN baseline will use an agreed strategy for producing video-level predictions from sampled frames.

### Gesture Labels

Gesture classes must be centrally defined in:

```text
src/gesture_recognition/data/gesture_labels.py
```

Both models and the webcam application must use the same class indices.

For example, if class index `0` represents a specific gesture, every component must interpret it consistently.

### Model Output

Each model should return one score per gesture class.

During inference, the system will convert these scores into class probabilities when needed.

The webcam application should be able to load either model using a consistent prediction interface.

### Model Saving

Trained TensorFlow/Keras models should use the supported Keras model format:

```text
model_name.keras
```

The exact checkpoint location and model version must be recorded in the experiment documentation.

### Integration Testing

Before integrating components into the main application, verify that:

- The dataset loader produces tensors of the correct shape.
- Both models accept the agreed input format.
- Predictions use the correct gesture labels.
- Saved models can be loaded successfully.
- Webcam preprocessing matches training preprocessing.
- Command mappings correspond to the correct gestures.
- Low-confidence predictions do not execute computer commands.

---

## 13. Getting Started

The following setup instructions describe the intended environment. Training scripts and dependencies will become available as development progresses.

### Prerequisites

- Python (team-approved compatible version)
- Git
- VS Code or another Python IDE
- A webcam for application testing
- Access to the selected Jester dataset classes

GPU acceleration is useful for training but is not strictly required for initial development.

### Step 1 — Clone Repository

```bash
git clone <repository-url>
cd gesture-recognition
```

### Step 2 — Create Virtual Environment

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 3 — Install Dependencies

Once `requirements.txt` has been finalized:

```bash
pip install -r requirements.txt
```

Expected core dependencies include:

```text
tensorflow
opencv-python
numpy
scikit-learn
matplotlib
pandas
PyYAML
pyautogui
pytest
```

KerasTuner may be added for optimization experiments.

Package versions will be pinned after the team confirms a compatible development environment.

### Step 4 — Prepare Dataset

After the preprocessing pipeline is implemented:

```bash
python scripts/prepare_data.py
```

### Step 5 — Train Models

Once the shared training entry points are implemented:

```bash
python scripts/run_training.py --model cnn
```

```bash
python scripts/run_training.py --model cnn_gru
```

### Step 6 — Evaluate Models

```bash
python scripts/run_evaluation.py
```

### Step 7 — Run Webcam Application

```bash
python scripts/run_demo.py
```

**Note:** These are planned command interfaces. They are not guaranteed to work until the corresponding scripts are implemented and integrated.

---

## 14. Testing Strategy

Each member should test their component independently before integration.

### Unit Tests

Tests should cover:

- Frame extraction and resizing
- Tensor shapes
- Dataset loading
- Model forward passes
- Evaluation metric calculations
- Gesture-to-command mapping
- Confidence and cooldown logic

### Integration Tests

Tests should confirm that:

- Data flows correctly into both models.
- Trained models can be loaded for inference.
- Webcam frames are processed consistently.
- Predictions are converted into the appropriate commands.
- Unsupported or uncertain predictions do not trigger commands.

### Real-World Testing

The system will be tested with:

- Different individuals
- Different backgrounds
- Different lighting conditions
- Different gesture speeds
- Similar-looking gestures
- Unrelated hand movements

Performance will be documented rather than assuming successful real-time operation.

---

## 15. Project Meetings and Communication

The team should meet at least once per week to review development progress.

### Weekly Check-In

Each member should provide:

1. What was completed since the previous meeting?
2. What is currently being developed?
3. Are there any blockers?
4. Does another member need to provide anything?
5. What will be completed before the next meeting?

### Collaboration Expectations

- Communicate blockers early.
- Keep your branch updated.
- Avoid duplicate development work.
- Share experiment results with the group.
- Document significant implementation decisions.
- Review teammates' pull requests.
- Test changes before merging.
- Respect agreed internal deadlines.

Work should be divided by primary responsibility, but resolving integration problems is a shared effort.

---

## 16. Assignment Deliverables

### Proposal — October 26, 2026

**Weight: 2%**

Required content:
- Introduction
- Problem Definition
- Proposed Plan
- Evaluation Metrics
- Proposed Timeline

### Mid-Progress Meeting — November 16, 2026

**Weight: 8%**

Required content:
- Overview of completed work
- Experiments conducted
- Challenges encountered
- Updated timeline
- Preliminary results and visualizations
- Discussion of instructor feedback

### Final Report and Presentation — December 14, 2026

**Weight: 24%**

The report should include:

- Introduction and problem statement
- Model descriptions
- Optimization process
- Experiments and results
- Evaluation metrics
- Challenges and solutions
- Conclusion and future work
- Appendix

The final presentation should explain the architecture, optimization choices, experimental findings, and overall conclusions.

Our team also intends to demonstrate the real-time webcam application.

The instructor's project instructions remain the authoritative reference for grading and deliverable requirements.

---

## 17. Project Completion Criteria

The project will be considered complete when:

- [ ] Selected dataset classes are prepared and documented.
- [ ] CNN baseline is trained and evaluated.
- [ ] CNN-GRU model is trained and evaluated.
- [ ] Optimization experiments are completed and documented.
- [ ] Both models are compared using consistent metrics.
- [ ] Final test results and figures are generated.
- [ ] Webcam application can load the selected model.
- [ ] Recognized gestures can execute their intended commands.
- [ ] Confidence and cooldown controls are implemented.
- [ ] Real-world testing is completed and documented.
- [ ] Final source code is merged into `main`.
- [ ] Setup and reproduction instructions are updated.
- [ ] Final report is completed.
- [ ] Final presentation is prepared.
- [ ] Project materials are submitted by December 14, 2026.

Improving upon the CNN baseline is a research objective, not a guaranteed result. A well-documented experiment that demonstrates limitations or trade-offs is still a valid project outcome.

---

## 18. Final Notes for the Team

Our project's main academic objective is not simply to build a computer application that responds to hand gestures.

**The priority is to develop, optimize, and evaluate a deep learning model using a systematic experimental approach.**

The webcam application is an important demonstration of how the model can be used in practice, but the model comparison and optimization results are central to the assignment.

To complete the project successfully, we need:

1. Reproducible data preparation and model training.
2. A working CNN baseline.
3. A working CNN-GRU model.
4. Meaningful optimization experiments.
5. A fair evaluation of model performance.
6. A functional demonstration where feasible.
7. Clear technical documentation and communication.

**Target internal completion date: December 7, 2026**  
**Official final submission date: December 14, 2026**

---