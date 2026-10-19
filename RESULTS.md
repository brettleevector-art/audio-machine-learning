# WILDLIGHT — Environmental Audio Intelligence

## 1. Project Overview

WILDLIGHT is an environmental audio classification research-engineering project investigating how deep learning models recognize environmental sounds.

The system takes a five-second audio recording, converts it into a mel spectrogram, and predicts one of 50 ESC-50 sound categories.

The project investigates three training approaches and evaluates model performance across the official ESC-50 folds.

### Research Questions

1. Can a CNN trained from scratch learn useful environmental sound patterns?
2. Does SpecAugment improve generalization to unseen audio clips?
3. Does pretrained ResNet18 perform better than the original CNN?
4. How consistent is performance across all five official validation folds?

## 2. Dataset and Preprocessing

| Property | Configuration |
|---|---|
| Dataset | ESC-50 |
| Total audio clips | 2,000 |
| Sound categories | 50 |
| Duration per clip | 5 seconds |
| Sample rate | 22,050 Hz |
| Samples per clip | 110,250 |
| Mel-frequency bins | 128 |
| Training folds | 1, 2, 3, 4 |
| Validation fold | 5 |
| Training clips | 1,600 |
| Validation clips | 400 |

### Preprocessing Pipeline

1. Load the audio and resample it to 22,050 Hz.
2. Fix the waveform length to 110,250 samples.
3. Generate a 128-bin mel spectrogram.
4. Convert the spectrogram to decibel values.
5. Apply the project's existing normalization: `(m + 40) / 40`.
6. Pass the resulting tensor to the model.

The official ESC-50 fold split is retained to reduce the risk of related recordings appearing in both training and validation data.

## 3. Models and Experiments

### Experiment 1 — Baseline CNN

The baseline CNN is trained from scratch using three convolutional blocks with 32, 64, and 128 feature channels, followed by global average pooling and a 50-class output layer.

The model is trained using AdamW, a learning rate of `0.001`, weight decay of `0.01`, batch size of 32, and 30 epochs.

Checkpoint: `esc50_cnn.pth`

### Experiment 2 — CNN with SpecAugment

The same CNN architecture is trained with frequency masking and time masking applied to the training spectrograms. Validation data remains unaugmented.

The experiment investigates whether spectrogram masking helps the model generalize to unseen audio examples.

Checkpoint: `esc50_cnn_specaugment.pth`

### Experiment 3 — Pretrained ResNet18

ResNet18 uses pretrained weights, a modified first convolution that accepts single-channel mel spectrograms, and a final classification layer with 50 outputs.

The model is fine-tuned on ESC-50 using the same training and validation fold arrangement.

Checkpoints:
- Final epoch: `esc50_resnet18_last.pth`
- Best observed fold-5 validation accuracy: `esc50_resnet18_best.pth`

## 4. Fold-5 Model Comparison

All four checkpoints were evaluated using the same 400 clips from official fold 5.

| Model | Checkpoint | Correct Predictions | Accuracy |
|---|---|---:|---:|
| Baseline CNN | `esc50_cnn.pth` | 186 / 400 | 46.50% |
| CNN + SpecAugment | `esc50_cnn_specaugment.pth` | 150 / 400 | 37.50% |
| ResNet18 — final checkpoint | `esc50_resnet18_last.pth` | 273 / 400 | 68.25% |
| ResNet18 — best checkpoint (diagnostic) | `esc50_resnet18_best.pth` | 307 / 400 | 76.75% |

Visualization: `model_comparison_fold5.png`

### Interpretation

The final ResNet18 checkpoint achieved 68.25% accuracy on fold 5, compared with 46.50% for the baseline CNN. This is an improvement of 21.75 percentage points on this validation fold.

The SpecAugment final checkpoint achieved 37.50%, which was lower than the baseline CNN's final-checkpoint accuracy in this run.

The best ResNet18 checkpoint achieved 76.75% on fold 5. This checkpoint was selected based on its performance on that same validation fold. Consequently, its measured fold-5 accuracy may be optimistically biased and should be treated as a diagnostic result rather than the primary comparison.

The final-checkpoint comparison remains the primary fold-5 comparison. These results alone do not establish which approach will perform most consistently across all five folds.

## 5. SpecAugment Training Observations

- Best observed fold-5 validation accuracy: 51.00%.
- Best observed epoch: epoch 23 in the zero-based training log (the 24th epoch).
- Final-epoch validation accuracy: 37.50%.
- Saved checkpoint: `esc50_cnn_specaugment.pth`.

The saved SpecAugment checkpoint contains the weights from the final epoch, not the epoch with the best observed validation accuracy. The difference highlights why both the training trajectory and the saved checkpoint need to be considered when interpreting results.

## 6. Current Project Status

### Completed

- ESC-50 dataset setup.
- Dataset class and mel-spectrogram preprocessing.
- Baseline CNN implementation, training, and evaluation.
- SpecAugment training experiment.
- Pretrained ResNet18 implementation and training.
- Fold-5 checkpoint comparison.
- Model comparison visualization.

### Remaining Work

1. Complete five-fold cross-validation using the official ESC-50 folds.
2. Record the validation accuracy for each fold.
3. Calculate the mean and standard deviation of the five final-fold accuracies.
4. Assess performance consistency across folds.
5. Implement prediction on external audio using the same preprocessing function used by the dataset.
6. Add the planned Gradio audio-upload interface.

## 7. Evaluation Protocol and Limitations

Fold-5 accuracy is a single-split measurement and may depend on the particular validation fold.

The best ResNet18 checkpoint was selected using fold-5 validation performance, so its fold-5 result is not an independent estimate of generalization.

Five-fold cross-validation remains necessary to investigate how consistently the ResNet18 approach performs across the predefined folds.

The project will retain the existing dataset, core preprocessing, model comparison sequence, and research questions. Further conclusions will be based on recorded experimental results rather than assumed improvements.

## 8. Reproducibility

The project uses Python, PyTorch, TorchVision, Torchaudio, Librosa, NumPy, Pandas, Matplotlib, and scikit-learn for data loading, modelling, training, and evaluation.

The repository contains the training and evaluation scripts. Model checkpoints may be regenerated by running the corresponding training scripts with the ESC-50 dataset available in the expected folder structure.

The next milestone is five-fold cross-validation, followed by prediction on external audio using the shared preprocessing pipeline.


## 9. Five-Fold ResNet18 Cross-Validation

| Validation Fold | Final Accuracy | Best Accuracy | Best Epoch |
|---|---:|---:|---:|
| Fold 1 | 71.50% | 77.25% | 28 |
| Fold 2 | 78.75% | 82.00% | 25 |
| Fold 3 | 73.00% | 77.50% | 10 |
| Fold 4 | 80.75% | 85.00% | 22 |
| Fold 5 | 74.75% | 74.75% | 30 |

- Mean final accuracy: 75.75%
- Standard deviation: 3.48 percentage points

Visualization: `resnet18_five_fold_cv.png`

The mean summarizes final validation accuracy across the five
official folds. The standard deviation describes variation
between folds.

These results provide an estimate of ResNet18 performance
across the ESC-50 folds. They do not establish performance
on every possible environmental recording.

## 10. Audio Inference

- Shared audio preprocessing is implemented in `dataset.py`.
- `predict.py` loads a trained ResNet18 checkpoint and returns
  the five highest-scoring ESC-50 categories.
- `app.py` provides a local Gradio audio-upload interface.
- Training and inference share the same sample rate, clip length,
  mel-spectrogram configuration, and normalization.

### Running the interface

Activate the project virtual environment and run:

    python app.py

Open the local URL printed by Gradio and upload an audio recording
to view the five highest-scoring sound categories.

Prediction scores are softmax outputs and should not be interpreted
as calibrated probabilities.
