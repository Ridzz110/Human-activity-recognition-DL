# Human Activity Recognition Using Deep Learning

This project explores human activity recognition (HAR) from smartphone sensor data using several neural network architectures. The workflow is built around the UCI Human Activity Recognition Using Smartphones Dataset, which contains sensor windows collected from 30 volunteers performing six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying.

## Project goal

The goal is to classify activities from smartphone accelerometer and gyroscope signals using deep learning models and compare their performance.

## Methodology

1. Data loading and labeling
   - Loaded the training and test sets from the dataset folder `human+activity+recognition+using+smartphones`.
   - Mapped activity IDs to their class names using `activity_labels.txt`.

2. Feature preparation
   - The dataset contains 561 engineered features derived from the raw inertial sensor streams.
   - The features were normalized using `StandardScaler` to improve model convergence.
   - A validation split (10%) was created from the training set to monitor performance during training.

3. Model comparison
   - Single-layer feedforward network
   - Multilayer perceptron (MLP) with hidden layers and dropout
   - PCA + MLP to assess dimensionality reduction
   - CNN applied to the raw 128-step, 9-channel sequence representation
   - LSTM/RNN model for temporal sequence learning

4. Evaluation
   - Models were evaluated on the held-out test set using accuracy, precision, recall, and F1-score.
   - The best-performing model was selected based on the highest F1-score.

## Results

The best model in this work was the Multilayer Feedforward Network, which achieved the highest overall performance on the test set.

| Model | Accuracy | Precision | Recall | F1-Score |
| --- | ---: | ---: | ---: | ---: |
| Single-Layer Feedforward | 0.94096 | 0.94471 | 0.94096 | 0.94109 |
| Multilayer Feedforward | 0.95114 | 0.95180 | 0.95114 | 0.95105 |
| MLP + PCA | 0.93994 | 0.94049 | 0.93994 | 0.93969 |
| CNN | 0.92331 | 0.92678 | 0.92331 | 0.92210 |
| LSTM/RNN | 0.90499 | 0.90820 | 0.90499 | 0.90395 |

Key takeaway:
- The MLP model performed best with an accuracy of approximately 95.1%.
- The CNN and LSTM approaches were competitive but underperformed the MLP on this feature-based HAR task.
- This suggests that the engineered sensor features already provide strong discriminative information for activity classification.

## Quick start

To run the notebook in this environment:

```powershell
cd "C:\Ridzz Personal\UNI\ANN\ANN Project"
.\env\Scripts\Activate.ps1
jupyter notebook
```

Then open `HAR-Project.ipynb` in the browser and run the cells.

## Files in this repository

- `HAR-Project.ipynb` — main analysis and model comparison notebook
- `human+activity+recognition+using+smartphones/` — UCI HAR dataset files
- `env/` — local Python environment used for the project

## References

- UCI Human Activity Recognition Using Smartphones Dataset
- Original dataset publication by Anguita et al. (2013)
