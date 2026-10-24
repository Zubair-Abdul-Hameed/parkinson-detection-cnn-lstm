# Early Parkinson's Disease Detection Using CNN and LSTM

*A voice-based screening prototype that classifies speech recordings as healthy or Parkinson's using mel spectrograms, a CNN baseline, and a hybrid CNN-LSTM.*

---

## About the Project

Parkinson's disease (PD) is a progressive neurodegenerative disorder, and changes in speech (reduced loudness, monotone pitch, imprecise articulation) are among its earliest measurable signs. Diagnosis today still depends largely on subjective clinical assessment by a specialist, and specialists are scarce in many regions, including Nigeria. Voice is cheap and non-invasive to capture, which makes it an attractive basis for low-cost screening. This project explores how far deep learning can go on that idea. It is a research prototype and a screening aid, not a diagnostic tool.

The system takes a voice recording (`.wav`) and predicts whether it comes from a healthy control (HC) or a person with Parkinson's disease (PD), along with a confidence score. It trains and compares two models under an identical training protocol: a convolutional neural network (CNN) baseline and a hybrid CNN-LSTM that adds temporal modelling on top of the convolutional features. The better-balanced model, the CNN-LSTM, is wrapped in a lightweight Flask web app where a user uploads a recording and gets back a prediction.

Under the hood, the project combines two public datasets, NeuroVoz (Spanish) and IPVS (Italian), giving 170 subjects and 3,729 recordings. Every recording passes through the same preprocessing chain: silence trimming, resampling to 16 kHz mono, and peak normalisation. The data is split at the **subject level** (70/15/15, stratified by dataset × label, giving 119 / 26 / 25 subjects), so no speaker ever appears in more than one split. This is what keeps the reported numbers from being inflated by speaker leakage. Recordings are cut into overlapping 2-second windows (1-second hop), producing 30,492 windows. Each window becomes a 128 × 63 log-mel spectrogram, z-score normalised with a mean and standard deviation computed from **training windows only**. Because some subjects contribute up to 21× more windows than others, training uses a subject-balanced weighted sampler so that long recordings don't dominate learning. The CNN uses three conv blocks with global average pooling (~93k parameters). The CNN-LSTM uses three conv blocks that pool frequency aggressively but time only once, handing a 31-step sequence to a single-layer LSTM (hidden size 64, ~143k parameters). Both are trained with the same optimiser, schedule, batch size, loss, and seed. Predictions are evaluated at the **recording level** by averaging window probabilities and thresholding at 0.5. On the held-out test set (518 recordings from 25 unseen subjects), the CNN reached 73.0% accuracy with 97.3% recall but only 49.2% specificity (ROC-AUC 0.923). The CNN-LSTM reached 76.8% accuracy with 86.7% recall and 67.2% specificity (ROC-AUC 0.861). In short, the CNN is a high-sensitivity screener that raises many false alarms, while the CNN-LSTM is more balanced. Both performed near-perfectly on the Italian IPVS recordings (~98% accuracy) but noticeably worse on the Spanish NeuroVoz recordings (68–73%).

---

## Demo
https://github.com/user-attachments/assets/a48c7603-05ca-49b2-ae6f-22256945f5a8

<img width="1350" height="825" alt="metric_comparison" src="https://github.com/user-attachments/assets/8bcea222-6ab7-44c9-9df6-dfa086012388" />

The app is intentionally not hosted online. At roughly 70–77% recording-level accuracy, the model isn't reliable enough to be used on arbitrary audio by the public, so the demo above shows its behaviour instead.

---

## Known Issues and Limitations

- **No statistical significance testing.** Results come from a single subject-level train/val/test split rather than k-fold cross-validation, so the performance gap between the CNN and CNN-LSTM has not been shown to be statistically significant.
- **Checkpoint selection used validation loss, not F1.** Validation loss was noisy on this small validation set (26 subjects), and the CNN-LSTM's "best" checkpoint ended up being epoch 1. Validation F1 consistently peaked at later epochs, so selecting on F1 would likely have produced a stronger model. Both models were not retrained with this fix.
- **Limited and uneven data.** Training used only 119 subjects. The ~22,000 training windows are highly correlated, so the effective sample size is bounded by subject count, which likely explains why validation loss plateaued around 0.41–0.45 across different regularisation settings. Performance is also much weaker on NeuroVoz than on IPVS, and the small IPVS test set (86 recordings) means its near-perfect scores should be read cautiously.
- **No Nigerian or English-language speech.** The models were trained and tested only on Spanish and Italian recordings, so their behaviour on other languages and recording conditions is untested.
- **Uncalibrated confidence and fixed threshold.** The confidence score is the raw averaged model probability, not a calibrated probability of disease, and the 0.5 decision threshold was not tuned. The CNN in particular ranks recordings well (high AUC) but is biased toward predicting PD at this threshold.
- **Not clinically validated.** This system has not been tested with clinicians or patients and must not be used for diagnosis.
