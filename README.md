# Age Prediction from a Face Photo

A small CNN that estimates a person's age from a 48×48 face image, with a Streamlit app that takes a webcam photo and shows the predicted age.

**🔗 Deployed Model:** <https://agepredictioncnn.streamlit.app>

## Project structure

```
├── Age_Predictor.ipynb        # data loading, training, evaluation
├── app.py                     # Streamlit app
├── utkface_age_model.keras    # Trained model
└── requirements.txt
```

## Data

- 23,708 face images from UTKFace, resized to 48×48 RGB. The age is read from the file name (range: 1 to 116).
- Split 80/20 into train and test, then the train part was split 80/20 again into train (~15,170) and validation (~3,790), all with `random_state=42`. The test set was used only for the final evaluation.
- Source: Kaggle <https://www.kaggle.com/datasets/jangedoo/utkface-new>

## Model architecture

```
Input (48×48×3) → Rescaling(1/255)
  → Conv2D(32, 3×3, ReLU) → MaxPooling(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPooling(2×2)
  → Flatten → Dense(128, ReLU) → Dense(1)
```

Choices:
- **Regression**
- **Adam, MAE loss**
- **Early stopping**

## Performance (test set)

| Model | MAE (years) |
|-------|-------------|
| Baseline (always predict the mean age) | 15.41 |
| **This CNN** | **7.14** |

On average the prediction is about 7 years off, less than half the baseline error.

## Limitations

Where it is likely to fail:
- **Error of about ±7 years.**
- **Low resolution.**
- **Dataset bias.** UTKFace does not represent all ages, ethnicities and image conditions equally, so accuracy will vary between groups.
- **Privacy.** Webcam images are processed in the app session only. If you deploy your own copy, do not store user images without consent.
