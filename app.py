import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from keras.models import load_model

TEST_MAE = 7.14

st.title('Age Prediction')

@st.cache_resource
def get_model():
    return load_model('utkface_age_model.keras', compile=False)
model = get_model()

file = st.camera_input('Take a photo (face in the center)')


if file is not None:
    img = Image.open(file).convert("RGB")
    small = ImageOps.fit(img, (48, 48))

    X = np.array(small, dtype="float32").reshape(1, 48, 48, 3)

    prediction = model.predict(X, verbose=0)[0][0]
    age = int(round(max(prediction, 0)))

    st.subheader(f"Predicted age: {age} years")
    st.caption(f"Model MAE: {TEST_MAE} years (average error on the test set)")
    st.image(small, caption="What the model sees (48 × 48)", width=200)