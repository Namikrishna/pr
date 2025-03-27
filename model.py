import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate Dummy Dataset (100 Samples)
np.random.seed(42)
temperature = np.random.uniform(20, 40, 100)
rainfall = np.random.uniform(100, 300, 100)
fertilizer = np.random.uniform(50, 200, 100)
yield_value = 2.5 * temperature + 0.8 * rainfall + 1.2 * fertilizer + np.random.normal(0, 10, 100)

# Create DataFrame
df = pd.DataFrame({
    "temperature": temperature,
    "rainfall": rainfall,
    "fertilizer": fertilizer,
    "yield": yield_value
})

# Split Data
X = df[["temperature", "rainfall", "fertilizer"]]
y = df["yield"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model trained and saved successfully!")
