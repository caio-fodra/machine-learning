import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

DATA_PATH = "text_classification_dataset_600.csv"
RANDOM_STATE = 42

# 2. Loading the Dataset
data = pd.read_csv(DATA_PATH)
X = data["text"].astype(str)
y = data["label"].astype(str)

# 3. Splitting the Data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 4. Text Preprocessing: Converting Text to Numeric Features
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 5. Training the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 6. Making Predictions
y_pred = model.predict(X_test_vectorized)

# 7. Evaluating the Model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=np.unique(y))

print(f"Accuracy: {accuracy * 100}%")

class_labels = np.unique(y)

plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# 8. Prediction on Unseen Data
unseen_inputs = [
    # sports (esportes)
    "The striker finished the counterattack and the coach celebrated the late goal in the stadium",
    "After halftime the captain organized the defense and the team won the playoff match on penalties",
    "The referee stopped play briefly and the goalkeeper saved a powerful shot to keep the lead",

    # technology (tecnologia)
    "We deployed a new microservice to the production cluster and reduced latency after fixing an API timeout",
    "The security team patched the router firmware and verified encryption settings in the zero trust setup",
    "Engineers debugged a memory leak in the container pipeline and monitored CPU load during peak traffic",

    # politics (política)
    "The senate debated the budget bill and the opposition leader demanded stronger oversight in committee",
    "During the press briefing the minister defended tax reform and promised transparency to constituents",
    "A close floor vote approved the amendment and the coalition negotiated the final legislation wording",

    # entertainment (entretenimento)
    "The director premiered the film at the festival and critics praised the cinematography and soundtrack",
    "The singer performed on the concert stage and the crowd applauded the set design and lighting cues",
    "After the season finale the producer discussed the plot twist and fans shared reactions online",

    # science (ciência)
    "Researchers measured a protein with a microscope and validated the hypothesis using a strict control group",
    "The team sequenced the genome and reported statistical significance after repeating calibration checks",
    "Astronomy scientists observed an exoplanet with a telescope and analyzed the dataset to quantify uncertainty",

    # business (negócios)
    "The retailer reviewed quarterly earnings and improved cash flow by tightening inventory turnover controls",
    "During the earnings call executives explained pricing strategy and how market share affected revenue growth",
    "The finance group audited the balance sheet and forecasted operating costs to protect unit economics",
]

print("\nPredictions on unseen examples:")
for i, text in enumerate(unseen_inputs, start=1):
    text_vectorized = vectorizer.transform([text])
    pred = model.predict(text_vectorized)[0]
    print(f"{i:02d}. {pred} -> {text}")
