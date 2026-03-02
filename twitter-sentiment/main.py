import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['polarity', 'text']
print(df.head())

df = df[df.polarity != 2] #remove neutral tweets

df['polarity'] = df['polarity'].map({0: 0, 4: 1}) # 0 for negative, 1 for positive

print(df['polarity'].value_counts())

def clean_text(text):
    return text.lower()

df['clean_text'] = df['text'].apply(clean_text) #convert text to lowercase for consistency

print(df[['text', 'clean_text']].head())

x_train, x_test, y_train, y_test = train_test_split(
    df['clean_text'], 
    df['polarity'], 
    test_size=0.2, 
    random_state=42
)

print(f'Training set size: {len(x_train)}')
print(f'Test set size: {len(x_test)}')

#TF-IDF transforma textos em uma matriz numérica de “importância” das palavras: 
#aumenta o peso de termos que aparecem muito em um documento (TF) e reduz o peso 
#de termos comuns em muitos documentos (IDF). Aqui o TfidfVectorizer gera features 
#com unigramas e bigramas, limitadas a 5000, ajusta no treino (fit_transform) e aplica 
#no teste (transform), resultando em matrizes esparsas no formato (n_amostras, 5000).

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

print("TF-IDF shape (train):", x_train_tfidf.shape)
print("TF-IDF shape (test):", x_test_tfidf.shape)

bnb = BernoulliNB()
bnb.fit(x_train_tfidf, y_train)

bnb_preds = bnb.predict(x_test_tfidf)

print("Bernoulli Naive Bayes Accuracy:", accuracy_score(y_test, bnb_preds))
print("\nBernoulliNB Classification Report:\n", classification_report(y_test, bnb_preds))

#O BNB é um modelo probabilístico que assume que as features são independentes,
#adequado para dados binários (presença/ausência de palavras).

svm = LinearSVC(max_iter= 1000)
svm.fit(x_train_tfidf, y_train)

#O SVM é um modelo de margem máxima que busca um hiperplano que melhor separa as classes,
#eficaz em espaços de alta dimensão, como o TF-IDF.

svm_preds = svm.predict(x_test_tfidf)

print("SVM Accuracy:", accuracy_score(y_test, svm_preds))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_preds))

logreg = LogisticRegression(max_iter=100)
logreg.fit(x_train_tfidf, y_train)

logreg_pred = logreg.predict(x_test_tfidf)

print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, logreg_pred))

#A Regressão Logística é um modelo linear que estima a probabilidade de uma classe,
#adequado para classificação binária, interpretável e eficiente em grandes conjuntos de dados.
#Que combina com o modelo Positivo/Negativo do dataset.

sample_tweets = ["I love this!", "I hate that!", "It was okay, not great."]
sample_vec = vectorizer.transform(sample_tweets)

print("\nSample Predictions:")
print("BernoulliNB:", bnb.predict(sample_vec))
print("SVM:", svm.predict(sample_vec))
print("Logistic Regression:", logreg.predict(sample_vec))

acc_bnb = accuracy_score(y_test, bnb_preds)
acc_svm = accuracy_score(y_test, svm_preds)
acc_log = accuracy_score(y_test, logreg_pred)

models = ["BernoulliNB", "SVM", "LogReg"]
accs = [acc_bnb, acc_svm, acc_log]

plt.figure(figsize=(7, 4))
plt.bar(models, accs)
plt.ylim(0.70, 0.85)
plt.title("Accuracy por modelo (1.6M tweets)")
plt.ylabel("Accuracy")
for i, v in enumerate(accs):
    plt.text(i, v + 0.003, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, logreg_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neg", "Pos"])
disp.plot(values_format="d")
plt.title("Matriz de Confusão — Logistic Regression")
plt.tight_layout()
plt.show()

feature_names = vectorizer.get_feature_names_out()
coefs = logreg.coef_[0]

TOP_N = 20

top_pos_idx = np.argsort(coefs)[-TOP_N:][::-1]
top_neg_idx = np.argsort(coefs)[:TOP_N]

top_pos_terms = feature_names[top_pos_idx]
top_neg_terms = feature_names[top_neg_idx]

top_pos_vals = coefs[top_pos_idx]
top_neg_vals = coefs[top_neg_idx]

plt.figure(figsize=(9, 6))
plt.barh(top_pos_terms[::-1], top_pos_vals[::-1])
plt.title(f"Top {TOP_N} termos/bigramas associados a POSITIVO (LogReg)")
plt.xlabel("Coeficiente (quanto maior, mais positivo)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
plt.barh(top_neg_terms[::-1], top_neg_vals[::-1])
plt.title(f"Top {TOP_N} termos/bigramas associados a NEGATIVO (LogReg)")
plt.xlabel("Coeficiente (quanto menor, mais negativo)")
plt.tight_layout()
plt.show()
