import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from collections import Counter
import os

# Vorbereitung
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# CSV einlesen
df = pd.read_csv("persona_bias_optimized.csv")

# Leere/fehlende Outputs filtern
df = df[df['output'].notna() & (df['output'] != "No output generated.")]

# Ordner für Plots
os.makedirs("plots", exist_ok=True)

### 1. Verteilung der Toxizität
plt.figure(figsize=(8, 5))
sns.histplot(df["toxicity"].dropna(), bins=30, kde=True)
plt.title("Verteilung der Toxizität")
plt.xlabel("Toxizität")
plt.ylabel("Anzahl")
plt.savefig("plots/toxicity_distribution.png")
plt.close()

### 2. Sentiment nach Topic
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="topic", hue="sentiment_label")
plt.title("Sentiment-Verteilung pro Thema")
plt.xticks(rotation=30)
plt.ylabel("Anzahl")
plt.savefig("plots/sentiment_per_topic.png")
plt.close()

### 3. Toxizität nach Religion (Boxplot)
plt.figure(figsize=(12, 6))
top_religions = df["religion"].value_counts().index[:6]
sns.boxplot(data=df[df["religion"].isin(top_religions)], x="religion", y="toxicity")
plt.title("Toxizität nach Religion")
plt.xticks(rotation=45)
plt.savefig("plots/toxicity_per_religion.png")
plt.close()

### 4. Toxizität nach Geschlecht (Violinplot)
plt.figure(figsize=(6, 5))
sns.violinplot(data=df[df["gender"].notna()], x="gender", y="toxicity")
plt.title("Toxizität nach Geschlecht")
plt.savefig("plots/toxicity_per_gender.png")
plt.close()

### 5. Häufigste negativen Wörter (aus NEGATIVE Sentiments)
negative_texts = df[df["sentiment_label"] == "NEGATIVE"]["output"].dropna().tolist()
words = [
    word.lower()
    for text in negative_texts
    for word in text.split()
    if word.lower() not in STOPWORDS and word.isalpha()
]
word_freq = Counter(words).most_common(30)

# WordCloud
wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(dict(word_freq))
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Häufige Wörter aus NEGATIVEN Antworten")
plt.savefig("plots/negative_words_wordcloud.png")
plt.close()

### Optional: Drucke häufige Worte als Text
print("\nTop 20 häufige Wörter in negativen Antworten:")
for word, freq in word_freq[:20]:
    print(f"{word}: {freq}")

print("\n✅ Alle Plots wurden gespeichert im Ordner `plot2/`.")
