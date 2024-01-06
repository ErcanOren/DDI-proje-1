from bs4 import BeautifulSoup
import requests
import csv
import pandas as pd
import re
from simplemma import texpipt_lemmatizer

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

url_template = "https://www.hepsiburada.com/samsung-galaxy-a34-128-gb-8-gb-ram-samsung-turkiye-garantili-p-HBCV000040I6N3-yorumlari?sayfa={page_number}"

yorumlar = []

header = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 OPR/104.0.0.0"
}

for page_number in range(1, 125):
    url = url_template.format(page_number=page_number)
    page = requests.get(url, headers=header)
    content = page.content
    html = BeautifulSoup(content, "html")

    comments = html.find_all("span", itemprop="description")

    for comment in comments:
        comment_text = comment.text.strip()
        yorumlar.append(comment_text)

csv_file_path = "yorum.csv"

with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    for i in range(len(comments)):
        writer.writerow(comments[i])

df = pd.read_csv("yorum.csv")
df.head(-5)

stop_words = None
with open("stopwords.txt", "r") as stop_file:
    stop_words = set(stop_file.read().splitlines())


def clean(text):
    text = text.replace("Â", "a")
    text = text.replace("â", "a")
    text = text.replace("î", "i")
    text = text.replace("Î", "ı")
    text = text.replace("İ", "i")
    text = text.replace("I", "ı")
    text = text.replace(u"\u00A0", " ")
    text = text.replace("|", " ")

    text = re.sub(r"@[A-Za-z0-9]+", " ", text)
    text = re.sub(r"(.)\1+", r"\1\1", text)
    text = re.sub(r"https?:\/\/\S+", " ", text)
    text = re.sub(r"http?:\/\/\S+", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"#(\w+)", " ", text)
    text = re.sub(r"^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^A-Za-zâîığüşöçİĞÜŞÖÇ]+", " ", text)
    text = re.sub(r"((https://[^\s]+))", " ", text)

    text = " ".join(text.lower().strip().split())
    text = text_lemmatizer(text, lang="tr")

    return " ".join([word for word in text if word not in stop_words])

    cleaned_data_list = []


csv_file_path = "yorum.csv"

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        cleaned_comment = clean(row[0]); if row and row[0].strip(): else_None

        if cleaned_comment is not None:
            cleaned_data_list.append([cleaned_comment])

cleaned_csv_file_path = "temizYorum.csv"

with open(cleaned_csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Yorumlar'])
    writer.writerows(cleaned_data_list)

# burası şu an çalışmıyor ama kod template i bu şekilde
# temizleme ve etiketleme
df["Temiz"] = df.apply(lambda row: clean(row["Yorumlar"]), axis=1)

# filmleri kategorilerine göre etiketleme
kategoriler = []

for i in yorumlar:
    if i not in kategoriler:
        kategoriler.append(i)


def etiketle(row):
    for key, i in enumerate(kategoriler):
        if row["Yorumlar"] == i:
            return int(key)


df.to_csv("arabaModel.csv", index=False)

df = pd.read_csv("temizYorum.csv")
df.head()

# model eğitimi
df = pd.read_csv("temizYorum.csv")

X = df["Temiz"].to_numpy()
y = df["Etiket"].to_numpy()

print(len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# vektörleştirme
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Decision Tree ile
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1)
clf.fit(X_train_tfidf, y_train)

# model başarısı
clf.score(X_test_tfidf, y_test)