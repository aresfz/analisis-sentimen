import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi Flask
app = Flask(__name__)

# Memuat model dan tfidf_vectorizer yang sudah disimpan
model = joblib.load('model/new_model_sushiNB.pkl')
tfidf_vectorizer = joblib.load('model/new_sushi_tfidf.pkl')

# Membaca dataset
file_path = 'dataset/sentimen_sushi.csv'  # Sesuaikan dengan lokasi file Anda
data = pd.read_csv(file_path)

# Menghapus baris yang memiliki nilai NaN pada kolom 'processed_komentar'
data = data.dropna(subset=['processed_komentar'])

# Memisahkan fitur dan label
X = data['processed_komentar']
y = data['sentimen']

# Menghitung confusion matrix
y_pred = model.predict(tfidf_vectorizer.transform(X))
cm = confusion_matrix(y, y_pred)

# Menentukan urutan label untuk confusion matrix (termasuk label 'Netral')
labels = ['Negatif', 'Netral', 'Positif']  # Tentukan urutan label sesuai dengan keinginan

# Visualisasi confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')

# Menyimpan confusion matrix sebagai gambar
confusion_matrix_path = os.path.join('static', 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)
plt.close()

# Visualisasi Distribusi Sentimen
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='sentimen', palette='Set2')
sentiment_distribution_path = os.path.join('static', 'sentiment_distribution.png')
plt.savefig(sentiment_distribution_path)
plt.close()

# Visualisasi Kata dengan Bobot TF-IDF Tertinggi
X_tfidf = tfidf_vectorizer.transform(X)
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
tfidf_scores = X_tfidf.sum(axis=0).A1
top_n = 10
top_indices = np.argsort(tfidf_scores)[::-1][:top_n]
top_words = feature_names[top_indices]
top_scores = tfidf_scores[top_indices]

# Visualisasi kata dengan bobot TF-IDF tertinggi
plt.figure(figsize=(10, 6))
sns.barplot(x=top_scores, y=top_words, palette='viridis')
top_tfidf_words_path = os.path.join('static', 'top_tfidf_words.png')
plt.savefig(top_tfidf_words_path)
plt.close()

# Fungsi untuk melakukan prediksi sentimen
def predict_sentiment(comment):
    comment_tfidf = tfidf_vectorizer.transform([comment])
    prediction = model.predict(comment_tfidf)
    return prediction[0]

@app.route('/')
@app.route('/page/<int:page_num>')
def index(page_num=1):
    items_per_page = 10
    start_idx = (page_num - 1) * items_per_page
    end_idx = start_idx + items_per_page

    # Menangani filter sentimen
    sentimen_filter = request.args.get('sentimen_filter', 'all')

    if sentimen_filter != 'all':
        filtered_data = data[data['sentimen'] == sentimen_filter]
    else:
        filtered_data = data

    # Ambil data yang dipaginasi
    paginated_data = filtered_data.iloc[start_idx:end_idx]

    # Hitung akurasi model
    y_pred = model.predict(tfidf_vectorizer.transform(X))
    accuracy = accuracy_score(y, y_pred)

    total_items = len(filtered_data)
    total_pages = (total_items // items_per_page) + (1 if total_items % items_per_page > 0 else 0)

    # Tentukan halaman pertama dan terakhir yang akan ditampilkan dalam pagination
    page_range_start = max(1, page_num - 2)
    page_range_end = min(total_pages, page_num + 2)

    return render_template(
        'index.html',
        data=paginated_data.to_dict(orient='records'),
        page_num=page_num,
        total_pages=total_pages,
        page_range_start=page_range_start,
        page_range_end=page_range_end,
        accuracy=accuracy,
        sentiment_distribution_path='sentiment_distribution.png',
        top_tfidf_words_path='top_tfidf_words.png'
    )


if __name__ == '__main__':
    app.run(debug=True)
