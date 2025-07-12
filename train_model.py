import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import nltk

# Download stopwords (sekali saja, bisa di-skip jika sudah)
nltk.download('stopwords')

# 1. Baca data dari file CSV
df1 = pd.read_csv("dataset/news.csv")
df2 = pd.read_csv("dataset/news_100.csv")

df = pd.concat([df1, df2], ignore_index=True)

# (opsional) hapus duplikat jika ada
df = df.drop_duplicates()

# 2. Pisahkan data input (teks berita) dan label (real/fake)
x = df['text']
y = df['label']

# 3. Bagi data menjadi data latih dan data uji (80:20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 4. Ubah teks menjadi angka menggunakan TF-IDF
vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# 5. Buat dan latih model AI
model = MultinomialNB()
model.fit(x_train_vec, y_train)

# 6. Evaluasi model dengan data uji
y_pred = model.predict(x_test_vec)

print("\n=== Evaluasi Model Deteksi Berita Hoaks ===")
print("Akurasi (tingkat ketepatan prediksi):", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=["Hoaks", "Valid"], zero_division=0))

print("\nMatriks Kebingungan (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))

# 7. Simpan model dan vectorizer agar bisa dipakai di file lain (GUI/project)
joblib.dump(model, 'model/saved_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

print("\nModel dan vectorizer berhasil disimpan ✅")
