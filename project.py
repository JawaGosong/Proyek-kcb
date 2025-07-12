from newspaper import Article
import joblib

# Load model dan vectorizer
model = joblib.load('model/saved_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Ambil link berita
url = input("Masukkan link berita: ")

try:
    artikel = Article(url)
    artikel.download()
    artikel.parse()

    teks = artikel.title + " " + artikel.text  # Gabung judul + isi
    teks_vec = vectorizer.transform([teks])
    hasil = model.predict(teks_vec)[0]

    if hasil == "fake":
        print("❌ HOAKS")
    else:
        print("✅ VALID")
except:
    print("Gagal mengambil isi berita. Link mungkin tidak valid.")
