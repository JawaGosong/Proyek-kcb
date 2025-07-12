import tkinter as tk
from tkinter import messagebox, scrolledtext
import joblib

# Load model dan vectorizer
model = joblib.load('model/saved_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

# Prediksi teks
def cek_berita():
    semua_input = entry.get("1.0", "end-1c").strip()

    if not semua_input:
        messagebox.showwarning("Input Kosong", "Masukkan minimal satu berita.")
        return

    baris_teks = semua_input.splitlines()

    output.config(state='normal')

    for teks in baris_teks:
        teks = teks.strip()
        if len(teks.split()) < 2:
            output.insert(tk.END, f"\n⚠️ Teks terlalu pendek: '{teks}'\n", "ai")
            continue

        # Tampilkan input user
        output.insert(tk.END, f"\n🧑 Kamu:\n{teks}\n", "user")

        # Prediksi
        teks_vec = vectorizer.transform([teks])
        hasil = model.predict(teks_vec)[0]

        if hasil == "fake":
            respon = "❌ Ini berita terindikasi HOAKS"
        else:
            respon = "✅ Ini berita terindikasi VALID"

        # Tampilkan hasil AI
        output.insert(tk.END, f"\n🤖 AI:\n{respon}\n", "ai")

    output.config(state='disabled')
    output.yview(tk.END)
    entry.delete("1.0", "end")


# Reset
def clear_all():
    entry.delete("1.0", "end")
    output.config(state='normal')
    output.delete("1.0", "end")
    output.config(state='disabled')

# Setup Window
root = tk.Tk()
root.title("💬 Deteksi Berita Hoaks - GPT Style")
root.geometry("800x600")
root.configure(bg="#1e1e1e")

# Output Area
output = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 11), bg="#1e1e1e", fg="white", bd=0)
output.tag_config("user", foreground="#00bfff")
output.tag_config("ai", foreground="#90ee90")
output.config(state='disabled')
output.pack(padx=20, pady=(20, 10), fill=tk.BOTH, expand=True)

# Frame input bawah
bottom_frame = tk.Frame(root, bg="#1e1e1e")
bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=20, pady=10)

entry = tk.Text(bottom_frame, height=3, width=80, font=("Segoe UI", 11), bg="#2c2c2c", fg="white", insertbackground="white", relief="flat")
entry.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="we")

btn_submit = tk.Button(bottom_frame, text="Kirim", command=cek_berita, bg="#007bff", fg="white",
                       font=("Segoe UI", 10, "bold"), padx=20, pady=5, relief="flat")
btn_submit.grid(row=0, column=1)

btn_clear = tk.Button(bottom_frame, text="🧹 Clear", command=clear_all, bg="#6c757d", fg="white",
                      font=("Segoe UI", 9), relief="flat")
btn_clear.grid(row=1, column=1, pady=(5, 0))

# Biarkan input box nempel bawah
bottom_frame.grid_columnconfigure(0, weight=1)

root.mainloop()
