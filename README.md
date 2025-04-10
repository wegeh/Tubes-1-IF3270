# Tugas Besar 1 IF3270 Pembelajaran Mesin - Feedforward Neural Network

## Deskripsi
Repository ini berisi implementasi model Feedforward Neural Network (FFNN) untuk Tugas Besar 1 IF3270 Pembelajaran Mesin. Proyek ini mencakup eksperimen dan analisis berbagai teknik, seperti regularisasi (L1 dan L2) serta RMS Norm, untuk meningkatkan performa model. Selain kode sumber Python, repository ini juga menyediakan file Jupyter Notebook (.ipynb) untuk eksplorasi interaktif serta folder `doc` yang berisi laporan lengkap.

## Cara Setup dan Menjalankan Program
1. **Clone Repository:**
   ```bash
   git clone https://github.com/wegeh/Tubes-1-IF3270.git
   cd Tubes-1-IF3270
   ```

2. **Setup Environment (virtual environment pada Python):**
   ```bash
    python -m venv env
    source env/bin/activate   # Untuk Linux/MacOS
    env\Scripts\activate   # Untuk Windows
   ```

3. **Install Dependencies: Pastikan Anda telah menginstal pip. Kemudian, jalankan:**
   ```bash
    pip install -r requirements.txt
   ```

4. **Menjalankan Program: Untuk menjalankan program, masuk ke folder src dan jalankan file Python yang diinginkan:**
   ```bash
    python src/<nama_file_python.py>
   ```
    Anda juga dapat membuka dan menjalankan file Jupyter Notebook (.ipynb) yang tersedia untuk eksplorasi lebih lanjut.

## Struktur Repository
- **folder src**: Berisi kode sumber Python untuk implementasi FFNN dan eksperimen.
- **folder doc**: Berisi dokumen laporan lengkap.
- **requirements.txt**: Daftar dependencies yang diperlukan.
- **README.md**: Penjelasan mengenai tugas dan dokumentasi.

## Pembagian Tugas (Kelompok 47)
- **Filbert (13522021)**
  - **Kontribusi**: Laporan, L1 dan L2 Regularization, bonus activation function, eksperiment.
- **Benardo (13522055)**
  - **Kontribusi**: Laporan, RMS Norm, bonus initialization method, eksperiment.
- **William Glory Henderson (13522113)**
  - **Kontribusi**: Laporan, FFNN class (all method), activation and loss function, initialization method.