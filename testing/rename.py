import os

# Ganti dengan lokasi folder tempat file-file PDF acak itu berada
folder_target = r'capstone_data_scientist/data/raw/JavaDeveloper'

def rename_files_to_sequential(path):
    print(f"Mengecek isi folder: {path}")
    
    # Ambil semua nama file di dalam folder, filter yang .pdf aja biar aman
    # Kalau di OS lu tersembunyi, ini tetep bakal kebaca
    files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    
    # Opsional: Urutkan nama file secara alfabetis sebelum di-rename
    files.sort()
    
    print(f"Ditemukan {len(files)} file PDF. Mulai proses ganti nama...")
    
    # Pake enumerate buat ngeluarin angka urut (0, 1, 2, dst) secara otomatis
    for index, old_name in enumerate(files):
        # Bangun alamat lengkap (path) untuk file lama
        old_file_path = os.path.join(path, old_name)
        
        # Bangun alamat lengkap untuk nama file baru (0.pdf, 1.pdf, dst)
        new_name = f"{index}.pdf"
        new_file_path = os.path.join(path, new_name)
        
        try:
            # Eksekusi penggantian nama
            os.rename(old_file_path, new_file_path)
            # print(f"Sukses: {old_name} -> {new_name}") # Uncomment baris ini kalau mau lihat log per file
        except Exception as e:
            print(f"Gagal ganti nama {old_name}: {e}")

    print("Proses selesai bro! Coba cek folder lu sekarang.")

# Panggil fungsinya
rename_files_to_sequential(folder_target)