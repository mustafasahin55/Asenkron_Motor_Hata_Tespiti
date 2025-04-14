import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ham verileri içeren klasör yolu
klasor_yolu = r"C:\Users\musta\PycharmProjects\faultDetect\dataset\2_CSV_Data_Files"

# Örnek bir CSV dosyasını seçelim
csv_dosyalar = glob.glob(os.path.join(klasor_yolu, "*.csv"))
ornek_dosya = csv_dosyalar[0]  # İlk dosyayı örnek olarak alalım

# Ham veriyi oku (skiprows=1 ile başlık satırını atlayarak)
ham_veri = pd.read_csv(ornek_dosya, skiprows=1)

print(f"Örnek dosya: {os.path.basename(ornek_dosya)}")
print(f"Ham veri boyutu: {ham_veri.shape}")
print("Ham veri sütunları:", ham_veri.columns.tolist())
print("İlk 5 satır:")
print(ham_veri.head())

# Sütun isimlerini daha anlaşılır hale getirelim (örnek olarak)
yeni_sutun_isimleri = {
    ham_veri.columns[0]: 'Acc1',
    ham_veri.columns[1]: 'Acc2',
    ham_veri.columns[2]: 'Acc3',
    ham_veri.columns[3]: 'Mikrofon',
    ham_veri.columns[4]: 'Sıcaklık'
}
ham_veri = ham_veri.rename(columns=yeni_sutun_isimleri)

# 1. Histogram - Tüm sensör verileri
plt.figure(figsize=(15, 10))
for i, col in enumerate(['Acc1', 'Acc2', 'Acc3', 'Mikrofon', 'Sıcaklık']):
    plt.subplot(2, 3, i+1)
    plt.hist(ham_veri[col], bins=30, alpha=0.7)
    plt.title(f'{col} Histogramı')
    plt.xlabel(f'{col} Değerleri')
    plt.ylabel('Frekans')
    plt.grid(True)
plt.tight_layout()
plt.savefig('ham_veri_histogramlar.png')
plt.show()

# 2. Zaman Serisi Grafiği - İlk 1000 veri noktası
plt.figure(figsize=(15, 10))
for i, col in enumerate(['Acc1', 'Acc2', 'Acc3', 'Mikrofon']):
    plt.subplot(2, 2, i+1)
    plt.plot(ham_veri[col].iloc[:1000], label=col)
    plt.title(f'{col} Zaman Serisi (İlk 1000 Veri Noktası)')
    plt.xlabel('Veri Noktası')
    plt.ylabel('Değer')
    plt.grid(True)
    plt.legend()
plt.tight_layout()
plt.savefig('ham_veri_zaman_serisi.png')
plt.show()

# 3. Scatter Plot - Acc1 vs Acc2, Acc1 vs Acc3
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.scatter(ham_veri['Acc1'], ham_veri['Acc2'], alpha=0.5, s=5)
plt.title('Acc1 vs Acc2 Scatter Plot')
plt.xlabel('Acc1 Değerleri')
plt.ylabel('Acc2 Değerleri')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(ham_veri['Acc1'], ham_veri['Acc3'], alpha=0.5, s=5)
plt.title('Acc1 vs Acc3 Scatter Plot')
plt.xlabel('Acc1 Değerleri')
plt.ylabel('Acc3 Değerleri')
plt.grid(True)
plt.tight_layout()
plt.savefig('ham_veri_scatter_plots.png')
plt.show()

# 4. Boxplot - Tüm sensör verileri
plt.figure(figsize=(12, 6))
sns.boxplot(data=ham_veri[['Acc1', 'Acc2', 'Acc3', 'Mikrofon', 'Sıcaklık']])
plt.title('Sensör Verileri Boxplot')
plt.xlabel('Sensörler')
plt.ylabel('Değerler')
plt.grid(True)
plt.savefig('ham_veri_boxplot.png')
plt.show()

# 5. Korelasyon Matrisi
plt.figure(figsize=(10, 8))
corr = ham_veri[['Acc1', 'Acc2', 'Acc3', 'Mikrofon', 'Sıcaklık']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Sensör Verileri Korelasyon Matrisi')
plt.savefig('ham_veri_korelasyon.png')
plt.show()

# 6. Yoğunluk Grafiği (KDE)
plt.figure(figsize=(15, 10))
for i, col in enumerate(['Acc1', 'Acc2', 'Acc3', 'Mikrofon']):
    plt.subplot(2, 2, i+1)
    sns.kdeplot(ham_veri[col], fill=True)
    plt.title(f'{col} Yoğunluk Grafiği')
    plt.xlabel(f'{col} Değerleri')
    plt.ylabel('Yoğunluk')
    plt.grid(True)
plt.tight_layout()
plt.savefig('ham_veri_yogunluk.png')
plt.show()

# 7. 3D Scatter Plot - Acc1, Acc2, Acc3
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Veri boyutu çok büyükse, örnekleme yapalım
sample_size = min(5000, len(ham_veri))
sample = ham_veri.sample(sample_size)
ax.scatter(sample['Acc1'], sample['Acc2'], sample['Acc3'], alpha=0.5, s=10)
ax.set_xlabel('Acc1')
ax.set_ylabel('Acc2')
ax.set_zlabel('Acc3')
ax.set_title('3D Scatter Plot - Acc1, Acc2, Acc3')
plt.savefig('ham_veri_3d_scatter.png')
plt.show()
