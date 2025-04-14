import os
import pandas as pd

# Temel yol
base_dir = r"C:\Users\musta\PycharmProjects\faultDetect"
splits = ["TrainData", "TestData", "VerifyData"]

# Sonuçları bu sözlükte tutacağız
dataframes = {}

# Bellek tasarrufu için satır başına okuma adedi
CHUNKSIZE = 10000

for split in splits:
    folder_path = os.path.join(base_dir, split)
    data_list = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            label = os.path.splitext(file_name)[0]

            # CSV'yi parçalar halinde oku
            for chunk in pd.read_csv(file_path, dtype='float32', chunksize=CHUNKSIZE):
                chunk["label"] = label
                data_list.append(chunk)

    # Split adıyla eşleşen DataFrame'i oluştur
    dataframes[split] = pd.concat(data_list, ignore_index=True)

# Artık elimizde 3 ayrı DataFrame var:
train_df = dataframes["TrainData"]
test_df = dataframes["TestData"]
verify_df = dataframes["VerifyData"]

# İlk 5 satırı incelemek istersen:
print("Train örneği:")
print(train_df.head())

print("\nTest örneği:")
print(test_df.head())

print("\nVerify örneği:")
print(verify_df.head())
