import os
import pandas as pd
import glob

klasor_yolu = r"C:\Users\musta\PycharmProjects\faultDetect\dataset\2_CSV_Data_Files"
trainYuzde = 70
testYuzde = 20
verifyYuzde = 10
trainOut = r"C:\Users\musta\PycharmProjects\faultDetect\TrainData"
testOut = r"C:\Users\musta\PycharmProjects\faultDetect\TestData"
verifyOut = r"C:\Users\musta\PycharmProjects\faultDetect\VerifyData"

csv_dosyalar = glob.glob(os.path.join(klasor_yolu, "*.csv"))


for dosya in csv_dosyalar:
    df = pd.read_csv(dosya, skiprows=1)

    # Her 10 satırın ortalamasını al
    df_avg = df.groupby(df.index // 420).mean(numeric_only=True).round(4)

    satir_sayisi = len(df_avg)
    train = int(satir_sayisi * trainYuzde / 100)
    test = train + int(satir_sayisi * testYuzde / 100)

    df_train = df_avg[:train]
    df_test = df_avg[train:test]
    df_verify = df_avg[test:]

    dosya_adi = os.path.basename(dosya)

    df_train.to_csv(os.path.join(trainOut, dosya_adi), index=False)
    df_test.to_csv(os.path.join(testOut, dosya_adi), index=False)
    df_verify.to_csv(os.path.join(verifyOut, dosya_adi), index=False)

print("İşlem tamamlandı.")


