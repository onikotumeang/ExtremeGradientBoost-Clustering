import pandas as pd
file_path = r"C:\Users\ronny\Documents\knime_models\boxplot_analysis\metering_nec_20kw - cleaned.csv"
df_csv = pd.read_csv(file_path, encoding='latin1')
print(df_csv)

#Lihat 5 data pertama
df_csv.head()
print(df_csv.head())

#Lihat 5 data terakhir
df_csv.tail()
print(df_csv.tail())

#Info struktur data
print(df_csv.info())

print(df_csv.describe())
print('TRANSMITTER_FWD_POWER')
print(df_csv['TRANSMITTER_FWD_POWER'].describe())
print("Nama Kolom")

print(df_csv.columns)

print(df_csv.isnull().sum())

print(df_csv.duplicated().sum())

print(df_csv['TRANSMITTER_FWD_POWER'].unique())

print(df_csv.shape)