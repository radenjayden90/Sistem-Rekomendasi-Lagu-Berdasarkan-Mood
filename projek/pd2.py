import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler

# Load data
df = pd.read_csv("songs_normalize.csv")

print(df.shape)
print(df.info())
print(df.describe())

# Null checking
print(df.isna().sum())        # jumlah missing
print((df == 0).sum())        # jumlah nilai 0

# Fix format: hilangkan spasi berlebih
for col in df.select_dtypes(include="object"):
    df[col] = df[col].str.strip()

# Standarkan genre
df["genre"] = df["genre"].str.lower()

# Ubah explicit ke 0/1
df["explicit"] = df["explicit"].astype(str).str.lower().map({"true": 1, "false": 0})

# Missing value handling
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else np.nan)

# Outlier detection (IQR)
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR

    outliers = df[(df[col] < low) | (df[col] > high)]
    print(col, "Jumlah outlier:", len(outliers))

# Outlier handling (clip)
for col in num_cols:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

# Konversi duration_ms ke numeric dan menit
df["duration_ms"] = pd.to_numeric(df["duration_ms"], errors="coerce")
df["duration_min"] = df["duration_ms"] / 60000

#Remove duplicates
df.drop_duplicates(inplace=True)

# Genre ke multi-label
df["genre_list"] = df["genre"].str.replace(" ", "", regex=False).str.split(",")

# Encode multi-genre
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df["genre_list"])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
df = pd.concat([df, genre_df], axis=1)



# Normalisasi
num_cols = df.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Simpan data bersih
df.to_csv("song_Preprocessing.csv", index=False)