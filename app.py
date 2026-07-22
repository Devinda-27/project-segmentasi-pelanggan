import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

# Judul aplikasi

st.title("Segmentasi Pelanggan Toko Buku")

# Membaca dataset

df = pd.read_csv("Data Penjualan Toko Buku.csv")

# Menampilkan nama kolom

st.subheader("Nama Kolom Dataset")
st.write(df.columns.tolist())

# Menampilkan data awal

st.subheader("Data Awal")
st.dataframe(df.head())

# Pilih kolom yang akan digunakan

st.subheader("Pilih Kolom untuk Clustering")

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

x_col = st.selectbox("Pilih Kolom X", numeric_cols)
y_col = st.selectbox("Pilih Kolom Y", numeric_cols)

# Menentukan jumlah cluster

k = st.slider("Jumlah Cluster", 2, 5, 3)

# Proses clustering

X = df[[x_col, y_col]]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# K-Means

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

# Hierarchical Clustering

hc = AgglomerativeClustering(n_clusters=k)
df["Cluster_Hierarchical"] = hc.fit_predict(X_scaled)

# Visualisasi K-Means

st.subheader("Visualisasi K-Means")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster_KMeans"])
ax.set_title("Hasil K-Means Clustering")
st.pyplot(fig)

# Hasil clustering

st.subheader("Hasil Clustering")
st.dataframe(df)

# Karakteristik cluster

st.subheader("Karakteristik Cluster K-Means")

karakteristik = df.groupby("Cluster_KMeans")[[x_col, y_col]].mean().round(2)

st.dataframe(karakteristik)

# Menentukan golongan pelanggan

cluster_tertinggi = karakteristik[y_col].idxmax()
cluster_terendah = karakteristik[y_col].idxmin()

def tentukan_golongan(cluster):
if cluster == cluster_tertinggi:
return "Pelanggan Prioritas"
elif cluster == cluster_terendah:
return "Pelanggan Berisiko"
else:
return "Pelanggan Potensial"

df["Golongan_Pelanggan"] = df["Cluster_KMeans"].apply(tentukan_golongan)

# Menampilkan golongan pelanggan

st.subheader("Golongan Pelanggan")
st.dataframe(df[[x_col, y_col, "Cluster_KMeans", "Golongan_Pelanggan"]])

st.success("Clustering berhasil dijalankan! Pilih kolom numerik yang sesuai untuk melihat hasil segmentasi pelanggan.")
