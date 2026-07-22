import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

=========================
JUDUL APLIKASI
=========================

st.title("Aplikasi Segmentasi Pelanggan Toko Buku")
st.write("""
Aplikasi ini digunakan untuk mengelompokkan pelanggan toko buku berdasarkan
karakteristik pembelian menggunakan algoritma K-Means dan Hierarchical Clustering.
""")

=========================
MEMBACA DATASET
=========================

df = pd.read_csv("Data Penjualan Toko Buku.csv")

st.subheader("Dataset Penjualan Toko Buku")
st.dataframe(df.head())

=========================
PREPROCESSING DATA
=========================

st.subheader("Preprocessing Data")

Menghapus missing value dan duplikat

df = df.dropna()
df = df.drop_duplicates()

Mengubah tanggal pembelian menjadi format datetime

df["tanggal pembelian"] = pd.to_datetime(df["tanggal pembelian"])

Membuat data customer

customer = df.groupby("nama_customer").agg(
Frekuensi_Transaksi=("id_transaksi", "count"),
Jumlah_Pembelian=("jumlah", "sum"),
Total_Belanja=("total", "sum")
).reset_index()

st.write("Data pelanggan setelah agregasi:")
st.dataframe(customer.head())

=========================
NORMALISASI DATA
=========================

X = customer[["Frekuensi_Transaksi", "Jumlah_Pembelian", "Total_Belanja"]]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

=========================
K-MEANS CLUSTERING
=========================

st.subheader("Hasil K-Means Clustering")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

=========================
HIERARCHICAL CLUSTERING
=========================

st.subheader("Hasil Hierarchical Clustering")

hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
customer["Cluster_Hierarchical"] = hc.fit_predict(X_scaled)

=========================
VISUALISASI DENGAN PCA
=========================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

Visualisasi K-Means

fig1, ax1 = plt.subplots(figsize=(6, 4))
scatter1 = ax1.scatter(
X_pca[:, 0],
X_pca[:, 1],
c=customer["Cluster_KMeans"]
)
ax1.set_title("Visualisasi K-Means")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
st.pyplot(fig1)

Visualisasi Hierarchical

fig2, ax2 = plt.subplots(figsize=(6, 4))
scatter2 = ax2.scatter(
X_pca[:, 0],
X_pca[:, 1],
c=customer["Cluster_Hierarchical"]
)
ax2.set_title("Visualisasi Hierarchical Clustering")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
st.pyplot(fig2)

=========================
DENDROGRAM
=========================

st.subheader("Dendrogram Hierarchical Clustering")

linked = linkage(X_scaled, method="ward")

fig3, ax3 = plt.subplots(figsize=(10, 5))
dendrogram(linked, truncate_mode="lastp", p=20, ax=ax3)
ax3.set_title("Dendrogram")
st.pyplot(fig3)

=========================
EVALUASI MODEL
=========================

st.subheader("Evaluasi Model")

silhouette_kmeans = silhouette_score(X_scaled, customer["Cluster_KMeans"])
silhouette_hc = silhouette_score(X_scaled, customer["Cluster_Hierarchical"])

dbi_kmeans = davies_bouldin_score(X_scaled, customer["Cluster_KMeans"])
dbi_hc = davies_bouldin_score(X_scaled, customer["Cluster_Hierarchical"])

evaluasi = pd.DataFrame({
"Algoritma": ["K-Means", "Hierarchical Clustering"],
"Silhouette Score": [silhouette_kmeans, silhouette_hc],
"Davies-Bouldin Index": [dbi_kmeans, dbi_hc]
})

st.dataframe(evaluasi)

=========================
MENENTUKAN GOLONGAN PELANGGAN
=========================
Karakteristik cluster K-Means

karakteristik_kmeans = customer.groupby("Cluster_KMeans")[[
"Frekuensi_Transaksi",
"Jumlah_Pembelian",
"Total_Belanja"
]].mean()

cluster_tertinggi = karakteristik_kmeans["Total_Belanja"].idxmax()
cluster_terendah = karakteristik_kmeans["Total_Belanja"].idxmin()

customer["Golongan_KMeans"] = customer["Cluster_KMeans"].map(
lambda x: "Pelanggan Prioritas" if x == cluster_tertinggi
else "Pelanggan Berisiko" if x == cluster_terendah
else "Pelanggan Potensial"
)

Karakteristik cluster Hierarchical

karakteristik_hc = customer.groupby("Cluster_Hierarchical")[[
"Frekuensi_Transaksi",
"Jumlah_Pembelian",
"Total_Belanja"
]].mean()

cluster_hc_tertinggi = karakteristik_hc["Total_Belanja"].idxmax()
cluster_hc_terendah = karakteristik_hc["Total_Belanja"].idxmin()

customer["Golongan_Hierarchical"] = customer["Cluster_Hierarchical"].map(
lambda x: "Pelanggan Prioritas" if x == cluster_hc_tertinggi
else "Pelanggan Berisiko" if x == cluster_hc_terendah
else "Pelanggan Potensial"
)

=========================
HASIL AKHIR SEGMENTASI
=========================

st.subheader("Golongan Pelanggan Berdasarkan Clustering")

st.dataframe(customer[[
"nama_customer",
"Frekuensi_Transaksi",
"Jumlah_Pembelian",
"Total_Belanja",
"Golongan_KMeans",
"Golongan_Hierarchical"
]])

=========================
KARAKTERISTIK TIAP GOLONGAN
=========================

st.subheader("Karakteristik Tiap Golongan (K-Means)")

karakteristik_golongan = customer.groupby("Golongan_KMeans")[[
"Frekuensi_Transaksi",
"Jumlah_Pembelian",
"Total_Belanja"
]].mean().round(2)

st.dataframe(karakteristik_golongan)

=========================
INTERPRETASI
=========================

st.subheader("Interpretasi Segmentasi Pelanggan")

st.write("""

Pelanggan Prioritas: pelanggan dengan frekuensi transaksi, jumlah pembelian, dan total belanja yang tinggi.
Pelanggan Potensial: pelanggan dengan karakteristik pembelian sedang dan berpotensi ditingkatkan loyalitasnya.
Pelanggan Berisiko: pelanggan dengan frekuensi transaksi dan total belanja rendah, sehingga perlu strategi promosi khusus.
""")

st.success("Analisis segmentasi pelanggan berhasil ditampilkan dengan metode K-Means dan Hierarchical Clustering.")
