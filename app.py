import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA

# Judul aplikasi
st.title("Aplikasi Segmentasi Pelanggan Toko Buku")
st.write("Analisis segmentasi pelanggan berdasarkan data penjualan toko buku menggunakan K-Means dan Hierarchical Clustering.")

# Membaca dataset
df = pd.read_csv("Data Penjualan Toko Buku.csv")

# Menampilkan dataset
st.subheader("Dataset Penjualan")
st.dataframe(df.head())

# Preprocessing
df = df.dropna()
df = df.drop_duplicates()
df["tanggal pembelian"] = pd.to_datetime(df["tanggal pembelian"])

# Membuat data customer
customer = df.groupby("nama_customer").agg(
    Frekuensi_Transaksi=("id_transaksi", "count"),
    Jumlah_Pembelian=("jumlah", "sum"),
    Total_Belanja=("total", "sum")
).reset_index()

# Menghapus outlier
Q1 = customer["Total_Belanja"].quantile(0.25)
Q3 = customer["Total_Belanja"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

customer = customer[
    (customer["Total_Belanja"] >= lower_bound) &
    (customer["Total_Belanja"] <= upper_bound)
]

# Normalisasi
X = customer[["Frekuensi_Transaksi", "Jumlah_Pembelian", "Total_Belanja"]]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
customer["Cluster_Hierarchical"] = hc.fit_predict(X_scaled)

# PCA untuk visualisasi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualisasi K-Means
st.subheader("Visualisasi Hasil Clustering K-Means")

fig1, ax1 = plt.subplots(figsize=(6, 4))
scatter1 = ax1.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=customer["Cluster_KMeans"]
)
ax1.set_title("Hasil Clustering K-Means")
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")

st.pyplot(fig1)

# Visualisasi Hierarchical
st.subheader("Visualisasi Hasil Hierarchical Clustering")

fig2, ax2 = plt.subplots(figsize=(6, 4))
scatter2 = ax2.scatter(
    X_pca[:, 0], X_pca[:, 1],
    c=customer["Cluster_Hierarchical"]
)
ax2.set_title("Hasil Hierarchical Clustering")
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")

st.pyplot(fig2)

# Dendrogram
st.subheader("Dendrogram Hierarchical Clustering")

linked = linkage(X_scaled, method="ward")

fig3, ax3 = plt.subplots(figsize=(10, 5))
dendrogram(linked, truncate_mode="lastp", p=20, ax=ax3)
ax3.set_title("Dendrogram Hierarchical Clustering")

st.pyplot(fig3)

# Evaluasi model
score_kmeans = silhouette_score(X_scaled, customer["Cluster_KMeans"])
score_hc = silhouette_score(X_scaled, customer["Cluster_Hierarchical"])

dbi_kmeans = davies_bouldin_score(X_scaled, customer["Cluster_KMeans"])
dbi_hc = davies_bouldin_score(X_scaled, customer["Cluster_Hierarchical"])

st.subheader("Hasil Evaluasi Model")

hasil = pd.DataFrame({
    "Algoritma": ["K-Means", "Hierarchical Clustering"],
    "Silhouette Score": [score_kmeans, score_hc],
    "Davies-Bouldin Index": [dbi_kmeans, dbi_hc]
})

st.dataframe(hasil)

# Hasil segmentasi
st.subheader("Hasil Segmentasi Pelanggan")
st.dataframe(customer)

# Rata-rata tiap cluster
st.subheader("Karakteristik Cluster K-Means")

karakteristik = customer.groupby("Cluster_KMeans")[[
    "Frekuensi_Transaksi",
    "Jumlah_Pembelian",
    "Total_Belanja"
]].mean().round(2)

st.dataframe(karakteristik)

# Interpretasi
st.subheader("Interpretasi Segmentasi")

st.write("""
- **Cluster dengan Total_Belanja tinggi** dapat dianggap sebagai pelanggan prioritas.
- **Cluster dengan Frekuensi_Transaksi tinggi** menunjukkan pelanggan yang sering bertransaksi.
- **Cluster dengan nilai rendah** dapat menjadi target promosi untuk meningkatkan loyalitas pelanggan.
""")

# Karakteristik cluster K-Means
karakteristik_kmeans = customer.groupby("Cluster_KMeans")[[
    "Frekuensi_Transaksi",
    "Jumlah_Pembelian",
    "Total_Belanja"
]].mean()

# Menentukan cluster dengan total belanja tertinggi dan terendah
cluster_tertinggi = karakteristik_kmeans["Total_Belanja"].idxmax()
cluster_terendah = karakteristik_kmeans["Total_Belanja"].idxmin()

# Memberi label golongan pelanggan untuk K-Means
customer["Golongan_KMeans"] = customer["Cluster_KMeans"].map(
    lambda x: "Pelanggan Prioritas" if x == cluster_tertinggi
    else "Pelanggan Berisiko" if x == cluster_terendah
    else "Pelanggan Potensial"
)

# Karakteristik cluster Hierarchical
karakteristik_hc = customer.groupby("Cluster_Hierarchical")[[
    "Frekuensi_Transaksi",
    "Jumlah_Pembelian",
    "Total_Belanja"
]].mean()

cluster_hc_tertinggi = karakteristik_hc["Total_Belanja"].idxmax()
cluster_hc_terendah = karakteristik_hc["Total_Belanja"].idxmin()

# Memberi label golongan pelanggan untuk Hierarchical
customer["Golongan_Hierarchical"] = customer["Cluster_Hierarchical"].map(
    lambda x: "Pelanggan Prioritas" if x == cluster_hc_tertinggi
    else "Pelanggan Berisiko" if x == cluster_hc_terendah
    else "Pelanggan Potensial"
)

# Menampilkan hasil akhir
st.subheader("Golongan Pelanggan Berdasarkan Clustering")
st.dataframe(customer[[
    "nama_customer",
    "Frekuensi_Transaksi",
    "Jumlah_Pembelian",
    "Total_Belanja",
    "Golongan_KMeans",
    "Golongan_Hierarchical"
]])
