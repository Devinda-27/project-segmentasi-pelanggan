import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistem Segmentasi Pelanggan", layout="wide")

# ======================
# Header
# ======================
st.title("📊 Sistem Segmentasi Pelanggan")
st.markdown("""
Sistem ini digunakan untuk **menganalisis dataset transaksi pelanggan** dan **mengelompokkan pelanggan**
berdasarkan perilaku pembelian menggunakan metode **K-Means Clustering**.

### Cara menggunakan sistem:
1. Upload file dataset dalam format **CSV**.
2. Pastikan dataset memiliki kolom: **nama_customer, id_transaksi, jumlah, total**.
3. Sistem akan membersihkan data secara otomatis.
4. Sistem akan mengelompokkan pelanggan menjadi beberapa cluster.
5. Hasil segmentasi dapat diunduh dalam format CSV.
""")

# ======================
# Upload File
# ======================
uploaded_file = st.file_uploader("📁 Upload Dataset CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)

    st.success("Dataset berhasil diupload!")

    # ======================
    # Informasi Dataset
    # ======================
    st.subheader("📌 Informasi Dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Data Duplikat", df.duplicated().sum())

    st.write("**Data kosong pada setiap kolom:**")
    st.dataframe(df.isnull().sum().reset_index().rename(columns={0: "Jumlah Kosong"}))

    st.write("**Preview Dataset:**")
    st.dataframe(df.head())

    # ======================
    # Pembersihan Data
    # ======================
    st.subheader("🧹 Pembersihan Data")

    before_rows = df.shape[0]

    # Hapus data kosong dan duplikat
    df = df.dropna()
    df = df.drop_duplicates()

    after_rows = df.shape[0]

    st.write(f"Data sebelum dibersihkan: **{before_rows} baris**")
    st.write(f"Data setelah dibersihkan: **{after_rows} baris**")

    # ======================
    # Membuat Data Pelanggan
    # ======================
    st.subheader("👥 Data Pelanggan")

    customer = df.groupby("nama_customer").agg(
        Frekuensi_Transaksi=("id_transaksi", "count"),
        Jumlah_Pembelian=("jumlah", "sum"),
        Total_Belanja=("total", "sum")
    ).reset_index()

    st.dataframe(customer.head())

    # ======================
    # Normalisasi
    # ======================
    X = customer[["Frekuensi_Transaksi", "Jumlah_Pembelian", "Total_Belanja"]]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ======================
    # Clustering K-Means
    # ======================
    st.subheader("🤖 Proses Clustering K-Means")

    n_clusters = st.slider("Pilih jumlah cluster", min_value=2, max_value=5, value=3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer["Cluster"] = kmeans.fit_predict(X_scaled)

    # Evaluasi
    silhouette = silhouette_score(X_scaled, customer["Cluster"])

    st.metric("Silhouette Score", f"{silhouette:.4f}")

    # ======================
    # Hasil Clustering
    # ======================
    st.subheader("📋 Hasil Segmentasi Pelanggan")
    st.dataframe(customer)

    # ======================
    # Visualisasi Cluster
    # ======================
    st.subheader("📈 Visualisasi Cluster")

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        customer["Frekuensi_Transaksi"],
        customer["Total_Belanja"],
        c=customer["Cluster"]
    )

    ax.set_xlabel("Frekuensi Transaksi")
    ax.set_ylabel("Total Belanja")
    ax.set_title("Visualisasi Cluster Pelanggan")
    plt.colorbar(scatter, ax=ax)

    st.pyplot(fig)

    # ======================
    # Interpretasi Cluster
    # ======================
    st.subheader("📝 Interpretasi Cluster")

    cluster_summary = customer.groupby("Cluster").agg(
        Jumlah_Pelanggan=("nama_customer", "count"),
        Rata_Frekuensi=("Frekuensi_Transaksi", "mean"),
        Rata_Total_Belanja=("Total_Belanja", "mean")
    ).reset_index()

    st.dataframe(cluster_summary)

    for i in range(n_clusters):
        cluster_data = cluster_summary[cluster_summary["Cluster"] == i]

        freq = cluster_data["Rata_Frekuensi"].values[0]
        belanja = cluster_data["Rata_Total_Belanja"].values[0]

        st.write(f"### Cluster {i}")

        if freq > customer["Frekuensi_Transaksi"].mean() and belanja > customer["Total_Belanja"].mean():
            st.success("Pelanggan Loyal: sering bertransaksi dan memiliki total belanja tinggi.")
        elif freq < customer["Frekuensi_Transaksi"].mean() and belanja < customer["Total_Belanja"].mean():
            st.warning("Pelanggan Pasif: jarang bertransaksi dan memiliki total belanja rendah.")
        else:
            st.info("Pelanggan Potensial: memiliki potensi untuk ditingkatkan menjadi pelanggan loyal.")

    # ======================
    # Download Hasil
    # ======================
    st.subheader("📥 Download Hasil")

    csv = customer.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Hasil Segmentasi CSV",
        data=csv,
        file_name="hasil_segmentasi_pelanggan.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload file CSV untuk memulai analisis.")
