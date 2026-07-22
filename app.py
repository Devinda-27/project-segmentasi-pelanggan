import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistem Segmentasi Pelanggan", layout="wide")

st.title("📊 Sistem Segmentasi Pelanggan")
st.write("Upload dataset CSV untuk melakukan segmentasi pelanggan menggunakan K-Means dan Hierarchical Clustering.")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Awal")
    st.dataframe(df.head())

    # Preprocessing
    df = df.dropna()
    df = df.drop_duplicates()

    # Konversi tanggal jika ada
    if "tanggal pembelian" in df.columns:
        df["tanggal pembelian"] = pd.to_datetime(df["tanggal pembelian"])

    # Membuat data customer
    customer = df.groupby("nama_customer").agg(
        Frekuensi_Transaksi=("id_transaksi", "count"),
        Jumlah_Pembelian=("jumlah", "sum"),
        Total_Belanja=("total", "sum")
    ).reset_index()

    # Penanganan outlier
    Q1 = customer["Total_Belanja"].quantile(0.25)
    Q3 = customer["Total_Belanja"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    customer = customer[
        (customer["Total_Belanja"] >= lower_bound) &
        (customer["Total_Belanja"] <= upper_bound)
    ]

    # Feature untuk clustering
    X = customer[["Frekuensi_Transaksi", "Jumlah_Pembelian", "Total_Belanja"]]

    # Normalisasi
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    customer["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)

    # Hierarchical Clustering
    hc = AgglomerativeClustering(n_clusters=3, linkage="ward")
    customer["Cluster_Hierarchical"] = hc.fit_predict(X_scaled)

    # Evaluasi
    score_kmeans = silhouette_score(X_scaled, customer["Cluster_KMeans"])
    score_hc = silhouette_score(X_scaled, customer["Cluster_Hierarchical"])

    dbi_kmeans = davies_bouldin_score(X_scaled, customer["Cluster_KMeans"])
    dbi_hc = davies_bouldin_score(X_scaled, customer["Cluster_Hierarchical"])

    # Menampilkan hasil evaluasi
    st.subheader("Hasil Evaluasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Silhouette Score K-Means", f"{score_kmeans:.4f}")
        st.metric("Davies-Bouldin Index K-Means", f"{dbi_kmeans:.4f}")

    with col2:
        st.metric("Silhouette Score Hierarchical", f"{score_hc:.4f}")
        st.metric("Davies-Bouldin Index Hierarchical", f"{dbi_hc:.4f}")

    # Menampilkan hasil clustering
    st.subheader("Hasil Segmentasi Pelanggan")
    st.dataframe(customer)

    # Visualisasi K-Means
    st.subheader("Visualisasi Cluster K-Means")

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        customer["Frekuensi_Transaksi"],
        customer["Total_Belanja"],
        c=customer["Cluster_KMeans"]
    )
    ax.set_xlabel("Frekuensi Transaksi")
    ax.set_ylabel("Total Belanja")
    ax.set_title("Cluster K-Means")
    plt.colorbar(scatter, ax=ax)

    st.pyplot(fig)

    # Download hasil
    csv = customer.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Hasil Segmentasi",
        data=csv,
        file_name="hasil_segmentasi.csv",
        mime="text/csv"
    )
