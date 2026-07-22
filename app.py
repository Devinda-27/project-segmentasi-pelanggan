import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Segmentasi Pelanggan Toko Buku", layout="wide")

# Header
st.title("📚 Sistem Segmentasi Pelanggan Toko Buku")
st.write(
    "Sistem ini menganalisis data transaksi pelanggan toko buku dan "
    "mengelompokkan pelanggan berdasarkan perilaku pembelian mereka."
)

# Upload file
uploaded_file = st.file_uploader("Upload Dataset Transaksi Toko Buku (CSV)", type=["csv"])

if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    # Membersihkan data
    df = df.dropna()
    df = df.drop_duplicates()

    st.success("Data berhasil dibersihkan dari nilai kosong dan duplikat.")

    # Membuat data pelanggan
    customer = df.groupby("nama_customer").agg(
        Frekuensi_Transaksi=("id_transaksi", "nunique"),
        Jumlah_Buku=("jumlah", "sum"),
        Total_Belanja=("total", "sum")
    ).reset_index()

    st.subheader("Data Pelanggan")
    st.dataframe(customer)

    # Pilih jumlah cluster
    n_clusters = st.slider(
        "Pilih jumlah cluster pelanggan",
        min_value=2,
        max_value=10,
        value=3
    )

    # Normalisasi data
    X = customer[["Frekuensi_Transaksi", "Jumlah_Buku", "Total_Belanja"]]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    customer["Cluster"] = kmeans.fit_predict(X_scaled)

    # Evaluasi
    silhouette = silhouette_score(X_scaled, customer["Cluster"])

    st.subheader("Hasil Evaluasi")
    st.metric("Silhouette Score", f"{silhouette:.4f}")

    # Hasil clustering
    st.subheader("Hasil Segmentasi Pelanggan")
    st.dataframe(customer)

    # Visualisasi
    st.subheader("Visualisasi Cluster Pelanggan")

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        customer["Frekuensi_Transaksi"],
        customer["Total_Belanja"],
        c=customer["Cluster"]
    )

    ax.set_xlabel("Frekuensi Transaksi")
    ax.set_ylabel("Total Belanja")
    ax.set_title("Cluster Pelanggan Toko Buku")
    plt.colorbar(scatter, ax=ax)

    st.pyplot(fig)

    # Interpretasi cluster
    st.subheader("Interpretasi Cluster")

    summary = customer.groupby("Cluster").agg(
        Jumlah_Pelanggan=("nama_customer", "count"),
        Rata_Frekuensi=("Frekuensi_Transaksi", "mean"),
        Rata_Jumlah_Buku=("Jumlah_Buku", "mean"),
        Rata_Total_Belanja=("Total_Belanja", "mean")
    ).reset_index()

    st.dataframe(summary)

    for i in range(n_clusters):
        st.write(f"### Cluster {i}")

        row = summary[summary["Cluster"] == i].iloc[0]

        if row["Rata_Total_Belanja"] > summary["Rata_Total_Belanja"].mean():
            st.success(
                "Pelanggan Loyal: sering membeli buku dan memiliki total belanja tinggi."
            )
        else:
            st.info(
                "Pelanggan Potensial/Pasif: perlu strategi promosi untuk meningkatkan pembelian."
            )

    # Download hasil
    csv = customer.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Hasil Segmentasi",
        data=csv,
        file_name="hasil_segmentasi_pelanggan_toko_buku.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload dataset transaksi toko buku dalam format CSV.")
