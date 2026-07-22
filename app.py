import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sistem Segmentasi Pelanggan", layout="wide")

st.title("📊 Sistem Segmentasi Pelanggan")
st.write("Upload dataset pelanggan dalam format CSV, lalu pilih kolom yang ingin digunakan untuk proses clustering.")

# Upload file
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    # Membersihkan data
    df = df.dropna()
    df = df.drop_duplicates()

    st.success("Data berhasil dibersihkan dari nilai kosong dan duplikat.")

    # Memilih kolom ID pelanggan
    customer_col = st.selectbox(
        "Pilih kolom ID/Nama Pelanggan",
        options=df.columns
    )

    # Memilih kolom numerik untuk clustering
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    selected_features = st.multiselect(
        "Pilih kolom numerik untuk clustering",
        options=numeric_columns,
        default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
    )

    if len(selected_features) >= 2:
        # Membuat data pelanggan berdasarkan kolom yang dipilih
        customer = df.groupby(customer_col)[selected_features].sum().reset_index()

        st.subheader("Data Pelanggan Setelah Agregasi")
        st.dataframe(customer.head())

        # Memilih jumlah cluster
        n_clusters = st.slider(
            "Pilih jumlah cluster (n_cluster)",
            min_value=2,
            max_value=10,
            value=3,
            step=1
        )

        # Normalisasi data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(customer[selected_features])

        # Proses K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        customer["Cluster"] = kmeans.fit_predict(X_scaled)

        # Evaluasi model
        silhouette = silhouette_score(X_scaled, customer["Cluster"])

        st.subheader("Hasil Evaluasi")
        st.metric("Silhouette Score", f"{silhouette:.4f}")

        # Menampilkan hasil clustering
        st.subheader("Hasil Segmentasi Pelanggan")
        st.dataframe(customer)

        # Visualisasi cluster menggunakan dua fitur pertama
        st.subheader("Visualisasi Cluster")

        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(
            customer[selected_features[0]],
            customer[selected_features[1]],
            c=customer["Cluster"]
        )

        ax.set_xlabel(selected_features[0])
        ax.set_ylabel(selected_features[1])
        ax.set_title("Visualisasi Cluster Pelanggan")
        plt.colorbar(scatter, ax=ax)

        st.pyplot(fig)

        # Ringkasan cluster
        st.subheader("Ringkasan Cluster")
        summary = customer.groupby("Cluster")[selected_features].mean()
        st.dataframe(summary)

        # Download hasil
        csv = customer.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Hasil Segmentasi",
            data=csv,
            file_name="hasil_segmentasi.csv",
            mime="text/csv"
        )

    else:
        st.warning("Pilih minimal 2 kolom numerik untuk melakukan clustering.")
