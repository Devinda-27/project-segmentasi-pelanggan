import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Dashboard Segmentasi Pelanggan Toko Buku", layout="wide")

# ======================
# Membaca dataset dari GitHub
# ======================
DATA_PATH = "Data Penjualan Toko Buku.csv"
df = pd.read_csv(DATA_PATH)

# ======================
# Header
# ======================
st.title("📚 Dashboard Segmentasi Pelanggan Toko Buku")
st.markdown("""
Dashboard ini digunakan untuk menganalisis perilaku pelanggan toko buku dan mengelompokkan pelanggan berdasarkan pola pembelian menggunakan **K-Means Clustering**.
""")

# ======================
# Pembersihan Data
# ======================
df = df.dropna()
df = df.drop_duplicates()

# Konversi tanggal
df["tanggal pembelian"] = pd.to_datetime(df["tanggal pembelian"])

# ======================
# Ringkasan Dataset
# ======================
st.subheader("📊 Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Transaksi", df["id_transaksi"].nunique())
col2.metric("Jumlah Pelanggan", df["nama_customer"].nunique())
col3.metric("Jumlah Buku Terjual", int(df["jumlah"].sum()))
col4.metric("Total Pendapatan", f"Rp {df['total'].sum():,.0f}")

# ======================
# Preview Dataset
# ======================
st.subheader("📋 Preview Dataset")
st.dataframe(df.head())

# ======================
# Statistik Deskriptif
# ======================
st.subheader("📈 Statistik Deskriptif")
st.dataframe(df[["jumlah", "total"]].describe())

# ======================
# Membuat Data Pelanggan
# ======================
customer = df.groupby("nama_customer").agg(
    Frekuensi_Transaksi=("id_transaksi", "nunique"),
    Jumlah_Buku=("jumlah", "sum"),
    Total_Belanja=("total", "sum")
).reset_index()

# ======================
# Analisis Perilaku Pelanggan
# ======================
st.subheader("👥 Analisis Perilaku Pelanggan")

col1, col2, col3 = st.columns(3)
col1.metric("Rata-rata Frekuensi", f"{customer['Frekuensi_Transaksi'].mean():.2f}")
col2.metric("Rata-rata Buku Dibeli", f"{customer['Jumlah_Buku'].mean():.2f}")
col3.metric("Rata-rata Total Belanja", f"Rp {customer['Total_Belanja'].mean():,.0f}")

# ======================
# Pengaturan Cluster
# ======================
st.subheader("⚙️ Pengaturan Clustering")

n_clusters = st.slider(
    "Pilih jumlah cluster pelanggan",
    min_value=2,
    max_value=10,
    value=3
)

# ======================
# Normalisasi dan K-Means
# ======================
X = customer[["Frekuensi_Transaksi", "Jumlah_Buku", "Total_Belanja"]]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
customer["Cluster"] = kmeans.fit_predict(X_scaled)

# ======================
# Evaluasi Model
# ======================
silhouette = silhouette_score(X_scaled, customer["Cluster"])
dbi = davies_bouldin_score(X_scaled, customer["Cluster"])

st.subheader("📏 Evaluasi Model")

col1, col2 = st.columns(2)
col1.metric("Silhouette Score", f"{silhouette:.4f}")
col2.metric("Davies-Bouldin Index", f"{dbi:.4f}")

# ======================
# Hasil Segmentasi
# ======================
st.subheader("📌 Hasil Segmentasi Pelanggan")
st.dataframe(customer)

# ======================
# Visualisasi Cluster
# ======================
# ======================
# Visualisasi Cluster Modern
# ======================
st.subheader("🎨 Visualisasi Cluster Pelanggan")

fig = px.scatter(
    customer,
    x="Frekuensi_Transaksi",
    y="Total_Belanja",
    color="Cluster",
    size="Jumlah_Buku",
    hover_name="nama_customer",
    hover_data={
        "Frekuensi_Transaksi": True,
        "Jumlah_Buku": True,
        "Total_Belanja": ":,.0f"
    },
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="Sebaran Cluster Pelanggan",
    width=620,
    height=400
)

# Mengatur tampilan agar lebih modern
fig.update_layout(
    title_x=0.5,
    template="plotly_white",
    margin=dict(l=10, r=10, t=50, b=10),
    legend_title_text="Cluster",
    font=dict(size=12)
)

# Membuat titik lebih menarik
fig.update_traces(
    marker=dict(
        opacity=0.85,
        line=dict(width=1, color="white")
    )
)

st.plotly_chart(fig, use_container_width=False)

# ======================
# Top 10 Pelanggan
# ======================
st.subheader("🏆 Top 10 Pelanggan dengan Total Belanja Tertinggi")

top_customers = customer.sort_values(by="Total_Belanja", ascending=False).head(10)
st.dataframe(top_customers)

# ======================
# Ringkasan Cluster
# ======================
st.subheader("📝 Ringkasan Cluster")

summary = customer.groupby("Cluster").agg(
    Jumlah_Pelanggan=("nama_customer", "count"),
    Rata_Frekuensi=("Frekuensi_Transaksi", "mean"),
    Rata_Jumlah_Buku=("Jumlah_Buku", "mean"),
    Rata_Total_Belanja=("Total_Belanja", "mean")
).reset_index()

st.dataframe(summary)

# ======================
# Interpretasi Cluster
# ======================
st.subheader("💡 Interpretasi Cluster Pelanggan")

# Nilai rata-rata keseluruhan sebagai pembanding
avg_freq = customer["Frekuensi_Transaksi"].mean()
avg_belanja = customer["Total_Belanja"].mean()

for i in range(n_clusters):
    row = summary[summary["Cluster"] == i].iloc[0]

    # Menentukan kategori pelanggan
    if row["Rata_Total_Belanja"] > avg_belanja and row["Rata_Frekuensi"] > avg_freq:
        icon = "👑"
        title = "Pelanggan Loyal"
        color = "#E8F5E9"
        recommendation = "Berikan program loyalitas, diskon eksklusif, dan rekomendasi buku premium."
    elif row["Rata_Frekuensi"] > avg_freq:
        icon = "🌟"
        title = "Pelanggan Potensial"
        color = "#E3F2FD"
        recommendation = "Tingkatkan pembelian dengan promo bundling dan rekomendasi buku yang relevan."
    else:
        icon = "📌"
        title = "Pelanggan Pasif"
        color = "#FFF3E0"
        recommendation = "Gunakan strategi promosi dan diskon untuk menarik pelanggan kembali bertransaksi."

    # Membuat kartu interpretasi modern
    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:20px;
            border-radius:15px;
            margin-bottom:15px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);
        ">
            <h3 style="margin:0;">{icon} Cluster {i} — {title}</h3>
            <p style="margin-top:10px;">
                <b>Jumlah Pelanggan:</b> {int(row["Jumlah_Pelanggan"])} pelanggan
            </p>
            <p>
                <b>Rata-rata Frekuensi:</b> {row["Rata_Frekuensi"]:.2f}
            </p>
            <p>
                <b>Rata-rata Jumlah Buku:</b> {row["Rata_Jumlah_Buku"]:.2f}
            </p>
            <p>
                <b>Rata-rata Total Belanja:</b> Rp {row["Rata_Total_Belanja"]:,.0f}
            </p>
            <hr style="border:0;border-top:1px solid #ddd;">
            <p>
                <b>Rekomendasi Strategi:</b> {recommendation}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
# ======================
# Download Hasil
# ======================
csv = customer.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Hasil Segmentasi CSV",
    data=csv,
    file_name="hasil_segmentasi_pelanggan_toko_buku.csv",
    mime="text/csv"
)
