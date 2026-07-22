import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Dashboard Segmentasi Pelanggan Toko Buku",
    page_icon="📚",
    layout="wide"
)

# ======================
# Palet warna konsisten (senada dengan Set2 yang dipakai di scatter plot)
# ======================
PALETTE = px.colors.qualitative.Set2
ACCENT = "#66C2A5"      # hijau toska - warna utama
ACCENT_DARK = "#3C9F84"
BG_CARD = "#FFFFFF"
TEXT_MUTED = "#6B7280"

# ======================
# CSS Global — supaya seluruh dashboard terasa satu "bahasa desain"
# ======================
st.markdown(f"""
<style>
    .stApp {{
        background-color: #F7F9FA;
    }}

    /* Judul utama */
    h1 {{
        color: #1F2937;
        font-weight: 800;
    }}

    /* Subjudul section */
    h2, h3 {{
        color: #1F2937;
        font-weight: 700;
        padding-top: 0.4rem;
    }}

    /* Garis pemisah tiap section */
    .section-divider {{
        height: 3px;
        background: linear-gradient(90deg, {ACCENT} 0%, rgba(102,194,165,0) 100%);
        border-radius: 4px;
        margin: 0.2rem 0 1.2rem 0;
    }}

    /* Kartu metric ala "modern dashboard" */
    div[data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border: 1px solid #E5E7EB;
        border-left: 5px solid {ACCENT};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }}
    div[data-testid="stMetric"] label {{
        color: {TEXT_MUTED} !important;
        font-weight: 600;
    }}
    div[data-testid="stMetricValue"] {{
        color: #111827;
        font-weight: 800;
    }}

    /* Tabel/dataframe */
    div[data-testid="stDataFrame"] {{
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }}

    /* Slider label */
    .stSlider label {{
        font-weight: 600;
        color: #1F2937;
    }}

    /* Tombol download */
    .stDownloadButton button {{
        background-color: {ACCENT};
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 700;
        padding: 0.6rem 1.4rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }}
    .stDownloadButton button:hover {{
        background-color: {ACCENT_DARK};
        color: white;
    }}

    /* Kartu interpretasi cluster */
    .cluster-card {{
        background-color: {BG_CARD};
        border-radius: 12px;
        padding: 1rem 1.3rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        margin-bottom: 0.9rem;
    }}
</style>
""", unsafe_allow_html=True)


def section_title(title: str):
    """Judul section + garis aksen tipis di bawahnya, biar tiap bagian konsisten."""
    st.subheader(title)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


def style_table(df_to_style, money_cols=None, decimal_cols=None):
    """Beri gradasi warna senada aksen hijau pada tabel supaya lebih 'modern'."""
    money_cols = money_cols or []
    decimal_cols = decimal_cols or []
    numeric_cols = df_to_style.select_dtypes(include="number").columns.tolist()

    styler = df_to_style.style.background_gradient(
        subset=numeric_cols, cmap="Greens"
    )
    fmt = {}
    for c in money_cols:
        if c in df_to_style.columns:
            fmt[c] = "Rp {:,.0f}"
    for c in decimal_cols:
        if c in df_to_style.columns:
            fmt[c] = "{:.2f}"
    if fmt:
        styler = styler.format(fmt)
    return styler


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
Dashboard ini digunakan untuk menganalisis perilaku pelanggan toko buku dan mengelompokkan pelanggan
berdasarkan pola pembelian menggunakan **K-Means Clustering**.
""")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

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
section_title("📊 Ringkasan Dataset")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Transaksi", df["id_transaksi"].nunique())
col2.metric("Jumlah Pelanggan", df["nama_customer"].nunique())
col3.metric("Jumlah Buku Terjual", int(df["jumlah"].sum()))
col4.metric("Total Pendapatan", f"Rp {df['total'].sum():,.0f}")

# ======================
# Preview Dataset
# ======================
section_title("📋 Preview Dataset")
st.dataframe(df.head(), use_container_width=True)

# ======================
# Statistik Deskriptif
# ======================
section_title("📈 Statistik Deskriptif")

col_a, col_b = st.columns([1, 1.2])
with col_a:
    st.dataframe(
        df[["jumlah", "total"]].describe().style.background_gradient(cmap="Greens"),
        use_container_width=True
    )
with col_b:
    fig_hist = px.histogram(
        df, x="total", nbins=30,
        color_discrete_sequence=[ACCENT],
        title="Distribusi Total Transaksi"
    )
    fig_hist.update_layout(
        title_x=0.5, template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=330
    )
    st.plotly_chart(fig_hist, use_container_width=True)

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
section_title("👥 Analisis Perilaku Pelanggan")

col1, col2, col3 = st.columns(3)
col1.metric("Rata-rata Frekuensi", f"{customer['Frekuensi_Transaksi'].mean():.2f}")
col2.metric("Rata-rata Buku Dibeli", f"{customer['Jumlah_Buku'].mean():.2f}")
col3.metric("Rata-rata Total Belanja", f"Rp {customer['Total_Belanja'].mean():,.0f}")

# ======================
# Pengaturan Cluster
# ======================
section_title("⚙️ Pengaturan Clustering")

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

section_title("📏 Evaluasi Model")

col1, col2 = st.columns(2)
col1.metric("Silhouette Score", f"{silhouette:.4f}")
col2.metric("Davies-Bouldin Index", f"{dbi:.4f}")

# ======================
# Hasil Segmentasi
# ======================
section_title("📌 Hasil Segmentasi Pelanggan")
st.dataframe(
    style_table(customer, money_cols=["Total_Belanja"]),
    use_container_width=True
)

# ======================
# Visualisasi Cluster
# ======================
section_title("🎨 Visualisasi Cluster Pelanggan")

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
    color_discrete_sequence=PALETTE,
    title="Sebaran Cluster Pelanggan",
    width=620,
    height=400
)

fig.update_layout(
    title_x=0.5,
    template="plotly_white",
    margin=dict(l=10, r=10, t=50, b=10),
    legend_title_text="Cluster",
    font=dict(size=12)
)

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
section_title("🏆 Top 10 Pelanggan dengan Total Belanja Tertinggi")

top_customers = customer.sort_values(by="Total_Belanja", ascending=False).head(10)

col_a, col_b = st.columns([1, 1])
with col_a:
    st.dataframe(
        style_table(top_customers, money_cols=["Total_Belanja"]),
        use_container_width=True
    )
with col_b:
    fig_top = px.bar(
        top_customers.sort_values("Total_Belanja"),
        x="Total_Belanja",
        y="nama_customer",
        orientation="h",
        color="Total_Belanja",
        color_continuous_scale="Greens",
        title="Top 10 Pelanggan"
    )
    fig_top.update_layout(
        title_x=0.5, template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380, yaxis_title="", xaxis_title="Total Belanja (Rp)",
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_top, use_container_width=True)

# ======================
# Ringkasan Cluster
# ======================
section_title("📝 Ringkasan Cluster")

summary = customer.groupby("Cluster").agg(
    Jumlah_Pelanggan=("nama_customer", "count"),
    Rata_Frekuensi=("Frekuensi_Transaksi", "mean"),
    Rata_Jumlah_Buku=("Jumlah_Buku", "mean"),
    Rata_Total_Belanja=("Total_Belanja", "mean")
).reset_index()

col_a, col_b = st.columns([1, 1.1])
with col_a:
    st.dataframe(
        style_table(summary, money_cols=["Rata_Total_Belanja"], decimal_cols=["Rata_Frekuensi", "Rata_Jumlah_Buku"]),
        use_container_width=True
    )
with col_b:
    fig_summary = px.bar(
        summary,
        x="Cluster",
        y="Rata_Total_Belanja",
        color="Cluster",
        color_discrete_sequence=PALETTE,
        text_auto=".2s",
        title="Rata-rata Belanja per Cluster"
    )
    fig_summary.update_layout(
        title_x=0.5, template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380, showlegend=False, xaxis_title="Cluster", yaxis_title="Rata-rata Belanja (Rp)"
    )
    st.plotly_chart(fig_summary, use_container_width=True)

# ======================
# Interpretasi Cluster
# ======================
section_title("💡 Interpretasi Cluster")

for i in range(n_clusters):
    row = summary[summary["Cluster"] == i].iloc[0]

    if row["Rata_Total_Belanja"] > summary["Rata_Total_Belanja"].mean():
        label, icon, tone = "Pelanggan Loyal", "🟢", "sering membeli buku dan memiliki total belanja tinggi."
    elif row["Rata_Frekuensi"] > summary["Rata_Frekuensi"].mean():
        label, icon, tone = "Pelanggan Potensial", "🔵", "cukup sering bertransaksi dan berpotensi menjadi pelanggan loyal."
    else:
        label, icon, tone = "Pelanggan Pasif", "🟠", "jarang bertransaksi dan perlu strategi promosi untuk meningkatkan pembelian."

    st.markdown(f"""
    <div class="cluster-card">
        <h4 style="margin:0 0 0.3rem 0;">{icon} Cluster {i} — {label}</h4>
        <p style="margin:0; color:{TEXT_MUTED};">
            Pelanggan pada cluster ini {tone}<br>
            <b>{int(row['Jumlah_Pelanggan'])}</b> pelanggan ·
            rata-rata <b>{row['Rata_Frekuensi']:.1f}</b> transaksi ·
            rata-rata belanja <b>Rp {row['Rata_Total_Belanja']:,.0f}</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ======================
# Download Hasil
# ======================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

csv = customer.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Hasil Segmentasi CSV",
    data=csv,
    file_name="hasil_segmentasi_pelanggan_toko_buku.csv",
    mime="text/csv"
)
