import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import time

# Optional matplotlib check (Styler.background_gradient requires matplotlib)
try:
    import matplotlib  # noqa: F401
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

download_nltk_data()

st.set_page_config(page_title="Neurax | Word Embeddings", layout="wide", initial_sidebar_state="expanded")

# Theme Selection Logic
theme_choice = st.sidebar.radio("UI Theme", ["Dark Mode 🌑", "Light Mode ☀️"], horizontal=True)

if theme_choice == "Dark Mode 🌑":
    bg_color = "#0f172a"
    text_color = "#f8fafc"
    sidebar_bg = "#1e293b"
    grid_color = "#334155"
    zeroline_color = "#475569"
    subtitle_color = "#94a3b8"
    label_color = "#cbd5e1"
    scene_bg = "#0f172a"
    marker_edge = "rgba(255,255,255,0.3)"
    widget_bg = "#1e293b"
    modebar_color = "rgba(255, 255, 255, 0.6)"
    modebar_active = "#38bdf8"
else:
    bg_color = "#f8fafc"
    text_color = "#0f172a"
    sidebar_bg = "#f1f5f9"
    grid_color = "#cbd5e1"
    zeroline_color = "#94a3b8"
    subtitle_color = "#64748b"
    label_color = "#475569"
    scene_bg = "#f8fafc"
    marker_edge = "rgba(0,0,0,0.3)"
    widget_bg = "#ffffff"
    modebar_color = "rgba(0, 0, 0, 0.6)"
    modebar_active = "#8b5cf6"

# --- Custom CSS for Attractive UI ---
st.markdown(f"""
<style>
/* Main Background */
.stApp {{
    background-color: {bg_color};
    font-family: 'Inter', sans-serif;
}}

/* Force Text Colors using more specific selectors to avoid overriding custom spans */
.stApp p:not(.gradient-text), 
.stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp span[class*="st-"] {{
    color: {text_color} !important;
}}

/* Widget Backgrounds (Text Area, Inputs, Selectbox, File Uploaders) */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > input,
div[data-baseweb="base-input"] > textarea,
section[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzoneInstructions"] > div > span {{
    background-color: {widget_bg} !important;
    color: {text_color} !important;
}}

/* Gradient Text for Title */
.gradient-text {{
    background: linear-gradient(to right, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 0;
}}
/* Subtitle */
.subtitle {{
    color: {subtitle_color} !important;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}}
/* Style Buttons */
.stButton>button {{
    background: linear-gradient(to right, #6366f1, #8b5cf6);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    width: 100%;
    transition: all 0.3s ease;
}}
.stButton>button:hover {{
    transform: scale(1.02);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}}
/* Sidebar Tweaks */
section[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    border-right: 1px solid {grid_color};
}}
/* Metric Card Enhancement */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {{
    color: #38bdf8 !important;
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="gradient-text">Neurax Embeddings</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced representation learning and visualization dashboard.</p>', unsafe_allow_html=True)

# Additional Sidebar Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Model Selection")
    model_type = st.selectbox("Choose Embedding Generation", ["Custom Word2Vec", "Pre-trained GloVe (Twitter 25d)"])
    
    if model_type == "Custom Word2Vec":
        st.subheader("Hyperparameters")
        vector_size = st.slider("Vector Size", 50, 300, 100, 50, help="Dimensionality of the feature vectors.")
        window_size = st.slider("Window Size", 2, 10, 5, help="Maximum distance between the current and predicted word within a sentence.")
        min_count = st.slider("Minimum Word Count", 1, 10, 2, help="Ignores all words with total frequency lower than this.")
        epochs = st.slider("Training Epochs", 5, 50, 15, help="Number of iterations (epochs) over the corpus.")
        
    st.subheader("Visualization")
    dim_red_technique = st.selectbox("Dimensionality Reduction", ["t-SNE (Non-Linear)", "PCA (Linear)"])
    n_components = st.radio("View Dimensions", [2, 3], horizontal=True)

# Main App Logic
col1, col2 = st.columns([1.2, 2.5])

with col1:
    st.markdown("### 📝 Data Source")
    if model_type == "Custom Word2Vec":
        data_source = st.radio("Provide text for training:", ["Sample Corpus", "Upload Text File"], horizontal=True)

        sample_text = """Artificial intelligence is a branch of computer science. Machine learning allows systems to learn from data without being explicitly programmed. Deep learning uses neural networks with many hidden layers. Natural language processing helps machines understand human languages. Word embeddings like Word2Vec and GloVe capture the semantic meanings of words in a high-dimensional continuous space. Visualization techniques like t-SNE reduce dimensions for human analysis. This is an advanced AI demonstration application."""

        raw_text = ""
        if data_source == "Sample Corpus":
            raw_text = st.text_area("Corpus Content:", sample_text, height=250)
        else:
            uploaded_file = st.file_uploader("Upload a .txt corpus", type=["txt"])
            if uploaded_file is not None:
                raw_text = uploaded_file.read().decode("utf-8")
                st.text_area("File Preview:", raw_text[:500] + "...", height=150)
    else:
        st.info("Using pre-trained GloVe representations. These were trained on 2 billion tweets.")
        st.markdown("Enter words you want to project into our visual space (comma separated):")
        default_words = "king, queen, royal, palace, man, woman, apple, orange, banana, car, vehicle, sad, happy, doctor, nurse, computer, software, internet, galaxy, universe"
        glove_words_input = st.text_area("Words to visualize:", default_words, height=180)

lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    # Use a regex-based tokenizer to avoid requiring NLTK punkt models
    tokens = re.findall(r"\b[a-z]{3,}\b", text)
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        stop_words = {"the", "and", "is", "in", "to", "of", "a", "for", "on", "with", "as", "by", "an", "be", "this", "that", "it"}
    # Advanced: Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

@st.cache_resource(show_spinner=False)
def load_glove_model():
    return api.load("glove-twitter-25")

with col2:
    if st.button("🚀 Process & Visualize"):
        
        if model_type == "Pre-trained GloVe (Twitter 25d)":
            with st.spinner("⏳ Downloading/Loading GloVe weights (may take a moment the first time)..."):
                start_time = time.time()
                try:
                    glove_model = load_glove_model()
                    load_time = time.time() - start_time
                    words = [w.strip().lower() for w in glove_words_input.split(',')]

                    # Filter to only words in vocab
                    valid_words = [w for w in set(words) if w in glove_model]
                    if len(valid_words) < 3:
                        st.error("Not enough valid words found in the GloVe vocabulary. Please use common English terms.")
                        st.stop()

                    vectors = np.array([glove_model[w] for w in valid_words])

                    model_to_query = glove_model
                    words_to_plot = valid_words

                    # Metrics Display
                    m1, m2 = st.columns(2)
                    m1.metric("Selected Words Plotted", len(words_to_plot))
                    m2.metric("GloVe Load Time", f"{load_time:.2f}s")

                except Exception as e:
                    st.error(f"Failed to load GloVe model: {e}")
                    st.stop()
                    
        else: # Custom Word2Vec
            if not raw_text.strip():
                st.error("Please provide some text data to train the model.")
                st.stop()

            # Preprocessing
            with st.spinner("⏳ Preprocessing text (Tokenization, Stopword Removal, Lemmatization)..."):
                 sentences_str = raw_text.split('.')
                 sentences = [preprocess_text(s) for s in sentences_str if len(s.strip()) > 0]
                 sentences = [s for s in sentences if len(s) > 0]

                 if not sentences:
                     st.error("Text does not contain valid words after cleaning.")
                     st.stop()

            with st.spinner("⏳ Training Advanced Word2Vec model..."):
                start_time = time.time()
                # Training
                model = Word2Vec(sentences, vector_size=vector_size, window=window_size, 
                                 min_count=min_count, epochs=epochs, workers=4)
                train_time = time.time() - start_time
                words_to_plot = list(model.wv.index_to_key)
                
                if len(words_to_plot) < 3:
                    st.error(f"Vocabulary too small ({len(words_to_plot)}). Try lowering Min Count or adding more text.")
                    st.stop()
                    
                vectors = np.array([model.wv[word] for word in words_to_plot])
                model_to_query = model.wv
                
                # Metrics Display
                st.markdown("##### 📈 Training Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("Vocabulary Size", len(words_to_plot))
                m2.metric("Corpus Sentences", len(sentences))
                m3.metric("Training Time", f"{train_time:.2f}s")
                
        # Dimensionality Reduction
        with st.spinner(f"⏳ Projecting {len(vectors)} vectors into {n_components}D via {dim_red_technique}..."):
            tech_name = dim_red_technique.split(" ")[0]
            if tech_name == "t-SNE":
                perplexity = min(30, max(2, len(words_to_plot) - 1))
                reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
            else:
                reducer = PCA(n_components=n_components)
                
            reduced_vectors = reducer.fit_transform(vectors)
            
            # Plotting DataFrame
            df = pd.DataFrame(reduced_vectors, columns=[f'Dim {i+1}' for i in range(n_components)])
            df['Word'] = words_to_plot

            # Plot styling using current theme
            marker_style = dict(size=14, line=dict(width=1, color=marker_edge), opacity=0.8)
            
            if n_components == 2:
                fig = px.scatter(df, x='Dim 1', y='Dim 2', text='Word', color='Dim 1', 
                                 color_continuous_scale='Magma', 
                                 title=f"Semantic Space Map ({tech_name})")
                fig.update_traces(textposition='bottom center', marker=marker_style, 
                                  textfont=dict(color=label_color, size=13, family='Inter'))
            else:
                fig = px.scatter_3d(df, x='Dim 1', y='Dim 2', z='Dim 3', text='Word', color='Dim 1', 
                                    color_continuous_scale='Magma',
                                    title=f"3D Semantic Space Map ({tech_name})")
                fig.update_traces(marker=dict(size=8, line=dict(width=0), opacity=0.9), 
                                  textfont=dict(color=text_color, size=11))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=text_color, family='Inter'),
                margin=dict(l=20, r=20, t=50, b=20),
                height=500,
                coloraxis_showscale=False,
                title_font=dict(size=20, color="#38bdf8"),
                modebar=dict(bgcolor='rgba(0,0,0,0)', color=modebar_color, activecolor=modebar_active)
            )
            
            if n_components == 2:
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=grid_color, zeroline=True, zerolinewidth=2, zerolinecolor=zeroline_color)
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=grid_color, zeroline=True, zerolinewidth=2, zerolinecolor=zeroline_color)
            else:
                fig.update_scenes(
                    xaxis_showgrid=True, yaxis_showgrid=True, zaxis_showgrid=True,
                    xaxis_gridcolor=grid_color, yaxis_gridcolor=grid_color, zaxis_gridcolor=grid_color,
                    bgcolor=scene_bg
                )

            st.plotly_chart(fig, width='stretch')
            
        st.divider()
        st.markdown("### 🔍 Semantic Similarity Engine")
        st.markdown(f"<p style='color:{subtitle_color};'>Extract the nearest neighbors vector-wise to determine context matching.</p>", unsafe_allow_html=True)

        search_col, result_col = st.columns([1, 1.5])
        with search_col:
            selected_word = st.selectbox("Query Word:", words_to_plot, index=0)
        with result_col:
            if selected_word:
                # Find most similar
                try:
                    sims = model_to_query.most_similar(selected_word, topn=4)
                    sim_df = pd.DataFrame(sims, columns=["Concept", "Correlation Score"])
                    if HAS_MATPLOTLIB:
                        st.dataframe(
                            sim_df.style.background_gradient(cmap="Purples", subset=["Correlation Score"]) 
                                  .format({"Correlation Score": "{:.4f}"}), 
                            width='stretch'
                        )
                    else:
                        sim_df["Correlation Score"] = sim_df["Correlation Score"].map(lambda x: f"{x:.4f}")
                        st.dataframe(sim_df, width='stretch')
                except Exception as e:
                    st.error(f"Could not compute similarities for '{selected_word}': {e}")