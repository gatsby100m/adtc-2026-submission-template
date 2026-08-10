import os
import re
import datetime
import time
import urllib.request
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Try importing local AI, vector, and PDF rendering libraries
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from pypdf import PdfReader
    from pdf2image import convert_from_path
    PDF_LIBS_AVAILABLE = True
except ImportError:
    PDF_LIBS_AVAILABLE = False

#=====================================================================
# DIRECTORIES & AUTO-DOWNLOAD CONFIGURATION
#=====================================================================
MODEL_DIR = "models"
MODEL_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
RAG_DIR = "rag_data"
CACHE_DIR = "page_cache"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Grouping file IDs by language for the RAG pipeline
KNOWLEDGE_BASE = {
    "english": {
        "VegetablesbyBayerTomatoDiseaseGuide...": "1gQ29XZTsMYNS6kdA22rUG18iML6q6ZHA",
        "Man_Maize_diseases_CIMMYT.pdf": "14U3dBZSdbJI5j07jpzj61wLh6lwxLAyD",
        "PRODUCTION-GUIDE-ON-TOMATO.pdf": "1jokdh9e3D1UVnYm-vrC5ov3tXwgS1KsY",
        "PestanddiseasemanualallPRAMandASHC.pdf": "1KRdC35MF1VLqgGzO3A6W5HUoaoNgDUWM",
        "322147478-Concise-Encyclopedia-of-Plant-...": "1cJgi9eGnx35CEMFziMoE8nyEKmfHoWxi"
    },
    "hausa": {
        "VegetablesbyBayerTomatoDiseaseGuide...": "1gQ29XZTsMYNS6kdA22rUG18iML6q6ZHA",
        "Man_Maize_diseases_CIMMYT.pdf": "14U3dBZSdbJI5j07jpzj61wLh6lwxLAyD",
        "PRODUCTION-GUIDE-ON-TOMATO.pdf": "1jokdh9e3D1UVnYm-vrC5ov3tXwgS1KsY",
        "PestanddiseasemanualallPRAMandASHC.pdf": "1KRdC35MF1VLqgGzO3A6W5HUoaoNgDUWM",
        "322147478-Concise-Encyclopedia-of-Plant-...": "1cJgi9eGnx35CEMFziMoE8nyEKmfHoWxi"
    }
}

def ensure_books_exist():
    """Checks data directory and auto-downloads your Google Drive textbooks using gdown into subfolders."""
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown")
        import gdown
        
    for lang, books in KNOWLEDGE_BASE.items():
        lang_dir = os.path.join(RAG_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)
        for filename, file_id in books.items():
            if "PASTE_HAUSA_ID" in file_id:
                continue
            destination_path = os.path.join(lang_dir, filename)
            if not os.path.exists(destination_path):
                with st.spinner(f"Downloading {filename} ({lang}) from Google Drive... Please wait."):
                    try:
                        gdown.download(id=file_id, output=destination_path, quiet=True)
                    except Exception as e:
                        st.error(f"Could not auto-download {filename}. Error: {e}")

# Run background textbook loader immediately
ensure_books_exist()

#=====================================================================
# FIXED: INITIALIZE MISSING GLOBAL LLM AND ENCODER MODELS
#=====================================================================
@st.cache_resource
def load_ai_models():
    """Safely instantiates embedding structures and local quantized model cores."""
    loaded_encoder = None
    loaded_llm = None
    
    if TRANSFORMERS_AVAILABLE:
        try:
            loaded_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pass
            
    if LLAMA_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            loaded_llm = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)
        except Exception:
            pass
    return loaded_encoder, loaded_llm

# Unpack models globally so all referencing functions can access them without NameErrors
encoder, llm = load_ai_models()

#=====================================================================
# CORE RESOURCE INITIALIZATION CORE (Optimized for 8GB RAM / HF Spaces)
#=====================================================================
@st.cache_resource
def index_all_downloaded_books_by_lang():
    """Loops through all downloaded PDFs partitioned by language subfolders."""
    if not TRANSFORMERS_AVAILABLE or not PDF_LIBS_AVAILABLE or encoder is None:
        return {}
    lang_indices = {}
    for lang in ["english", "hausa"]:
        lang_dir = os.path.join(RAG_DIR, lang)
        if not os.path.exists(lang_dir):
            continue
        master_chunks = []
        master_metadata = []
        for filename in os.listdir(lang_dir):
            if filename.endswith(".pdf"):
                file_path = os.path.join(lang_dir, filename)
                try:
                    reader = PdfReader(file_path)
                    for idx, page in enumerate(reader.pages):
                        raw_text = page.extract_text() or ""
                        if len(raw_text.strip()) > 50:
                            master_chunks.append(raw_text)
                            master_metadata.append({
                                "file_name": filename,
                                "file_path": file_path,
                                "page_num": idx + 1
                            })
                except Exception:
                    continue
        if master_chunks:
            with st.spinner(f"Indexing West African Crop Knowledge Bases ({lang.upper()})..."):
                db_embeddings = encoder.encode(master_chunks, convert_to_tensor=True, show_progress_bar=False)
                lang_indices[lang] = {
                    "chunks": master_chunks,
                    "metadata": master_metadata,
                    "embeddings": db_embeddings
                }
    return lang_indices

#=========================================================================
# 1. DEDICATED ENGLISH INDEX PIPELINE
#=========================================================================
@st.cache_resource
def index_english_library():
    """Independent engine that ONLY extracts and indexes English PDFs."""
    if not TRANSFORMERS_AVAILABLE or not PDF_LIBS_AVAILABLE or encoder is None:
        return {"chunks": [], "metadata": [], "embeddings": None}
    english_dir = os.path.join(RAG_DIR, "english")
    if not os.path.exists(english_dir):
        return {"chunks": [], "metadata": [], "embeddings": None}
    chunks, metadata = [], []
    for filename in os.listdir(english_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(english_dir, filename)
            try:
                reader = PdfReader(file_path)
                for idx, page in enumerate(reader.pages):
                    raw_text = page.extract_text() or ""
                    if len(raw_text.strip()) > 50:
                        chunks.append(raw_text)
                        metadata.append({"file_name": filename, "file_path": file_path, "page_num": idx + 1})
            except Exception:
                continue
    if chunks:
        try:
            embeddings = encoder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
            return {"chunks": chunks, "metadata": metadata, "embeddings": embeddings}
        except Exception:
            return {"chunks": [], "metadata": [], "embeddings": None}
    return {"chunks": [], "metadata": [], "embeddings": None}

#=========================================================================
# 2. DEDICATED HAUSA INDEX PIPELINE
#=========================================================================
@st.cache_resource
def index_hausa_library():
    """Independent engine that ONLY extracts and indexes Hausa PDFs."""
    if not TRANSFORMERS_AVAILABLE or not PDF_LIBS_AVAILABLE or encoder is None:
        return {"chunks": [], "metadata": [], "embeddings": None}
    hausa_dir = os.path.join(RAG_DIR, "hausa")
    if not os.path.exists(hausa_dir):
        return {"chunks": [], "metadata": [], "embeddings": None}
    chunks, metadata = [], []
    for filename in os.listdir(hausa_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(hausa_dir, filename)
            try:
                reader = PdfReader(file_path)
                for idx, page in enumerate(reader.pages):
                    raw_text = page.extract_text() or ""
                    if len(raw_text.strip()) > 50:
                        chunks.append(raw_text)
                        metadata.append({"file_name": filename, "file_path": file_path, "page_num": idx + 1})
            except Exception:
                continue
    if chunks:
        try:
            embeddings = encoder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
            return {"chunks": chunks, "metadata": metadata, "embeddings": embeddings}
        except Exception:
            return {"chunks": [], "metadata": [], "embeddings": None}
    return {"chunks": [], "metadata": [], "embeddings": None}

#=========================================================================
# 3. RUNTIME INITIALIZATION & BACKWARD COMPATIBILITY
#=========================================================================
english_db = index_english_library()
hausa_db = index_hausa_library()

db_chunks = english_db["chunks"]
db_metadata = english_db["metadata"]
db_embeddings = english_db["embeddings"]

CULTURAL_PROVERBS = [
    "Yoruba: Bí énìyàn bá șegbingbin, béèni yóò șekórè. (As we sow, so shall we reap.)",
    "Hausa: Mai hakuri yukan dafa dutse har ya sha romonsa. (The patient farmer cooks a stone and drinks its soup.)",
    "Swahili: Mvumilivu hula mbivu. (A patient person eats ripe fruit.)",
    "Igbo: Onye gbambo na ubi, owuwe ihe ubi ga-asacha anya mmiri ya. (He who labors in the field will have his tears wiped by the harvest.)"
]

#=====================================================================
# STREAMLIT GRAPHICAL INTERFACE (Part 2)
#=====================================================================
# Added missing tab keys to avoid blank headers or rendering errors
LANG_DICT = {
    "English": {
        "title": " SmartFarmAssistant",
        "subtitle": "AI-Powered West African Crop Advisor & Ledger Engine",
        "proverb_title": " Cultural Farm Wisdom",
        "submit_btn": "Analyze Symptoms",
        "crop_select": "Select Your Crop Type:",
        "date_input": "Select Planting Date:",
        "calc_btn": "Calculate Crop Timeline",
        "ledger_input": "Type transaction details (e.g., 'Sold maize for 50000 Naira'):",
        "log_btn": "Log Transaction Automatically",
        "diagnose_tab": "AI Advisor",
        "calendar_tab": "Timeline Calculator",
        "finance_tab": "Financial Ledger",
        "text_input_label": "Describe crop symptoms:"
    },
    "Hausa": {
        "title": " Mataimakin Manomi na AI",
        "subtitle": "Kwamfutar Shawarwari da Jagorancin Kudaden Gona",
        "proverb_title": " Karin Maganar Manoma",
        "submit_btn": "Bincika Alamomi",
        "crop_select": "Zabi Irin Amfanin Gona:",
        "date_input": "Zabi Ranar Shuka:",
        "calc_btn": "Lissafta Lokacin Gona",
        "ledger_input": "Rubuta bayanin kudi (misali, 'An sayar da masara kudin Naira 50000'):",
        "log_btn": "Shigarda Bayanin Kudi",
        "diagnose_tab": "Mataimakin AI",
        "calendar_tab": "Kalandar Gona",
        "finance_tab": "Littafin Kudi",
        "text_input_label": "Yi bayanin alamun rashin lafiyar amfanin gona:"
    }
}

if llm is None:
    st.warning("Application running in fallback lookup mode. AI vector features require active weights storage paths.")
else:
    st.success("AI Core, Multilingual Vector Map, and Local LLM Engine loaded successfully!")

col_lang, col_prov = st.columns(2)
with col_lang:
    selected_lang = st.selectbox("Language / Yare", ["English", "Hausa"])
    labels = LANG_DICT[selected_lang]
    
with col_prov:
    prov_idx = int(time.time() // 10) % len(CULTURAL_PROVERBS)
    st.info(f"**{labels['proverb_title']}**\n{CULTURAL_PROVERBS[prov_idx]}")

st.title(labels["title"])
st.subheader(labels["subtitle"])

def calculate_crop_timeline(crop, planting_date):
    try:
        if crop == "Maize":
            germination = planting_date + datetime.timedelta(days=5)
            flowering = planting_date + datetime.timedelta(days=55)
            harvest = planting_date + datetime.timedelta(days=110)
            return (f" Germination Expected: {germination.strftime('%B %d, %Y')}\n"
                    f" Flowering/Tasseling Stage: {flowering.strftime('%B %d, %Y')}\n"
                    f" Harvest Readiness Target: {harvest.strftime('%B %d, %Y')}")
        elif crop == "Cassava":
            root_initiation = planting_date + datetime.timedelta(days=30)
            canopy_closure = planting_date + datetime.timedelta(days=90)
            harvest = planting_date + datetime.timedelta(days=300)
            return (f" Root Initiation Phase: {root_initiation.strftime('%B %d, %Y')}\n"
                    f" Full Canopy Development: {canopy_closure.strftime('%B %d, %Y')}\n"
                    f" Harvest Readiness Target: {harvest.strftime('%B %d, %Y')}")
    except Exception as e:
        return f"Timeline calculator error: {e}"

def parse_financial_statement(statement_text):
    text_lower = statement_text.lower()
    numbers = [float(s) for s in re.findall(r'\d+', text_lower)]
    amount = numbers[0] if numbers else 0.0
    if "sold" in text_lower or "sayar" in text_lower or "revenue" in text_lower:
        st.session_state.revenue += amount
        return f" Automatically identified a sale! Logged +{amount:,.2f} Naira to Revenue."
    elif "labour" in text_lower or "lebur" in text_lower or "worker" in text_lower:
        st.session_state.labour_cost += amount
        return f" Logged -{amount:,.2f} Naira to Labour Costs."
    elif "fertilizer" in text_lower or "taki" in text_lower or "chemical" in text_lower:
        st.session_state.fertilizer_cost += amount
        return f" Logged -{amount:,.2f} Naira to Fertilizer Costs."
    elif "rent" in text_lower or "tractor" in text_lower or "kayan aiki" in text_lower:
        st.session_state.equipment_cost += amount
        return f" Logged -{amount:,.2f} Naira to Equipment Costs."
    else:
        st.session_state.other_expenses += amount
        return f" Categorized generic ledger transaction entry: -{amount:,.2f} Naira logged."

tab1, tab2, tab3 = st.tabs([
    labels.get("diagnose_tab"),
    labels.get("calendar_tab"),
    labels.get("finance_tab")
])

# --- TAB 1: AI ADVISOR & SYMPTOM INPUTS ---
with tab1:
    col_chat, col_viewer = st.columns([1.1, 0.9])
    with col_chat:
        st.markdown(f"### {labels.get('diagnose_tab')}")
        text_key = f"text_symptom_{st.session_state.get('input_counter', 0)}"
        audio_key = f"audio_symptom_{st.session_state.get('input_counter', 0)}"
        user_text = st.text_input(labels.get("text_input_label"), key=text_key)
        
        col_aud1, col_aud2 = st.columns(2)
        with col_aud1:
            user_audio = st.audio_input("Record audio symptoms / Rikodin sauti:", key=audio_key)
        with col_aud2:
            uploaded_audio = st.file_uploader(
                "Upload audio file / Dorawa sauti:",
                type=["wav", "mp3", "m4a", "ogg"],
                key=f"audio_file_uploader_{st.session_state.get('input_counter', 0)}"
            )
            
        if uploaded_audio is not None and user_audio is None:
            user_audio = uploaded_audio
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(labels["submit_btn"], type="primary", key="submit_symptom_btn"):
                if user_text:
                    with st.spinner("Processing analysis..."):
                        result = run_ai_advisory(user_text, selected_lang)
                        st.write(result)
                elif user_audio is not None:
                    st.info("Audio received locally. (Audio processing engine pipeline placeholder)")
                    with st.spinner("Processing analysis..."):
                        result = run_ai_advisory("spots", selected_lang)
                        st.write(result)
                else:
                    st.warning("Please provide either text or audio input first.")
        with col_btn2:
            if st.button("Delete & Clear Inputs / Goge Bayanai", key="clear_inputs_btn"):
                st.session_state.input_counter += 1
                st.session_state.current_page_img = None
                st.session_state.current_page_num = None
                st.session_state.current_book_name = None
                st.rerun()

    with col_viewer:
        viewer_title = " Encyclopedia Reference Viewer" if selected_lang == "English" else " Hoton Littafin Encyclopedia"
        st.markdown(f"### {viewer_title}")
        if st.session_state.current_page_img and os.path.exists(st.session_state.current_page_img):
            if "pest_disease_africa.pdf" in st.session_state.get("current_book_name", ""):
                info_msg = (
                    f" Matched Text Source: **Pests and Diseases of Tropical Crops** (Page {st.session_state.current_page_num})"
                    if selected_lang == "English" else
                    f" Bayani daga littafi: **Pests and Diseases of Tropical Crops** (Shafi na {st.session_state.current_page_num})"
                )
                st.info(info_msg)
                st.caption("This source is optimized as text-only reference data columns inside your RAM profile.")
            else:
                success_msg = (
                    f" Displaying relevant page {st.session_state.current_page_num} from `{st.session_state.current_book_name}`"
                    if selected_lang == "English" else
                    f" An nuna Shafi na {st.session_state.current_page_num} daga `{st.session_state.current_book_name}`"
                )
                st.success(success_msg)
                st.image(st.session_state.current_page_img, use_container_width=True)
        else:
            default_info = (
                "When you search for crop symptoms, the authentic visual textbook page matching your diagnosis will render here instantly completely offline."
                if selected_lang == "English" else
                "Bincika alamomin cututtuka zai nuna muku aihin shafin littafin aikin gona tare da hotuna ko jadawali a nan ba tare da internet ba."
            )
            st.info(default_info)

# --- TAB 2: TIMELINE CALCULATOR ---
with tab2:
    selected_crop = st.selectbox(labels["crop_select"], ["Maize", "Cassava"], key="tab2_crop_selector")
    planting_date = st.date_input(labels["date_input"], datetime.date.today(), key="tab2_date_picker")
    if st.button(labels["calc_btn"], key="tab2_generate_timeline_btn"):
        timeline_results = calculate_crop_timeline(selected_crop, planting_date)
        st.text(timeline_results)

# --- TAB 3: FINANCIAL LEDGER ---
with tab3:
    st.markdown("### Enter New Transactions / Shigarda Kudi")
    nlp_statement = st.text_input(
        labels["ledger_input"],
        key=f"nlp_stmt_{st.session_state.get('input_counter', 0)}"
    )
    if st.button(labels["log_btn"], key="tab3_nlp_log_btn"):
        if nlp_statement:
            parse_result = parse_financial_statement(nlp_statement)
            st.info(parse_result)
            st.rerun()
            
    st.markdown("---")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        sale_input = st.number_input("Crop Sales Revenue (Naira):", min_value=0.0, step=500.0, key="sale_in")
        if st.button("Add to Sales / Kara Kudin Sayarwa", key="tab3_add_sales_btn"):
            st.session_state.revenue += sale_input
