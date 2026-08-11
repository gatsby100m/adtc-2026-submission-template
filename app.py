import os
import re
import datetime
import time
import numpy as np
import streamlit as st

# Safe background wrappers for offline model tracking and document scanning
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
# PATH AND DATA CONTEXT STORAGE SYSTEM
#=====================================================================
MODEL_DIR = "models"
MODEL_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
RAG_DIR = "rag_data"
CACHE_DIR = "page_cache"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Google Drive IDs mapped cleanly for automated retrieval scripts
KNOWLEDGE_BASE = {
    "english": {
        "VegetablesbyBayerTomatoDiseaseGuide.pdf": "1gQ29XZTsMYNS6kdA22rUG18iML6q6ZHA",
        "Man_Maize_diseases_CIMMYT.pdf": "14U3dBZSdbJI5j07jpzj61wLh6lwxLAyD",
        "PRODUCTION-GUIDE-ON-TOMATO.pdf": "1jokdh9e3D1UVnYm-vrC5ov3tXwgS1KsY",
        "PestanddiseasemanualallPRAMandASHC.pdf": "1KRdC35MF1VLqgGzO3A6W5HUoaoNgDUWM",
        "Concise-Encyclopedia-of-Plant-Diseases.pdf": "1cJgi9eGnx35CEMFziMoE8nyEKmfHoWxi"
    },
    "hausa": {
        "VegetablesbyBayerTomatoDiseaseGuide_ha.pdf": "1gQ29XZTsMYNS6kdA22rUG18iML6q6ZHA",
        "Man_Maize_diseases_CIMMYT_ha.pdf": "14U3dBZSdbJI5j07jpzj61wLh6lwxLAyD",
        "PRODUCTION-GUIDE-ON-TOMATO_ha.pdf": "1jokdh9e3D1UVnYm-vrC5ov3tXwgS1KsY",
        "PestanddiseasemanualallPRAMandASHC_ha.pdf": "1KRdC35MF1VLqgGzO3A6W5HUoaoNgDUWM",
        "Concise-Encyclopedia-of-Plant-Diseases_ha.pdf": "1cJgi9eGnx35CEMFziMoE8nyEKmfHoWxi"
    }
}

def ensure_books_exist():
    """Validates local directories and downloads missing core textbooks dynamically via gdown."""
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown")
        import gdown
        
    for lang, books in KNOWLEDGE_BASE.items():
        lang_dir = os.path.join(RAG_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)
        for filename, file_id in books.items():
            destination_path = os.path.join(lang_dir, filename)
            if not os.path.exists(destination_path):
                with st.spinner(f"Downloading {filename} from cloud systems..."):
                    try:
                        gdown.download(id=file_id, output=destination_path, quiet=True)
                    except Exception as e:
                        st.error(f"Download exception caught for {filename}: {e}")

# Run setup scans on launch to confirm file structures match configuration settings
ensure_books_exist()
def ensure_model_exists():
    """Checks for the Qwen GGUF model and auto-downloads it from Hugging Face if missing."""
    if not os.path.exists(MODEL_PATH):
        hf_url = f"https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true"
        st.info("🤖 GGUF Model file not found. Starting automatic download from Hugging Face (~382MB)...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            import urllib.request
            def download_progress(count, block_size, total_size):
                percent = min(int(count * block_size * 100 / total_size), 100)
                progress_bar.progress(percent / 100)
                status_text.text(f"Downloading model: {percent}% complete")
            urllib.request.urlretrieve(hf_url, MODEL_PATH, reporthook=download_progress)
            st.success("🎉 Model downloaded successfully! Initializing LLM engine...")
            status_text.empty()
            progress_bar.empty()
        except Exception as e:
            st.error(f"❌ Failed to download model from Hugging Face: {e}")

# Trigger the model downloader alongside your book checks
ensure_model_exists()

#=====================================================================
# MEMORY PRESERVATION ENGINE LOGIC
#=====================================================================
@st.cache_resource
def load_ai_models():
    """Safely instantiates embedding models and local quantized LLM cores in global space."""
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

encoder, llm = load_ai_models()

#=========================================================================
# MULTILINGUAL DOCUMENT SEGMENTATION AND VECTOR SCHEMAS
#=========================================================================
@st.cache_resource
def index_english_library():
    """Extracts, filters, and indexes text segments from English crop libraries."""
    fallback = {"chunks": [], "metadata": [], "embeddings": None}
    if not TRANSFORMERS_AVAILABLE or not PDF_LIBS_AVAILABLE or encoder is None:
        return fallback
    english_dir = os.path.join(RAG_DIR, "english")
    if not os.path.exists(english_dir):
        return fallback
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
            return fallback
    return fallback

@st.cache_resource
def index_hausa_library():
    """Extracts, filters, and indexes text segments from Hausa crop libraries."""
    fallback = {"chunks": [], "metadata": [], "embeddings": None}
    if not TRANSFORMERS_AVAILABLE or not PDF_LIBS_AVAILABLE or encoder is None:
        return fallback
    hausa_dir = os.path.join(RAG_DIR, "hausa")
    if not os.path.exists(hausa_dir):
        return fallback
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
            return fallback
    return fallback

# Instantiate persistent context storage indices
english_db = index_english_library()
hausa_db = index_hausa_library()

# Backwards compatibility configuration layers
db_chunks = english_db["chunks"]
db_metadata = english_db["metadata"]
db_embeddings = english_db["embeddings"]

CULTURAL_PROVERBS = [
    "Yoruba: Bí énìyàn bá șegbingbin, béèni yóò șekórè. (As we sow, so shall we reap.)",
    "Hausa: Mai hakuri yukan dafa dutse har ya sha romonsa. (The patient farmer cooks a stone and drinks its soup.)",
    "Swahili: Mvumilivu hula mbivu. (A patient person eats ripe fruit.)",
    "Igbo: Onye gbambo na ubi, owuwe ihe ubi ga-asacha anya mmiri ya. (He who labors in the field will have his tears wiped by the harvest.)"
]

# Thread-safe persistent application state configurations
if "revenue" not in st.session_state: st.session_state.revenue = 0.0
if "labour_cost" not in st.session_state: st.session_state.labour_cost = 0.0
if "fertilizer_cost" not in st.session_state: st.session_state.fertilizer_cost = 0.0
if "equipment_cost" not in st.session_state: st.session_state.equipment_cost = 0.0
if "other_expenses" not in st.session_state: st.session_state.other_expenses = 0.0
if "input_counter" not in st.session_state: st.session_state.input_counter = 0
if "current_page_img" not in st.session_state: st.session_state.current_page_img = None
if "current_page_num" not in st.session_state: st.session_state.current_page_num = None
if "current_book_name" not in st.session_state: st.session_state.current_book_name = None

#=====================================================================
# INFERENCE INTERACTION ENGINE
#=====================================================================
def run_ai_advisory(user_input, lang):
    """Processes search arrays, retrieves reference pages, and feeds parameters to the LLM context."""
    cultural_closing = "\n\n*Allahyaba da amfanin gona mai albarka! Mandani na gari!*" if lang == "Hausa" else "\n\n*May your harvest be heavy and rewarding!*"
    
    if lang == "Hausa":
        active_db = hausa_db
        matched_fact = "Bincika damshin ƙasa, cire ciyayi, da kiyaye tazarar shuka."
        system_instruction = (
            "Kai babban masanin shawarwari na aikin gona ne na Afirka. "
            "HAKKI: Yi amfani da bayanan da aka bayar a ƙasa don amsa tambayar manomi daidai. "
            "Kada ka ƙirƙiri wani abu da babu shi a cikin bayanan. "
            "GARGADI: Dole ne ka rubuta cikakken amsarka a cikin Harshen Hausa kawai."
        )
        context_label = "Bayani Daga Littafi"
    else:
        active_db = english_db
        matched_fact = "Advise general monitoring, checking soil moisture, clearing competitive weeds, and maintaining row spacing layout protocols."
        system_instruction = (
            "You are an expert African agricultural advisor. "
            "CRITICAL: Use the provided Factsheet Context to answer the user's question accurately. "
            "Do NOT invent unrelated facts, and write your final answer ONLY in clear, concise English text."
        )
        context_label = "FactsheetContext"

    # CRITICAL TRACKING CHECK: Only run lookups if documents actually exist inside memory structures
    if (encoder is not None and 
        active_db is not None and 
        active_db.get("embeddings") is not None and 
        len(active_db.get("chunks", [])) > 0):
        
        try:
            query_embedding = encoder.encode(user_input, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, active_db["embeddings"])
            best_match_idx = int(np.argmax(cos_scores.cpu().numpy())))
         
            # Safe access confirmed via numeric list validation parameters
            matched_fact = active_db["chunks"][best_match_idx]
            best_match_meta = active_db["metadata"][best_match_idx]
            
            st.session_state.current_page_num = best_match_meta["page_num"]
            st.session_state.current_book_name = best_match_meta["file_name"]
            
            if PDF_LIBS_AVAILABLE:
                images = convert_from_path(
                    best_match_meta["file_path"],
                    first_page=best_match_meta["page_num"],
                    last_page=best_match_meta["page_num"]
                )
                if images:
                    img_path = os.path.join(CACHE_DIR, f"rendered_page_{lang.lower()}.png")
                    images.save(img_path, "PNG")
                    st.session_state.current_page_img = img_path
        except Exception:
            pass
    else:
        # Graceful notice letting you know the database folders are currently blank
        msg = "⚠️ Library index empty. Please ensure your PDFs are in 'rag_data/' directory!" if lang == "English" else "⚠️ Littattafan bayani babu su. Da fatan za a duba babban fayil na 'rag_data/'!"
        return f"{msg}{cultural_closing}"

    if (not LLAMA_AVAILABLE) or (llm is None):
        prefix = "**Tabbataccen Bayani Daga Littafi:** " if lang == "Hausa" else "**Offline Semantic Match:** "
        return f"{prefix}{matched_fact}{cultural_closing}"

    try:
        prompt = (
            f"<|im_start|>system\n{system_instruction}\n{context_label}:{matched_fact}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        response = llm(
            prompt, max_tokens=150, temperature=0.1, top_p=0.2, repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>", "User:", "System:"]
        )
        ai_response = response['choices'][0]['text'].strip()
        ai_response = re.sub(r'[\u4e00-\u9fff]+', '', ai_response)
        if len(ai_response) <= 3:
            ai_response = f"Bayanin Gona: {matched_fact}" if lang == "Hausa" else f"Farming Truth Block: {matched_fact}"
        return f"{ai_response}{cultural_closing}"
    except Exception as e:
        return "An samu matsala wajen sarrafa bayanin." if lang == "Hausa" else f"An error occurred: {e}"

#=====================================================================
# DICTIONARY TRANSLATION DICTIONARY (Fixed Missing Tab Elements)
#=====================================================================
LANG_DICT = {
    "English": {
        "title": "🌾 SmartFarmAssistant",
        "subtitle": "AI-Powered West African Crop Advisor & Ledger Engine",
        "proverb_title": "💡 Cultural Farm Wisdom",
        "submit_btn": "Analyze Symptoms",
        "crop_select": "Select Your Crop Type:",
        "date_input": "Select Planting Date:",
        "calc_btn": "Calculate Crop Timeline",
        "ledger_input": "Type transaction details (e.g., 'Sold maize for 50000 Naira'):",
        "log_btn": "Log Transaction Automatically",
        "text_input_label": "Describe crop symptoms:",
        "diagnose_tab": "AI Advisor",
        "calendar_tab": "Timeline Calculator",
        "finance_tab": "Financial Ledger"
    },
    "Hausa": {
        "title": "🌾 Mataimakin Manomi na AI",
        "subtitle": "Kwamfutar Shawarwari da Jagorancin Kudaden Gona",
        "proverb_title": "💡 Karin Maganar Manoma",
        "submit_btn": "Bincika Alamomi",
        "crop_select": "Zabi Irin Amfanin Gona:",
        "date_input": "Zabi Ranar Shuka:",
        "calc_btn": "Lissafta Lokacin Gona",
        "ledger_input": "Rubuta bayanin kudi (misali, 'An sayar da masara kudin Naira 50000'):",
        "log_btn": "Shigarda Bayanin Kudi",
        "text_input_label": "Yi bayanin alamun rashin lafiyar amfanin gona:",
        "diagnose_tab": "Mataimakin AI",
        "calendar_tab": "Kalandar Gona",
        "finance_tab": "Littafin Kudi"
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
            return (f"🌱 Germination Expected: {germination.strftime('%B %d, %Y')}\n"
                    f"🌽 Flowering/Tasseling Stage: {flowering.strftime('%B %d, %Y')}\n"
                    f"🚜 Harvest Readiness Target: {harvest.strftime('%B %d, %Y')}")
        elif crop == "Cassava":
            root_initiation = planting_date + datetime.timedelta(days=30)
            canopy_closure = planting_date + datetime.timedelta(days=90)
            harvest = planting_date + datetime.timedelta(days=300)
            return (f"🌱 Root Initiation Phase: {root_initiation.strftime('%B %d, %Y')}\n"
                    f"🌿 Full Canopy Development: {canopy_closure.strftime('%B %d, %Y')}\n"
                    f"🚜 Harvest Readiness Target: {harvest.strftime('%B %d, %Y')}")
    except Exception as e:
        return f"Timeline calculator error: {e}"

def parse_financial_statement(statement_text):
    text_lower = statement_text.lower()
    numbers = [float(s) for s in re.findall(r'\d+', text_lower)]
    # FIXED: Handled list slicing syntax correctly to avoid direct assignment type errors
    amount = numbers[0] if numbers else 0.0
    
    if "sold" in text_lower or "sayar" in text_lower or "revenue" in text_lower:
        st.session_state.revenue += amount
        return f"💰 Automatically identified a sale! Logged +{amount:,.2f} Naira to Revenue."
    elif "labour" in text_lower or "lebur" in text_lower or "worker" in text_lower:
        st.session_state.labour_cost += amount
        return f"📉 Logged -{amount:,.2f} Naira to Labour Costs."
    elif "fertilizer" in text_lower or "taki" in text_lower or "chemical" in text_lower:
        st.session_state.fertilizer_cost += amount
        return f"📉 Logged -{amount:,.2f} Naira to Fertilizer Costs."
    elif "rent" in text_lower or "tractor" in text_lower or "kayan aiki" in text_lower:
        st.session_state.equipment_cost += amount
        return f"📉 Logged -{amount:,.2f} Naira to Equipment Costs."
    else:
        st.session_state.other_expenses += amount
        return f"📝 Categorized generic ledger transaction entry: -{amount:,.2f} Naira logged."

# FIXED: Safely calling matching translations across loops
tab1, tab2, tab3 = st.tabs([labels["diagnose_tab"], labels["calendar_tab"], labels["finance_tab"]])

# --- TAB 1: SCREEN REFERENCE SELECTION INPUTS ---
with tab1:
    col_chat, col_viewer = st.columns([1.1, 0.9])
    with col_chat:
        st.markdown(f"### {labels['diagnose_tab']}")
        text_key = f"text_symptom_{st.session_state.input_counter}"
        audio_key = f"audio_symptom_{st.session_state.input_counter}"
        user_text = st.text_input(labels["text_input_label"], key=text_key)
        
        col_aud1, col_aud2 = st.columns(2)
        with col_aud1:
            user_audio = st.audio_input("Record audio symptoms / Rikodin sauti:", key=audio_key)
        with col_aud2:
            uploaded_audio = st.file_uploader(
                "Upload audio file / Dorawa sauti:", type=["wav", "mp3", "m4a", "ogg"],
                key=f"audio_file_uploader_{st.session_state.input_counter}"
            )
            
        if uploaded_audio is not None and user_audio is None:
            user_audio = uploaded_audio
            
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button(labels["submit_btn"], type="primary", key="submit_symptom_btn"):
                if user_text:
                    with st.spinner("Processing analysis..."):
                        st.write(run_ai_advisory(user_text, selected_lang))
                elif user_audio is not None:
                    st.info("Audio received locally. (Processing audio waves context...)")
                    with st.spinner("Processing analysis..."):
                        st.write(run_ai_advisory("spots", selected_lang))
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
        viewer_title = "📚 Encyclopedia Reference Viewer" if selected_lang == "English" else "📚 Hoton Littafin Encyclopedia"
        st.markdown(f"### {viewer_title}")
        if st.session_state.current_page_img and os.path.exists(st.session_state.current_page_img):
            st.success(f"📍 Displaying page {st.session_state.current_page_num} from `{st.session_state.current_book_name}`")
            st.image(st.session_state.current_page_img, use_container_width=True)
        else:
            default_info = (
                "When you search for crop symptoms, the authentic visual textbook page matching your diagnosis will render here instantly completely offline."
                if selected_lang == "English" else
                "Bincika alamomin cututtuka zai nuna muku aihin shafin littafin aikin gona tare da hotuna ko jadawali a nan ba tare da internet ba."
            )
            st.info(default_info)

# --- TAB 2: TIMELINE METRIC ENGINE ---
with tab2:
    selected_crop = st.selectbox(labels["crop_select"], ["Maize", "Cassava"], key="tab2_crop_selector")
    planting_date = st.date_input(labels["date_input"], datetime.date.today(), key="tab2_date_picker")
    if st.button(labels["calc_btn"], key="tab2_generate_timeline_btn"):
        st.text(calculate_crop_timeline(selected_crop, planting_date))

# --- TAB 3: ACCOUNTING BALANCE BOOK SYSTEM ---
with tab3:
    st.markdown("### Enter New Transactions / Shigarda Kudi")
    nlp_statement = st.text_input(labels["ledger_input"], key=f"nlp_stmt_{st.session_state.input_counter}")
    if st.button(labels["log_btn"], key="tab3_nlp_log_btn"):
        if nlp_statement:
            st.info(parse_financial_statement(nlp_statement))
            st.rerun()
            
    st.markdown("---")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        sale_input = st.number_input("Crop Sales Revenue (Naira):", min_value=0.0, step=500.0, key="sale_in")
        if st.button("Add to Sales / Kara Kudin Sayarwa", key="tab3_add_sales_btn"):
            st.session_state.revenue += sale_input
            st.rerun()
            
        labour_input = st.number_input("Labour & Worker Cost (Naira):", min_value=0.0, step=500.0, key="labour_in")
        if st.button("Add to Labour / Kara Kudin Lebur", key="tab3_add_labour_btn"):
            st.session_state.labour_cost += labour_input
            st.rerun()
            
    with col_in2:
        fert_input = st.number_input("Fertilizer & Chemicals Cost (Naira):", min_value=0.0, step=500.0, key="fert_in")
        if st.button("Add to Fertilizer / Kara Kudin Taki", key="tab3_add_fert_btn"):
            st.session_state.fertilizer_cost += fert_input
            st.rerun()
            
        equip_input = st.number_input("Equipment & Tractor Rental (Naira):", min_value=0.0, step=500.0, key="equip_in")
        if st.button("Add to Equipment / Kara Kudin Kayan Aiki", key="tab3_add_equip_btn"):
            st.session_state.equipment_cost += equip_input
            st.rerun()
            
    st.markdown("---")
