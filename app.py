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

# FORCE SYSTEM PERMISSIONS IMMEDIATELY AT RUNTIME
os.chmod(CACHE_DIR, 0o777)

# Google Drive IDs mapped cleanly for automated retrieval scripts
KNOWLEDGE_BASE = {
    "english": {
        "VegetablesbyBayerTomatoDiseaseGuide.pdf": "1lziyd4oXiiWK8zGBzz9zz12JSRlOiufy",
        "Man_Maize_diseases_CIMMYT.pdf": "1LzwK91UP8sBZ0dgnAWgjTfX9bXKTl0zS",
        "PRODUCTION-GUIDE-ON-TOMATO.pdf": "1BLWpBleJzN8icgpoyJuiwJRtgSK9Z8Rw",
        "PestanddiseasemanualallPRAMandASHC.pdf": "1aFo6Y57zheat6-FgwttnbBzwnHFl9EjL",
        "Concise-Encyclopedia-of-Plant-Diseases.pdf": "1ugRejJFvFKKCeTRR5jWrehYw6TUzaJZB"
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
        # If offline, this system call won't crash the Streamlit app interface
        try:
            os.system("pip install gdown")
            import gdown
        except Exception:
            return

    for lang, books in KNOWLEDGE_BASE.items():
        lang_dir = os.path.join(RAG_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)
        for filename, file_id in books.items():
            destination_path = os.path.join(lang_dir, filename)
            # Checks local disk first to avoid touching the network if file is present
            if not os.path.exists(destination_path):
                try:
                    with st.spinner(f"Downloading {filename} from cloud systems..."):
                        gdown.download(id=file_id, output=destination_path, quiet=True)
                except Exception:
                    pass

# SAFETY NET FOR LINE 81: Wrap the call so the app starts smoothly even if completely offline
try:
    ensure_books_exist()
except Exception:
    pass

def ensure_model_exists():
    """Checks for the Qwen GGUF model and auto-downloads it from Hugging Face if missing."""
    if not os.path.exists(MODEL_PATH):
        hf_url = f"https://huggingface.co"
        st.info(" GGUF Model file not found. Starting automatic download from Hugging Face (~382MB)...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            import urllib.request
            def download_progress(count, block_size, total_size):
                percent = min(int(count * block_size * 100 / total_size), 100)
                progress_bar.progress(percent / 100)
                status_text.text(f"Downloading model: {percent}% complete")
            urllib.request.urlretrieve(hf_url, MODEL_PATH, reporthook=download_progress)
            st.success("Model downloaded successfully! Initializing LLM engine...")
            status_text.empty()
            progress_bar.empty()
        except Exception as e:
            st.error(f"Failed to download model from Hugging Face: {e}")

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
    "Yoruba: Bí ẹniyàn bá șegbingbin, béèni yóò șekórè. (As we sow, so shall we reap.)",
    "Hausa: Mai Hakuri yukan dafa dutse har ya sha romonsa. (The patient farmer cooks a stone and drinks its soup.)",
    "Swahili: Mvumilivuh kula mbivu. (A patient person eats ripe fruit.)",
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
    """Processes search arrays, retrieves reference pages cleanly, and enforces 0.0 deterministic bounds."""
    cultural_closing = "\n\n*Allahu ya bada amfanin gona mai albarka! Madalla da yin nagari!*" if lang == "Hausa" else "\n\n*May your harvest be heavy and rewarding!*"
    
    # 1. Establish strict guardrail system prompts
    if lang == "Hausa":
        active_db = hausa_db
        fallback_msg = "Ba a sami alamun cutar a cikin littafin gona ba. Don Allah a sake duba alamun."
        system_instruction = (
            "Kai babban masanin shawarwari na aikin gona ne na Afirka.\n"
            "HAKKI: Ordered to use information from 'Bayani Daga Littafi' ONLY to answer the question.\n"
            "Idan bayanan ba su ƙunshi amsar ba, dan na rubuta: 'Symptom ba a samu a cikin littafin gona ba.'\n"
            "GARGADI: Kada ka yi amfani da sanin kanka na ciki. Rubuta amsarka cikin Harshen Hausa kawai."
        )
        context_label = "Bayani Daga Littafi"
    else:
        active_db = english_db
        fallback_msg = "Symptom not found in the local textbook manual. Please try rephrasing."
        system_instruction = (
            "You are a strict, offline African agricultural text reader.\n"
            "CRITICAL ORDER: Answer the user's question using ONLY the provided 'FactsheetContext' below.\n"
            "If the context does not explicitly mention or resolve the issue, your final answer MUST be exactly:\n"
            "'Symptom not found in the local textbook manual.'\n"
            "Do NOT use external pre-trained knowledge, do NOT extrapolate, and do NOT create fake citations."
        )
        context_label = "FactsheetContext"

    matched_fact = ""

    # 2. Vector Search Retrieval with a strict Similarity Score Threshold
    if encoder is not None and active_db["embeddings"] is not None and len(active_db["chunks"]) > 0:
        try:
            query_embedding = encoder.encode(user_input, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, active_db["embeddings"]).cpu().numpy()[0]
            best_match_idx = int(np.argmax(cos_scores))
            highest_score = cos_scores[best_match_idx]
            
            # CRITICAL THRESHOLD: If the book does not match the query by at least 30%, reject it
            if highest_score >= 0.30:
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
                        # FIX: convert_from_path returns a list; access index 0 to save the image
                        images[0].save(img_path, "PNG")
                        st.session_state.current_page_img = img_path
            else:
                # Force fallback if similarity score is too low
                return f"{fallback_msg}{cultural_closing}"
        except Exception as e:
            st.error(f"Error rendering PDF page image: {e}")

    # Fallback to structural message if the database is dry or empty
    if not matched_fact.strip():
        return f"{fallback_msg}{cultural_closing}"
        
    # STRATEGIC FORCED ROUTING: Use raw text match for Hausa, require LLM for English
    if lang == "Hausa":
        prefix = "**Tabbataccen Bayani Daga Littafi:**\n\n"
        return f"{prefix}{matched_fact}{cultural_closing}"
        
    # English path continues to the LLM core processor
    if (not LLAMA_AVAILABLE) or (llm is None):
        prefix = "**Offline Semantic Match:**\n\n"
        return f"{prefix}{matched_fact}{cultural_closing}"
        
    # 3. Secure prompt payload creation for English LLM generation
    try:
        prompt = (
            f"<|im_start|>system\n{system_instruction}\n{context_label}:{matched_fact}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        # Enforcing exact 0.0 behavior configurations for English
        response = llm(
            prompt,
            max_tokens=200,
            temperature=0.0,  # Strict accuracy lock
            top_p=1.0,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>", "User:", "System:"]
        )
        
        ai_response = response['choices']['text'].strip()
        ai_response = re.sub(r'[\u4e00-\u9fff]+', '', ai_response)  # Cleanup formatting artifacts
        
        if len(ai_response) <= 3:
            ai_response = f"Farming Truth Block: {matched_fact}"
            
        return f"{ai_response}{cultural_closing}"
        
    except Exception:
        # If the LLM engine fails or runs out of memory, safely output the raw textbook text
        prefix = "**Offline Semantic Match:**\n\n"
        return f"{prefix}{matched_fact}{cultural_closing}"
#=====================================================================
# DICTIONARY TRANSLATION DICTIONARY (Fixed Missing Tab Elements)
#=====================================================================
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
        "text_input_label": "Describe crop symptoms:",
        "diagnose_tab": "AI Advisor",
        "calendar_tab": "Timeline Calculator",
        "finance_tab": "Financial Ledger"
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
        "log_btn": "Shigar da Bayanin Kudi",
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
            return (f"Germination Expected: {germination.strftime('%B %d, %Y')}\n"
                    f"Flowering/Tasseling Stage: {flowering.strftime('%B %d, %Y')}\n"
                    f"Harvest Readiness Target: {harvest.strftime('%B %d, %Y')}")
        elif crop == "Cassava":
            root_initiation = planting_date + datetime.timedelta(days=30)
            canopy_closure = planting_date + datetime.timedelta(days=90)
            harvest = planting_date + datetime.timedelta(days=300)
            return (f"Root Initiation Phase: {root_initiation.strftime('%B %d, %Y')}\n"
                    f"Full Canopy Development: {canopy_closure.strftime('%B %d, %Y')}\n"
                    f"Harvest Readiness Target: {harvest.strftime('%B %d, %Y')}")
    except Exception as e:
        return f"Timeline calculator error: {e}"

def parse_financial_statement(statement_text):
    text_lower = statement_text.lower()
    numbers = [float(s) for s in re.findall(r'\d+', text_lower)]
    amount = numbers if numbers else 0.0
    
    if "sold" in text_lower or "sayar" in text_lower or "revenue" in text_lower:
        st.session_state.revenue += amount
        return f"Automatically identified a sale! Logged +{amount:,.2f} Naira to Revenue."
    elif "labour" in text_lower or "lebur" in text_lower or "worker" in text_lower:
        st.session_state.labour_cost += amount
        return f"Logged -{amount:,.2f} Naira to Labour Costs."
    elif "fertilizer" in text_lower or "taki" in text_lower or "chemical" in text_lower:
        st.session_state.fertilizer_cost += amount
        return f"Logged -{amount:,.2f} Naira to Fertilizer Costs."
    elif "rent" in text_lower or "tractor" in text_lower or "kayan aiki" in text_lower:
        st.session_state.equipment_cost += amount
        return f"Logged -{amount:,.2f} Naira to Equipment Costs."
    else:
        st.session_state.other_expenses += amount
        return f"Categorized generic ledger transaction entry: -{amount:,.2f} Naira logged."

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
            if st.button(labels["submit_btn"], type="primary", key="main_diagnostic_trigger"):
                if user_text.strip():
                    with st.spinner("Analyzing symptoms..." if selected_lang == "English" else "Ana duba alamun..."):
                        st.session_state.saved_user_text = user_text
                        st.session_state.last_ai_response = run_ai_advisory(user_text, selected_lang)
        with col_btn2:
            if st.button("Delete & Clear Inputs / Goge Bayanai", key="clear_inputs_btn"):
                st.session_state.input_counter += 1
                st.session_state.current_page_img = None
                st.session_state.current_page_num = None
                st.session_state.current_book_name = None
                st.session_state.last_ai_response = None
                st.rerun()

        # Render the text response inside the chat column
        if st.session_state.get("last_ai_response"):
            st.markdown("---")
            st.subheader("Advisor Response" if selected_lang == "English" else "Shafar Shawarwari")
            st.write(st.session_state.last_ai_response)

    # --- COLUMN 2: ENCYCLOPEDIA REFERENCE VIEWER ---
    with col_viewer:
        st.subheader("Encyclopedia Reference Viewer" if selected_lang == "English" else "Shafar Karatun Littafi")
        
        # FIXED: This visual page drawer now coordinates states cleanly without rerun interruptions
        if st.session_state.current_page_img and os.path.exists(st.session_state.current_page_img):
            st.markdown(f"**Source Document:** `{st.session_state.current_book_name}`")
            st.markdown(f"**Verified Matches Located on Page:** `{st.session_state.current_page_num}`")
            st.image(
                st.session_state.current_page_img,
                caption="Authentic textbook reference page rendered completely offline." if selected_lang == "English" else "Hoton littafi na gaskiya da aka ciro ba tare da intanet ba.",
                use_container_width=True
            )
        else:
            default_info = (
                "When you search for crop symptoms, the authentic visual textbook page matching your diagnosis will render here instantly completely offline."
            ) if selected_lang == "English" else (
                "Lokacin da kace bincika alamun cututtuka, shafin littafi gaskiyan agaske wanda ya dace da gano ku zai fito anan take ba tare da intanet ba."
            )
            st.info(default_info)

# --- TAB 2: TIMELINE METRIC ENGINE ---
with tab2:
    selected_crop = st.selectbox(labels["crop_select"], ["Maize", "Cassava"], key="tab2_crop_selector")
    planting_date = st.date_input(labels["date_input"], datetime.date.today(), key="tab2_date_picker")
    if st.button(labels["calc_btn"], key="tab2_generate_timeline_btn"):
        st.text(calculate_crop_timeline(selected_crop, planting_date))

# --- TAB 3: FINANCIAL LEDGER ---
with tab3:
    st.markdown("### Enter New Transactions / Shigar da Kudi")
    nlp_statement = st.text_input(
        labels["ledger_input"],
        key=f"nlp_stmt_{st.session_state.get('input_counter', 0)}"
    )
    
    if st.button(labels["log_btn"]):
        if nlp_statement:
            # Safely unpack the single number matrix directly from your custom parser
            text_lower = nlp_statement.lower()
            numbers = [float(s) for s in re.findall(r'\d+', text_lower)]
            amount = numbers[0] if numbers else 0.0
            
            # Context transaction classification layout engine routing rules
            if "sold" in text_lower or "sayar" in text_lower or "revenue" in text_lower:
                st.session_state.revenue += amount
                st.info(f"Automatically identified a sale! Logged +{amount:,.2f} Naira to Revenue.")
            elif "labour" in text_lower or "lebur" in text_lower or "worker" in text_lower:
                st.session_state.labour_cost += amount
                st.info(f"Logged -{amount:,.2f} Naira to Labour Costs.")
            elif "fertilizer" in text_lower or "taki" in text_lower or "chemical" in text_lower:
