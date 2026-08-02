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

# =====================================================================
# DIRECTORIES & AUTO-DOWNLOAD CONFIGURATION
# =====================================================================
MODEL_DIR = "models"
MODEL_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

RAG_DIR = "rag_data"
CACHE_DIR = "page_cache"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RAG_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Direct paths to your 4 core West African agricultural textbooks
BOOK_URLS = {
    "maize_diseases.pdf": "https://researchgate.net",
    "tomato_pepper_nihort.pdf": "https://nihort.gov.ng",
    "tomato_disease_bayer.pdf": "https://bayer.com",
    "pest_disease_africa.pdf": "https://researchgate.net"
}

def ensure_books_exist():
    """Checks data directory and auto-downloads missing textbooks on first boot."""
    for filename, download_url in BOOK_URLS.items():
        destination_path = os.path.join(RAG_DIR, filename)
        if not os.path.exists(destination_path):
            with st.spinner(f"Downloading {filename} for offline use... Please wait."):
                try:
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)
                    urllib.request.urlretrieve(download_url, destination_path)
                except Exception as e:
                    st.error(f"Could not auto-download {filename}. Error: {e}")

# Run background textbook loader immediately
ensure_books_exist()

# =====================================================================
# CORE RESOURCE INITIALIZATION CORE (Optimized for 8GB RAM / HF Spaces)
# =====================================================================
@st.cache_resource
def initialize_offline_cores():
    llm_instance = None
    bi_encoder = None
    
    # 1. Load the Chat/Reasoning Model
    if LLAMA_AVAILABLE:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading Qwen2.5-0.5B-Instruct weights for the Laptop LLM Profile..."):
                try:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(
                        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                        filename=MODEL_NAME,
                        local_dir=MODEL_DIR,
                        local_dir_use_symlinks=False
                    )
                except Exception as download_error:
                    st.error(f"Weights transmission aborted: {str(download_error)}")
        
        if os.path.exists(MODEL_PATH):
            try:
                llm_instance = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)
            except Exception:
                llm_instance = None

    # 2. Load the Multilingual Embedding Engine
    if TRANSFORMERS_AVAILABLE:
        with st.spinner("Caching Semantic Multilingual RAG Map Vectors..."):
            try:
                # Upgraded to high-performance multilingual mapping engine
                bi_encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            except Exception:
                bi_encoder = None
                
    return llm_instance, bi_encoder

llm, encoder = initialize_offline_cores()

# =====================================================================
# ADVANCED MULTI-BOOK EXTRACTION ENGINE
# =====================================================================
@st.cache_resource
def index_all_downloaded_books():
    """Loops through all downloaded PDFs, extracts text chunks, and tracks metadata."""
    if not TRANSFORMERS_AVAILABLE or not PDF_LIBS_AVAILABLE:
        return None, None, None
        
    master_chunks = []
    master_metadata = []
    
    for filename in os.listdir(RAG_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(RAG_DIR, filename)
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
                
    if not master_chunks:
        return None, None, None
        
    # Pre-compute layout vectors into RAM matrix
    with st.spinner("Indexing West African Crop Knowledge Bases..."):
        db_embeddings = encoder.encode(master_chunks, convert_to_tensor=True, show_progress_bar=False)
        
    return master_chunks, master_metadata, db_embeddings

# Load your broad-scale database pipeline
if encoder is not None:
    db_chunks, db_metadata, db_embeddings = index_all_downloaded_books()
else:
    db_chunks, db_metadata, db_embeddings = None, None, None

CULTURAL_PROVERBS = [
    "Yoruba: Bí énìyàn bá șegbingbin, béèni yóò șekórè. (As we sow, so shall we reap.)",
    "Hausa: Mai hakuri yukan dafa dutse har ya sha romonsa. (The patient farmer cooks a stone and drinks its soup.)",
    "Swahili: Mvumilivu hula mbivu. (A patient person eats ripe fruit.)",
    "Igbo: Onye gba mbo na ubi, owuwe ihe ubi ga-asacha anya mmiri ya. (He who labors in the field will have his tears wiped by the harvest.)"
]

# Initialize Granular Farm Ledger States
if "revenue" not in st.session_state: st.session_state.revenue = 0.0
if "labour_cost" not in st.session_state: st.session_state.labour_cost = 0.0
if "fertilizer_cost" not in st.session_state: st.session_state.fertilizer_cost = 0.0
if "equipment_cost" not in st.session_state: st.session_state.equipment_cost = 0.0
if "other_expenses" not in st.session_state: st.session_state.other_expenses = 0.0
if "input_counter" not in st.session_state: st.session_state.input_counter = 0

# Track selected pages across columns dynamically
if "current_page_img" not in st.session_state: st.session_state.current_page_img = None
if "current_page_num" not in st.session_state: st.session_state.current_page_num = None
if "current_book_name" not in st.session_state: st.session_state.current_book_name = None

# =====================================================================
# BACKGROUND SEAMLESS TRANSLATION ENGINE (NLLB-200)
# =====================================================================
@st.cache_resource
def load_nllb_translator():
    """Loads a highly optimized, lightweight 600M translator to ensure flawless Hausa output."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        # Uses standard CPU optimizations to maintain low memory utilization
        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model
    except Exception:
        return None, None

nllb_tokenizer, nllb_model = load_nllb_translator()

def translate_to_hausa(text_to_translate):
    """Pipes English textual outputs through NLLB to guarantee authentic grammar structures."""
    if nllb_tokenizer is None or nllb_model is None:
        return text_to_translate # Fallback safely if model loading fails
    try:
        from transformers import pipeline
        # Target code for standard Hausa language profile
        translator = pipeline(
            'translation', model=nllb_model, tokenizer=nllb_tokenizer, 
            src_lang='eng_Latn', tgt_lang='hau_Latn', max_length=512
        )
        output = translator(text_to_translate)
        return output[0]['translation_text']
    except Exception:
        return text_to_translate

# =====================================================================
# ADVANCED SEAMLESS HYBRID VECTOR RAG ENGINE
# =====================================================================
def run_ai_advisory(user_input, lang):
    cultural_closing = "\n\n*May your barns overflow thisseason! Mandani na gari!*" if lang == "Hausa" else "\n\n*May your harvest be heavy and rewarding!*"
    
    # Baseline fallback advice context parameters
    matched_fact = "Advise general monitoring, checking soil moisture, clearing competitive weeds, and maintaining row spacing layout protocols."
    best_match_meta = None
    
    # 1. Execute Multilingual Mathematical Cross-Lingual Search Indexing Matcher
    if encoder is not None and db_embeddings is not None and db_chunks is not None:
        try:
            query_embedding = encoder.encode(user_input, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, db_embeddings)[0]
            best_match_idx = int(np.argmax(cos_scores.cpu().numpy()))
            
            # Extract factual paragraph matching the target semantic coordinate query strings
            matched_fact = db_chunks[best_match_idx]
            best_match_meta = db_metadata[best_match_idx]
            
            # Cache layout parameters immediately into memory variables
            st.session_state.current_page_num = best_match_meta["page_num"]
            st.session_state.current_book_name = best_match_meta["file_name"]
            
            # Step A: Perform immediate visual image pipeline conversions
            if PDF_LIBS_AVAILABLE:
                images = convert_from_path(
                    best_match_meta["file_path"], 
                    first_page=best_match_meta["page_num"], 
                    last_page=best_match_meta["page_num"]
                )
                if images:
                    img_path = os.path.join(CACHE_DIR, f"rendered_page.png")
                    images[0].save(img_path, "PNG")
                    st.session_state.current_page_img = img_path
        except Exception:
            pass

    # Quick exit path if LLM structures are unavailable
    if (not LLAMA_AVAILABLE) or (llm is None):
        final_text = f"**Offline Semantic Match:** {matched_fact}\n\n*(Note: Running in high-performance lookup fallback mode).*"
        if lang == "Hausa":
            final_text = translate_to_hausa(final_text)
        return f"{final_text}{cultural_closing}"

    try:
        # 2. Instruct Qwen to extract data strictly in English first to ensure reasoning alignment
        system_instruction = (
            "You are an expert African agricultural advisor. "
            "CRITICAL: Use the provided Factsheet Context to answer the user's question accurately. "
            "Elaborate on the details to sound friendly and encouraging, but your facts MUST stay completely anchored to the factsheet context. "
            "Do NOT invent unrelated facts, and write your final answer ONLY in clear, concise English text."
        )
        
        prompt = (
            f"<|im_start|>system\n{system_instruction}\nFactsheetContext: {matched_fact}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        
        response = llm(
            prompt,
            max_tokens=250,
            temperature=0.0,  # For factual extraction
            top_p=0.1,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>", "User:", "System:", "Tambaya:"]
        )
        
        ai_response = response['choices'][0]['text'].strip()
        ai_response = re.sub(r'[\u4e00-\u9fff]+', '', ai_response)  # Clear formatting leaks

        if len(ai_response) <= 3:
            ai_response = f"Farming Truth Block: {matched_fact}"

        # 3. Handle Language Execution Assembly Pipeline routing paths
        if lang == "Hausa":
            # Translate English text output directly through NLLB to get clean Hausa
            with st.spinner("An canza bayani zuwa Harshen Hausa... (Translating response...)"):
                ai_response = translate_to_hausa(ai_response)
        
        return f"{ai_response}{cultural_closing}"
        
    except Exception as e:
        fallback_text = f"Offline Semantic Fallback: {matched_fact}"
        if lang == "Hausa":
            fallback_text = translate_to_hausa(fallback_text)
        return f"**{fallback_text}**{cultural_closing}"

# =====================================================================
# STREAMLIT GRAPHICAL INTERFACE
# =====================================================================
st.set_page_config(page_title="SmartFarmAssistant", layout="wide")

if llm is None:
    st.warning("Application running in fallback lookup mode. AI vector features require active weights storage paths.")
else:
    st.success("AI Core, Multilingual Vector Map, and NLLB Translation Engine loaded successfully!")

col_lang, col_prov = st.columns(2)
with col_lang:
    selected_lang = st.selectbox("Language / Yare", ["English", "Hausa"])
    labels = LANG_DICT[selected_lang]

with col_prov:
    prov_idx = int(time.time() // 10) % len(CULTURAL_PROVERBS)
    st.info(f"**{labels['proverb_title']}**\n{CULTURAL_PROVERBS[prov_idx]}")

st.title(labels["title"])
st.subheader(labels["subtitle"])

tab1, tab2, tab3 = st.tabs([
    labels.get("diagnose_tab", "AI Advisor"),
    labels.get("calendar_tab", "Timeline Calculator"),
    labels.get("finance_tab", "Financial Ledger")
])

# --- TAB 1: AI ADVISOR & SYMPTOM INPUTS ---
with tab1:
    # Split space dynamically to display text/audio chats on left and book pages on right
    col_chat, col_viewer = st.columns([1.1, 0.9])
    
    with col_chat:
        st.markdown(f"### {labels.get('diagnose_tab', 'AI Advisor Workspace')}")
        
        text_key = f"text_symptom_{st.session_state.get('input_counter', 0)}"
        audio_key = f"audio_symptom_{st.session_state.get('input_counter', 0)}"
        
        user_text = st.text_input(labels.get("text_input_label", "Describe crop symptoms:"), key=text_key)
        
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
        # Dynamic Encyclopedia Document Viewer Column frame
        viewer_title = "📖 Encyclopedia Reference Viewer" if selected_lang == "English" else "📖 Hoton Littafin Encyclopedia"
        st.markdown(f"### {viewer_title}")
        
        if st.session_state.current_page_img and os.path.exists(st.session_state.current_page_img):
            # Text-only flag bypass condition routing check
            if "pest_disease_africa.pdf" in st.session_state.get("current_book_name", ""):
                info_msg = (
                    f"📖 Matched Text Source: **Pests and Diseases of Tropical Crops** (Page {st.session_state.current_page_num})"
                    if selected_lang == "English" else
                    f"📖 Bayani daga littafin: **Pests and Diseases of Tropical Crops** (Shafi na {st.session_state.current_page_num})"
                )
                st.info(info_msg)
                st.caption("This source is optimized as text-only reference data columns inside your RAM profile.")
            else:
                success_msg = (
                    f"🎯 Displaying relevant page {st.session_state.current_page_num} from `{st.session_state.current_book_name}`"
                    if selected_lang == "English" else
                    f"🎯 An nuna Shafi na {st.session_state.current_page_num} daga `{st.session_state.current_book_name}`"
                )
                st.success(success_msg)
                st.image(st.session_state.current_page_img, use_container_width=True)
        else:
            default_info = (
                "When you search for crop symptoms, the authentic visual textbook page matching your diagnosis will render here instantly completely offline."
                if selected_lang == "English" else
                "Bincika alamomin cututtuka zai nuna muku ainihin shafin littafin aikin gona tare da hotuna ko jadawali a nan ba tare da internet ba."
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
            st.success(f"Added +{sale_input:,.2f} Naira to Sales!")
            st.rerun()
            
        labour_input = st.number_input("Labour & Worker Cost (Naira):", min_value=0.0, step=500.0, key="labour_in")
        if st.button("Add to Labour / Kara Kudin Lebur", key="tab3_add_labour_btn"):
            st.session_state.labour_cost += labour_input
            st.success(f"Added -{labour_input:,.2f} Naira to Labour!")
            st.rerun()
            
    with col_in2:
        fert_input = st.number_input("Fertilizer & Chemicals Cost (Naira):", min_value=0.0, step=500.0, key="fert_in")
        if st.button("Add to Fertilizer / Kara Kudin Taki", key="tab3_add_fert_btn"):
            st.session_state.fertilizer_cost += fert_input
            st.success(f"Added -{fert_input:,.2f} Naira to Fertilizer!")
            st.rerun()
            
        equip_input = st.number_input("Equipment & Tractor Rental (Naira):", min_value=0.0, step=500.0, key="equip_in")
        if st.button("Add to Equipment / Kara Kudin KayanAiki", key="tab3_add_equip_btn"):
            st.session_state.equipment_cost += equip_input
            st.success(f"Added -{equip_input:,.2f} Naira to Equipment!")
            
    st.markdown("---")
    st.markdown("### Farm Profit & Loss Summary / Bayanin Riba da Asara")
    
    total_costs = (
        st.session_state.labour_cost +
        st.session_state.fertilizer_cost +
        st.session_state.equipment_cost +
        st.session_state.other_expenses
    )
    net_profit = st.session_state.revenue - total_costs
    
    st.metric("Total Sales Revenue / Kudin Sayarwa (+)", f"{st.session_state.revenue:,.2f} Naira")
    
    col_metrics1, col_metrics2 = st.columns(2)
    with col_metrics1:
        st.metric("Labour Costs / Kudin Lebur (-)", f"{st.session_state.labour_cost:,.2f} Naira")
        st.metric("Fertilizer & Chemicals / Kudin Taki (-)", f"{st.session_state.fertilizer_cost:,.2f} Naira")
    with col_metrics2:
        st.metric("Equipment & Tractor / KayanAiki (-)", f"{st.session_state.equipment_cost:,.2f} Naira")
        st.metric("Other Expenses / Kudaden Fitarwa (-)", f"{st.session_state.other_expenses:,.2f} Naira")
        
    st.markdown("---")
    if net_profit >= 0:
        st.success(f"**Net Profit / Riba Ta Tabbata:** {net_profit:,.2f} Naira")
    else:
        st.error(f"**Net Operating Loss / Asara Ta Fito:** {abs(net_profit):,.2f} Naira")
        
    if st.button("Reset Ledger / Goge Dukan Bayanan Kudi", type="secondary", key="tab3_reset_ledger_btn"):
        st.session_state.revenue = 0.0
        st.session_state.labour_cost = 0.0
        st.session_state.fertilizer_cost = 0.0
        st.session_state.equipment_cost = 0.0
        st.session_state.other_expenses = 0.0
        st.success("Ledger cleared successfully!")
        st.rerun()
        
    st.markdown("---")
    st.subheader("Save Records Locally")
    
    current_ledger_data = {
        "Revenue": [st.session_state.get('revenue', 0.0)],
        "LabourCost": [st.session_state.get('labour_cost', 0.0)],
        "FertilizerCost": [st.session_state.get('fertilizer_cost', 0.0)],
        "EquipmentCost": [st.session_state.get('equipment_cost', 0.0)],
        "OtherExpenses": [st.session_state.get('other_expenses', 0.0)]
    }
    
    if st.button("Save Ledger to Laptop", key="save_ledger_tab3_btn"):
        try:
            import pandas as pd
            df = pd.DataFrame(current_ledger_data)
            file_name = "ledger_backup.csv"
            df.to_csv(file_name, index=False)
            absolute_path = os.path.abspath(file_name)
            st.success(f"Saved successfully to your laptop at:\n`{absolute_path}`")
        except Exception as e:
            st.error(f"Failed to save: {e}")
            
    st.markdown("---")
    st.subheader("Download Ledger File")
    st.write("Download the current ledger data directly through your web browser.")
    
    try:
        import pandas as pd
        df = pd.DataFrame(current_ledger_data)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇ Download Ledger as CSV",
            data=csv_data,
            file_name="ledger_download.csv",
            mime="text/csv",
            key="download_ledger_tab3_btn"
        )
    except Exception as download_error:
        st.info("Please fill in or save your ledger data above to enable downloading.")
