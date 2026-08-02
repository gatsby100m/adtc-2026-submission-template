%%writefile app.py
import os
import re
import datetime
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Try importing local AI and vector libraries
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

MODEL_DIR = "models"
MODEL_NAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

@st.cache_resource
def initialize_offline_cores():
    llm_instance = None
    bi_encoder = None
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if LLAMA_AVAILABLE:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading Qwen2.5-0.5B-Instruct weights..."):
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
                llm_instance = Llama(model_path=MODEL_PATH, n_ctx=1024, n_gpu_layers=-1)
            except Exception:
                llm_instance = None

    if TRANSFORMERS_AVAILABLE:
        with st.spinner("Caching Semantic RAG Vector Map vectors..."):
            try:
                bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                bi_encoder = None
                
    return llm_instance, bi_encoder

llm, encoder = initialize_offline_cores()

FARM_KNOWLEDGE_BASE = [
    "Maize Fertilizer Schedule: The first fertilizer application for maize should happen exactly 21 days after planting using NPK 15-15-15 compound fertilizer to develop roots. The second application must occur 42 days after planting using Urea to provide a high nitrogen boost for stalk growth.",
    "Cassava Leaf Spot Management: Cercospora Leaf Spot causes brown or dark spots on cassava leaves. This fungal infection thrives in humid conditions. Action: Ensure wide plant row spacing for better air ventilation, remove lower infected foliage, and apply copper-based fungicide if the outbreak is severe.",
    "Maize Stem Borer Pest Control: Maize Stem Borers are insects that tunnel holes into maize stalks, leading to withered or dried leaves in the center funnel. Action: Check stalks for small entry holes. Prepare and apply a natural neem extract solution directly into the top leaf funnel to kill larvae safely.",
    "General Soil and Watering Advice: Always monitor soil moisture before watering crops. Overwatering leads to waterlogged soil, suffocating plant roots and causing leaves to turn yellow or develop fungal spots. Keep farming plots cleared of competitive weeds.",
    "TakidabakidoriyaakasarHausa: Cutartabaganyenmasara(CMD)yanakawobakikodoriyaaganye. Matakimafikyaushinecireshukandayarubedawuridonhanayaduwa,kumaayiamfanida ingantaccenirinshukamaijurecututtuka."
]

if encoder is not None:
    db_embeddings = encoder.encode(FARM_KNOWLEDGE_BASE, convert_to_tensor=True)
else:
    db_embeddings = None

CULTURAL_PROVERBS = [
    "Yoruba: Bí énìyàn bá șe gbingbin, béè ni yóò șe kórè. (As we sow, so shall we reap.)",
    "Hausa: Mai hakuri yukan dafa dutse har ya sha romonsa. (The patient farmer cooks a stone and drinks its soup.)",
    "Swahili: Mvumilivu hula mbivu. (A patient person eats ripe fruit.)",
    "Igbo: Onye gbam bona ubi, owu wei he ubi ga-asacha anya mmiri ya. (He who labors in the field will have his tears wiped by the harvest.)"
]

if "revenue" not in st.session_state: st.session_state.revenue = 0.0
if "labour_cost" not in st.session_state: st.session_state.labour_cost = 0.0
if "fertilizer_cost" not in st.session_state: st.session_state.fertilizer_cost = 0.0
if "equipment_cost" not in st.session_state: st.session_state.equipment_cost = 0.0
if "other_expenses" not in st.session_state: st.session_state.other_expenses = 0.0
if "input_counter" not in st.session_state: st.session_state.input_counter = 0

LANG_DICT = {
    "English": {
        "title": "Offline Smart Farm Assistant",
        "subtitle": "Voice-First Agricultural Advisor & Ledger (Zero-Data Mode)",
        "diagnose_tab": "AI Advisor", "calendar_tab": "Timeline Calculator", "finance_tab": "Financial Ledger",
        "text_input_label": "Describe crop symptoms:", "submit_btn": "Ask Assistant",
        "crop_select": "Select Your Main Crop:", "date_input": "Planting Date:", "calc_btn": "Generate Farming Timeline",
        "ledger_input": "Transaction (e.g., 'I sold maize for 45000 Naira'):", "log_btn": "Log Transaction",
        "export_btn": "Save Local Text Report to Desktop", "proverb_title": "Traditional Wisdom"
    },
    "Hausa": {
        "title": "Mataimakin Manomi na Offline",
        "subtitle": "Shirin Bada Shawara da Kula da Kudi Ba tare da Internet ba",
        "diagnose_tab": "AI Advisor", "calendar_tab": "Tsarin Shuka", "finance_tab": "Littafin Kudi",
        "text_input_label": "Kwatanta matsalar amfanin gona:", "submit_btn": "Tambayi Mataimaki",
        "crop_select": "Zabi Irin Shukan Ku:", "date_input": "Ranar Shuka:", "calc_btn": "Lissafi Lokutan Aiki",
        "ledger_input": "Bayanin Kudi (Misali: 'Na sayar da masara akan Naira 45000'):", "log_btn": "Yi Rikodin Kudi",
        "export_btn": "Ajiye Rahoto a Desktop", "proverb_title": "Kararin Magana"
    }
}

def run_ai_advisory(user_input, lang):
    cultural_closing = "\n\n*May your barns overflow this season! Mandani na gari!*" if lang == "Hausa" else "\n\n*May your harvest be heavy and rewarding!*"
    matched_fact = "Advise general monitoring, checking soil moisture, clearing competitive weeds, and maintaining row spacing layout protocols."
    
    if encoder is not None and db_embeddings is not None:
        try:
            query_embedding = encoder.encode(user_input, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, db_embeddings)
            best_match_idx = int(np.argmax(cos_scores.cpu().numpy()))
            matched_fact = FARM_KNOWLEDGE_BASE[best_match_idx]
        except Exception:
            pass
            
    if (not LLAMA_AVAILABLE) or (llm is None):
        return f"**Offline Semantic Match:** {matched_fact}\n\n*(Note: Running in high-performance lookup fallback mode).*\n{cultural_closing}"
        
    try:
        if lang == "Hausa":
            system_instruction = (
                "You are an expert African agricultural advisor. "
                "CRITICAL: Use the provided Factsheet Context to answer the user's question accurately. "
                "TRANSLATION RULE: You must translate your final answer and write it ONLY in the Hausa language! "
                "Do NOT write in English. Do NOT write Chinese. Write ONLY in clear Hausa text."
            )
        else:
            system_instruction = (
                "You are an expert African agricultural advisor. "
                "CRITICAL: Use the provided Factsheet Context to answer the user's question accurately. "
                "Elaborate on the details to sound friendly and encouraging, but your facts MUST stay completely anchored to the factsheet context. "
                "Do NOT invent unrelated facts, and write ONLY in clear English text without Chinese characters."
            )
            
        prompt = (
            f"<|im_start|>system\n{system_instruction}\nFactsheet Context: {matched_fact}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
            
        response = llm(
            prompt,
            max_tokens=250,
            temperature=0.0,
            top_p=0.1,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>", "User:", "System:", "Tambaya:"]
        )
        
        ai_response = response['choices'][0]['text'].strip()
        ai_response = re.sub(r'[\u4e00-\u9fff]+', '', ai_response)
        
        if len(ai_response) > 3:
            return f"{ai_response}{cultural_closing}"
        else:
            return f"**Farming Truth Block:** {matched_fact}{cultural_closing}"
    except Exception as e:
        return f"**Offline Semantic Fallback:** {matched_fact}{cultural_closing}"

def calculate_crop_timeline(crop, start_date):
    if crop == "Maize":
        fert1 = start_date + datetime.timedelta(days=21)
        fert2 = start_date + datetime.timedelta(days=42)
        harvest_start = start_date + datetime.timedelta(days=90)
        harvest_end = start_date + datetime.timedelta(days=120)
        return f"Maize Timeline:\n- 1st Fertilizer (NPK): {fert1}\n- 2nd Fertilizer (Urea): {fert2}\n- Harvest Window: {harvest_start} to {harvest_end}"
    else:
        fert1 = start_date + datetime.timedelta(days=30)
        fert2 = start_date + datetime.timedelta(days=90)
        harvest_start = start_date + datetime.timedelta(days=270)
        return f"Cassava Timeline:\n- Maintenance Window: {fert1}\n- Root Bulking Boost: {fert2}\n- Harvest Begins: {harvest_start}"

def parse_financial_statement(stmt):
    stmt_lower = stmt.lower()
    sales_keywords = ["sold", "sales", "nasayar", "sayar"]
    labour_keywords = ["labour", "worker", "lebur", "kodago"]
    fert_keywords = ["fertilizer", "taki", "chemical", "fesa"]
    equip_keywords = ["equipment", "tractor", "hayar", "kudin aiki"]
    
    try:
        amount = float(re.findall(r'\d+', stmt)[0])
    except IndexError:
        return "Could not find a numerical value in your transaction note."
        
