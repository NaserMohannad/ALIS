import streamlit as st
import requests
import weaviate
from transformers import AutoTokenizer, pipeline
from weaviate.classes.init import Auth
from weaviate.classes.query import Rerank
import torch


def connect_to_db():
    headers = {
        "X-Cohere-Api-Key": st.secrets["COHERE_API_KEY"]
    }

    weaviate_url = st.secrets["veaviat_rest"]
    weaviate_api_key = st.secrets["weaviat_api_key"]

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        headers=headers
    )
    return client


@st.cache_resource
def load_router():
    model_checkpoint = "EN3IMI/RouterAraBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    classifier = pipeline("sentiment-analysis", model=model_checkpoint)
    return classifier

classifier = load_router()


def search_for_faq(user_query, client):
    collection = client.collections.use("FAQ")
    response = collection.query.hybrid(
        query=user_query,
        limit=10,
        alpha=0.40,
        rerank=Rerank(prop="content", query=user_query),
    )

    queries = []
    for idx, obj in enumerate(response.objects[:2]):
        message_to_router = f"{user_query} [SEP] {obj.properties['content']}"
        queries.append(message_to_router)

    return queries, response.objects[:3]

def search_for_laws(user_query, client):
    collection = client.collections.use("Laws")
    response = collection.query.hybrid(
        query=user_query,
        limit=10,
        alpha=0.40,
        rerank=Rerank(prop="text", query=user_query),
    )
    return response.objects[:3]


def router_decision(queries):
    results = classifier(queries)
    labels = [res['label'] for res in results]
    numeric_labels = [1 if label == 'LABEL_1' else 0 for label in labels]
    return any(numeric_labels)


def llm_response(query, docs, mode="faq"):
    chunks = []
    for i in docs:
        chunks.append(i.properties["content"] if mode=="faq" else i.properties["text"])

    API_KEY = st.secrets["perplixty_api"]
    ENDPOINT = "https://api.perplexity.ai/chat/completions"

    system_prompt = """
    You are an intelligent assistant specialized in Jordanian Land, Survey, and Legislation.
    Answer only from the provided context. If no relevant answer exists, reply in Arabic: "لا أعلم الجواب".
    Be accurate, concise, and prioritize Arabic answers.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Context:\n" + "\n".join(chunks) + "\n\nQuestion:\n" + query},
    ]

    data = {
        "model": "sonar-pro",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.5,
    }

    resp = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=data,
    )
    return resp.json()['choices'][0]['message']['content']


def IntelligentRAGSystem(query, client):
    faq_queries, faq_docs = search_for_faq(query, client)
    decision = router_decision(faq_queries)
    if decision:
        return llm_response(query, faq_docs, mode="faq")
    else:
        law_docs = search_for_laws(query, client)
        return llm_response(query, law_docs, mode="law")



import streamlit as st
import requests
import weaviate
from transformers import AutoTokenizer, pipeline
from weaviate.classes.init import Auth
from weaviate.classes.query import Rerank
import torch
import time
from PIL import Image


def connect_to_db():
    headers = {
        "X-Cohere-Api-Key": st.secrets["COHERE_API_KEY"]
    }

    weaviate_url = st.secrets["veaviat_rest"]
    weaviate_api_key = st.secrets["weaviat_api_key"]

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        headers=headers
    )
    return client


@st.cache_resource
def load_router():
    model_checkpoint = "EN3IMI/RouterAraBERT"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    classifier = pipeline("sentiment-analysis", model=model_checkpoint)
    return classifier

classifier = load_router()


def search_for_faq(user_query, client):
    collection = client.collections.use("FAQ")
    response = collection.query.hybrid(
        query=user_query,
        limit=10,
        alpha=0.40,
        rerank=Rerank(prop="content", query=user_query),
    )

    queries = []
    for idx, obj in enumerate(response.objects[:2]):
        message_to_router = f"{user_query} [SEP] {obj.properties['content']}"
        queries.append(message_to_router)

    return queries, response.objects[:3]

def search_for_laws(user_query, client):
    collection = client.collections.use("Laws")
    response = collection.query.hybrid(
        query=user_query,
        limit=10,
        alpha=0.40,
        rerank=Rerank(prop="text", query=user_query),
    )
    return response.objects[:3]


def router_decision(queries):
    results = classifier(queries)
    labels = [res['label'] for res in results]
    numeric_labels = [1 if label == 'LABEL_1' else 0 for label in labels]
    return any(numeric_labels)


def llm_response(query, docs, mode="faq"):
    chunks = []
    for i in docs:
        chunks.append(i.properties["content"] if mode=="faq" else i.properties["text"])

    API_KEY = st.secrets["perplixty_api"]
    ENDPOINT = "https://api.perplexity.ai/chat/completions"

    system_prompt = """
    أنت مساعد ذكي متخصص في مسائل الأراضي والمساحة في الأردن.
    أجب فقط من السياق المقدم. إذا لم تجد إجابة ذات صلة، أجب بالعربية: "لا أعلم الجواب".
    كن دقيقاً ومختصراً وركز على الإجابات العربية.
    لا تقدم أي روابط أو مراجع خارجية في إجابتك.
    تخصص فقط في المساعدة بشأن الأراضي والمساحة وليس الاستشارة القانونية العامة.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Context:\n" + "\n".join(chunks) + "\n\nQuestion:\n" + query},
    ]

    data = {
        "model": "sonar-pro",
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.5,
    }

    resp = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json=data,
    )
    return resp.json()['choices'][0]['message']['content']


def IntelligentRAGSystem(query, client):
    faq_queries, faq_docs = search_for_faq(query, client)
    decision = router_decision(faq_queries)
    if decision:
        return llm_response(query, faq_docs, mode="faq")
    else:
        law_docs = search_for_laws(query, client)
        return llm_response(query, law_docs, mode="law")


# ===== إعداد الصفحة =====
st.set_page_config(
    page_title="ALIS - مساعد الأراضي والمساحة الأردني", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== CSS محسن للواجهة الحديثة مع ثيم أحمر وأسود =====
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700&display=swap');
        
        * {
            direction: rtl;
            text-align: right;
        }
        
        .main {
            direction: rtl;
            text-align: right;
            padding: 0 !important;
        }
        
        .stApp {
            background: #0a0a0a;
            direction: rtl;
            color: #ffffff;
        }
        
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(135deg, #8b0000, #000000);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin-bottom: 3rem;
            box-shadow: 0 20px 40px rgba(139, 0, 0, 0.3);
            border: 1px solid #8b0000;
        }
        
        .main-title {
            font-size: 3rem !important;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
            font-family: 'Cairo', sans-serif;
            color: #ffffff;
        }
        
        .main-subtitle {
            font-size: 1.3rem;
            opacity: 0.95;
            font-weight: 400;
            color: #e0e0e0;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Welcome Card */
        .welcome-card {
            background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
            padding: 3rem 2rem;
            border-radius: 25px;
            text-align: center;
            margin: 3rem 0;
            box-shadow: 0 10px 30px rgba(139, 0, 0, 0.2);
            border: 1px solid #8b0000;
            color: #ffffff;
        }
        
        .welcome-card h3 {
            color: #ffffff;
            font-size: 2rem;
            margin-bottom: 1rem;
            font-family: 'Cairo', sans-serif;
            font-weight: 600;
        }
        
        .welcome-card p {
            color: #cccccc;
            font-size: 1.2rem;
            line-height: 1.8;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Input Section */
        .input-section {
            background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
            padding: 3rem;
            border-radius: 25px;
            box-shadow: 0 15px 35px rgba(139, 0, 0, 0.2);
            margin: 3rem 0;
            border: 1px solid #8b0000;
            color: #ffffff;
        }
        
        .section-title {
            font-size: 2rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 2rem;
            text-align: center;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Answer Box */
        .answer-container {
            background: linear-gradient(135deg, #2a1a1a, #3a2a2a);
            padding: 3rem;
            border-radius: 25px;
            border-right: 6px solid #8b0000;
            margin: 3rem 0;
            box-shadow: 0 15px 35px rgba(139, 0, 0, 0.3);
            color: #ffffff;
        }
        
        .answer-text {
            font-size: 1.2rem;
            line-height: 2.2;
            color: #ffffff;
            font-weight: 400;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #8b0000, #600000);
            color: white;
            border: none;
            border-radius: 30px;
            padding: 1rem 3rem;
            font-size: 1.2rem;
            font-weight: 600;
            box-shadow: 0 8px 20px rgba(139, 0, 0, 0.4);
            transition: all 0.3s ease;
            width: 100%;
            font-family: 'Cairo', sans-serif;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #600000, #450000);
            transform: translateY(-3px);
            box-shadow: 0 12px 25px rgba(139, 0, 0, 0.5);
        }
        
        /* Text Area */
        .stTextArea textarea {
            border-radius: 20px;
            border: 2px solid #8b0000;
            padding: 1.5rem;
            font-size: 1.2rem;
            font-family: 'Cairo', sans-serif;
            direction: rtl;
            text-align: right;
            background: #1a1a1a;
            color: #ffffff;
            transition: all 0.3s ease;
        }
        
        .stTextArea textarea:focus {
            border-color: #ff0000;
            box-shadow: 0 0 0 4px rgba(139, 0, 0, 0.3);
            outline: none;
            background: #2a2a2a;
        }
        
        /* Features Grid */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
            padding: 2.5rem 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(139, 0, 0, 0.2);
            transition: all 0.3s ease;
            border: 1px solid #8b0000;
            color: #ffffff;
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(139, 0, 0, 0.4);
        }
        
        .feature-card h4 {
            color: #ffffff;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            font-family: 'Cairo', sans-serif;
            font-weight: 600;
        }
        
        .feature-card p {
            color: #cccccc;
            font-size: 1.1rem;
            line-height: 1.7;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Quick Buttons */
        .quick-button {
            background: linear-gradient(135deg, #2a2a2a, #3a3a3a);
            border: 2px solid #8b0000;
            color: #ffffff;
            border-radius: 15px;
            padding: 1rem 1.5rem;
            font-weight: 500;
            font-family: 'Cairo', sans-serif;
            transition: all 0.3s ease;
            text-align: center;
            cursor: pointer;
            width: 100%;
            margin-bottom: 1rem;
        }
        
        .quick-button:hover {
            background: linear-gradient(135deg, #8b0000, #600000);
            border-color: #ff0000;
            color: #ffffff;
            transform: translateY(-2px);
        }
        
        /* Success Message */
        .success-message {
            background: linear-gradient(135deg, #2a1a1a, #3a2a2a);
            color: #ff6b6b;
            padding: 1.5rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            border-right: 5px solid #8b0000;
            font-family: 'Cairo', sans-serif;
            font-weight: 500;
            font-size: 1.1rem;
        }
        
        /* Footer */
        .footer {
            background: linear-gradient(135deg, #000000, #1a1a1a);
            color: white;
            text-align: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-top: 4rem;
            border: 1px solid #8b0000;
        }
        
        .footer h4 {
            font-size: 1.8rem;
            margin-bottom: 1rem;
            font-family: 'Cairo', sans-serif;
            font-weight: 600;
        }
        
        .footer p {
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            opacity: 0.9;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Progress Bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(135deg, #8b0000, #600000);
        }
        
        /* Warning and Error Messages */
        .stAlert {
            border-radius: 15px;
            border: none;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Cairo', sans-serif !important;
            color: #ffffff;
        }
        
        /* Regular text */
        p, div, span {
            font-family: 'Cairo', sans-serif;
            color: #ffffff;
        }
        
        /* Markdown content */
        .stMarkdown {
            font-family: 'Cairo', sans-serif;
            color: #ffffff;
        }
        
        /* Container spacing */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Dark mode toggle */
        .dark-mode-toggle {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1000;
        }
    </style>
""", unsafe_allow_html=True)

# ===== تهيئة حالة الجلسة =====
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# ===== وظيفة تحديث الاستعلام =====
def update_query(new_query):
    st.session_state.current_query = new_query

# ===== الصفحة الرئيسية =====
# Header
st.markdown("""
    <div class="main-header">
        <div class="main-title">ALIS - مساعد الأراضي والمساحة الأردني</div>
        <div class="main-subtitle">نظام ذكي متقدم متخصص في المساعدة بشؤون الأراضي والمساحة في الأردن</div>
    </div>
""", unsafe_allow_html=True)

# Welcome Section
st.markdown("""
    <div class="welcome-card">
        <h3>مرحباً بك في ALIS</h3>
        <p>مساعدك الذكي المتخصص في مسائل الأراضي والمساحة في الأردن. نحن متخصصون فقط في المساعدة بالأراضي والمساحة وليس الاستشارة القانونية أو التشريعات بشكل عام. احصل على إجابات دقيقة وموثوقة لجميع استفساراتك مع ضمان الخصوصية والأمان التام.</p>
    </div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("## لماذا ALIS؟")
features_html = """
    <div class="features-grid">
        <div class="feature-card">
            <h4>سرعة استثنائية</h4>
            <p>إجابات فورية ودقيقة على جميع استفساراتك حول الأراضي والمساحة خلال ثوانٍ معدودة</p>
        </div>
        <div class="feature-card">
            <h4>دقة متناهية</h4>
            <p>معلومات موثقة ومحدثة باستمرار من المصادر المتخصصة في الأراضي والمساحة</p>
        </div>
        <div class="feature-card">
            <h4>أمان وخصوصية</h4>
            <p>حماية كاملة لبياناتك واستفساراتك مع ضمان السرية التامة</p>
        </div>
        <div class="feature-card">
            <h4>تخصص مركز</h4>
            <p>تركيز كامل على مسائل الأراضي والمساحة في الأردن فقط</p>
        </div>
    </div>
"""
st.markdown(features_html, unsafe_allow_html=True)

# Input Section
st.markdown("""
    <div class="input-section">
        <div class="section-title">
            اطرح سؤالك حول الأراضي والمساحة واحصل على الإجابة الشافية
        </div>
""", unsafe_allow_html=True)

# Quick Questions
st.markdown("### أسئلة شائعة - اختر أحد الأسئلة التالية")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("رسوم التسجيل", key="q1", use_container_width=True):
        update_query("ما هي رسوم تسجيل الأراضي في الأردن؟")

with col2:
    if st.button("إجراءات النقل", key="q2", use_container_width=True):
        update_query("ما هي إجراءات نقل ملكية الأراضي؟")

with col3:
    if st.button("أنواع الأراضي", key="q3", use_container_width=True):
        update_query("ما هي أنواع الأراضي في الأردن؟")

with col4:
    if st.button("المساحة والحدود", key="q4", use_container_width=True):
        update_query("كيف يتم تحديد مساحة وحدود قطعة الأرض؟")

# Text area with current query
st.markdown("### اكتب سؤالك هنا")
query = st.text_area(
    "",
    height=120,
    placeholder="مثال: ما هي الإجراءات اللازمة لتسجيل قطعة أرض في الأردن؟",
    value=st.session_state.current_query,
    key="main_query"
)

st.markdown("</div>", unsafe_allow_html=True)

# Send Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    send = st.button("احصل على الإجابة الشافية", type="primary", use_container_width=True)

# Processing
if send:
    if not query.strip():
        st.warning("الرجاء إدخال سؤال قبل الإرسال")
    else:
        # Loading animation
        with st.spinner("جاري البحث في قاعدة البيانات المتخصصة..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            try:
                # ربط الباك إند بالفرونت إند
                client = connect_to_db()
                answer = IntelligentRAGSystem(query, client)
                client.close()
                
                # رسالة النجاح المحدثة (بدون علامة تعجب)
                st.markdown('<div class="success-message">تم الحصول على الإجابة</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="answer-container">
                        <div class="answer-text">{answer}</div>
                    </div>
                """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"عذراً، حدث خطأ أثناء معالجة السؤال: {e}")

# Footer
st.markdown("""
    <div class="footer">
        <h4>ALIS - مساعد الأراضي والمساحة الأردني</h4>
        <p>تم تطويره بواسطة فريق ALIS المتخصص | جميع الحقوق محفوظة © 2024</p>
        <p>info@alis.jo | +962-6-1234567 | www.alis.jo</p>
        <p>نظام ذكي متخصص في الأراضي والمساحة - مرخص من وزارة الاقتصاد الرقمي والريادة</p>
    </div>
""", unsafe_allow_html=True)









