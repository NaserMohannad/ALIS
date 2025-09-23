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


'''import streamlit as st
from PIL import Image
import time

# ===== إعداد الصفحة =====
st.set_page_config(
    page_title="ALIS - مساعد الأراضي والتشريعات الأردني 🇯🇴", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CSS محسن للواجهة الحديثة =====
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');
        
        * {
            direction: rtl;
            text-align: right;
        }
        
        .main {
            direction: rtl;
            text-align: right;
        }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Cairo', sans-serif;
            direction: rtl;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            direction: rtl;
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .main-title {
            font-size: 2.5rem !important;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        /* Welcome Card */
        .welcome-card {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            padding: 2rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            border: 1px solid #e9ecef;
        }
        
        .welcome-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        /* Input Section */
        .input-section {
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            margin: 2rem 0;
            border: 1px solid #e9ecef;
        }
        
        .section-title {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        /* Answer Box */
        .answer-container {
            background: linear-gradient(135deg, #e8f5e8, #f0f8f0);
            padding: 2rem;
            border-radius: 20px;
            border-right: 5px solid #27ae60;
            margin: 2rem 0;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        }
        
        .answer-text {
            font-size: 1.1rem;
            line-height: 2;
            color: #2c3e50;
            font-weight: 400;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: 600;
            box-shadow: 0 5px 15px rgba(39, 174, 96, 0.3);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #219a52, #27ae60);
            transform: translateY(-2px);
            box-shadow: 0 7px 20px rgba(39, 174, 96, 0.4);
        }
        
        /* Text Area */
        .stTextArea textarea {
            border-radius: 15px;
            border: 2px solid #e9ecef;
            padding: 1rem;
            font-size: 1.1rem;
            font-family: 'Cairo', sans-serif;
            direction: rtl;
            text-align: right;
        }
        
        .stTextArea textarea:focus {
            border-color: #27ae60;
            box-shadow: 0 0 0 3px rgba(39, 174, 96, 0.1);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: linear-gradient(180deg, #2c3e50, #34495e);
        }
        
        /* Stats Cards */
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            margin: 1rem 0;
            border-top: 4px solid #3498db;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: #2c3e50;
        }
        
        .stat-label {
            color: #7f8c8d;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }
        
        /* Footer */
        .footer {
            background: linear-gradient(135deg, #2c3e50, #34495e);
            color: white;
            text-align: center;
            padding: 2rem;
            border-radius: 15px;
            margin-top: 3rem;
        }
        
        /* Loading Animation */
        .loading-container {
            text-align: center;
            padding: 2rem;
        }
        
        /* Features Grid */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }
        
        .feature-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        }
        
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        
        /* Alerts */
        .stAlert {
            border-radius: 15px;
            border: none;
        }
        
        /* Success Message */
        .success-message {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            color: #155724;
            padding: 1rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            border-right: 4px solid #28a745;
        }
        
        /* Metrics */
        .stMetric {
            background: white;
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# ===== Sidebar =====
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem; color: white;">
            <h2 style="color: white;">🤖 ALIS</h2>
            <p style="color: #bdc3c7;">مساعدك القانوني الذكي</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 إحصائيات سريعة")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("الاستفسارات اليوم", "127", "12")
    with col2:
        st.metric("معدل الدقة", "95%", "2%")
    
    st.markdown("---")
    
    st.markdown("### 🔧 إعدادات")
    response_length = st.selectbox(
        "طول الإجابة",
        ["قصيرة", "متوسطة", "مفصلة"],
        index=1
    )
    
    include_sources = st.checkbox("إظهار المصادر", True)
    
    st.markdown("---")
    
    st.markdown("### 📞 تواصل معنا")
    st.info("📧 support@alis.jo\n📱 +962-6-1234567")

# ===== الصفحة الرئيسية =====
# Header
st.markdown("""
    <div class="main-header">
        <div class="main-title">🇯🇴 ALIS - مساعد الأراضي والتشريعات الأردني</div>
        <div class="main-subtitle">نظام ذكي متقدم للإجابة على استفساراتكم القانونية والعقارية</div>
    </div>
""", unsafe_allow_html=True)

# Welcome Section
st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">🌟</div>
        <h3>مرحباً بك في ALIS</h3>
        <p>مساعدك الذكي المتخصص في التشريعات الأردنية وقوانين الأراضي. احصل على إجابات دقيقة وموثوقة لجميع استفساراتك القانونية.</p>
    </div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("## ✨ مميزاتنا")
features_html = """
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h4>سرعة فائقة</h4>
            <p>إجابات فورية على استفساراتك</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h4>دقة عالية</h4>
            <p>معلومات موثوقة ومحدثة</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <h4>آمن ومحمي</h4>
            <p>حماية كاملة لبياناتك</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <h4>شامل ومتكامل</h4>
            <p>تغطية جميع القوانين الأردنية</p>
        </div>
    </div>
"""
st.markdown(features_html, unsafe_allow_html=True)

# Input Section
st.markdown("""
    <div class="input-section">
        <div class="section-title">
            ✍️ اطرح سؤالك هنا
        </div>
""", unsafe_allow_html=True)

# Tabs for different question types
tab1, tab2, tab3 = st.tabs(["💼 استفسارات عامة", "🏘️ قوانين الأراضي", "⚖️ التشريعات"])

with tab1:
    st.markdown("### اطرح أي سؤال قانوني")
    query = st.text_area(
        "",
        height=120,
        placeholder="مثال: ما هي الإجراءات اللازمة لتسجيل قطعة أرض في الأردن؟",
        key="general_query"
    )

with tab2:
    st.markdown("### أسئلة متخصصة في الأراضي")
    query = st.text_area(
        "",
        height=120,
        placeholder="مثال: ما هي رسوم تحويل ملكية قطعة أرض؟",
        key="land_query"
    )

with tab3:
    st.markdown("### استفسارات التشريعات")
    query = st.text_area(
        "",
        height=120,
        placeholder="مثال: ما هي القوانين المتعلقة بالإرث في الأردن؟",
        key="law_query"
    )

st.markdown("</div>", unsafe_allow_html=True)

# Quick Questions
st.markdown("### 🚀 أسئلة سريعة")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("رسوم التسجيل"):
        query = "ما هي رسوم تسجيل الأراضي في الأردن؟"

with col2:
    if st.button("شروط البيع"):
        query = "ما هي شروط بيع الأراضي؟"

with col3:
    if st.button("الإرث"):
        query = "كيف يتم توزيع الإرث حسب القانون الأردني؟"

with col4:
    if st.button("التأمين"):
        query = "ما هو التأمين العقاري المطلوب؟"

# Send Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    send = st.button("🚀 احصل على الإجابة", type="primary")

# Processing
if send:
    if not query.strip():
        st.warning("⚠️ الرجاء إدخال سؤال قبل الإرسال")
    else:
        # Loading animation
        with st.spinner("🔍 جاري البحث في قاعدة البيانات القانونية..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            try:
                # Simulated response - replace with your actual function
                # client = connect_to_db()
                # answer = IntelligentRAGSystem(query, client)
                # client.close()
                
                # Simulated answer for demonstration
                answer = """
                بناءً على القوانين الأردنية الحالية، إليك الإجابة التفصيلية:

                **الإجراءات المطلوبة:**
                1. تحضير الوثائق المطلوبة (سند الملكية، هوية مدنية)
                2. دفع الرسوم المقررة في دائرة الأراضي
                3. الحصول على موافقة البلدية إذا لزم الأمر
                4. إتمام عملية التسجيل النهائي

                **الرسوم المطلوبة:**
                - رسم التسجيل: 0.5% من قيمة العقار
                - رسم الطابع: 15 دينار أردني
                - رسوم إضافية حسب المنطقة

                **ملاحظات هامة:**
                - يجب أن تكون جميع الوثائق سارية المفعول
                - قد تختلف الإجراءات حسب نوع الأرض ومنطقتها
                """
                
                st.markdown('<div class="success-message">✅ تم الحصول على الإجابة بنجاح!</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="answer-container">
                        <div class="answer-text">{answer}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show sources if enabled
                if include_sources:
                    st.markdown("### 📚 المصادر")
                    with st.expander("اضغط لعرض المصادر"):
                        st.markdown("""
                        - قانون الأراضي الأردني رقم 40 لسنة 1952
                        - تعليمات دائرة الأراضي والمساحة
                        - النشرات الرسمية لوزارة العدل
                        """)
                
                # Rating section
                st.markdown("### 📝 تقييم الإجابة")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.button("⭐")
                with col2:
                    st.button("⭐⭐")
                with col3:
                    st.button("⭐⭐⭐")
                with col4:
                    st.button("⭐⭐⭐⭐")
                with col5:
                    st.button("⭐⭐⭐⭐⭐")
                    
            except Exception as e:
                st.error(f"❌ عذراً، حدث خطأ أثناء معالجة السؤال: {e}")

# Footer
st.markdown("""
    <div class="footer">
        <h4>🇯🇴 ALIS - مساعد الأراضي والتشريعات الأردني</h4>
        <p>تم تطويره بواسطة فريق ALIS | جميع الحقوق محفوظة © 2024</p>
        <p>📧 info@alis.jo | 📱 +962-6-1234567 | 🌐 www.alis.jo</p>
    </div>
""", unsafe_allow_html=True)
'''
import streamlit as st
from PIL import Image

# ===== إعداد الصفحة =====
st.set_page_config(
    page_title="ALIS - Jordan RAG Assistant 🇯🇴", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚖️"
)

# ===== CSS مخصص لتجميل الواجهة =====
st.markdown("""
    <style>
        /* تنسيق عام */
        * {
            direction: rtl;
            text-align: right;
        }
        
        body {
            background-color: #f8fafc;
            font-family: "Segoe UI", "Tahoma", "Arial", sans-serif;
        }
        
        /* رأس الصفحة */
        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 2rem 1rem;
            border-radius: 0 0 20px 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .title-text {
            font-size: 2.2rem !important;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .subtitle-text {
            font-size: 1.1rem;
            opacity: 0.9;
            margin-bottom: 0;
        }
        
        /* بطاقة الترحيب */
        .welcome-card {
            background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid #e1e8f0;
        }
        
        /* حاوية الإدخال */
        .input-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            border: 1px solid #e1e8f0;
        }
        
        /* مربع الإجابة */
        .answer-card {
            background: linear-gradient(135deg, #f0f9ff 0%, #e6f3ff 100%);
            padding: 1.8rem;
            border-radius: 12px;
            line-height: 1.8;
            font-size: 1.1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border-right: 4px solid #3b82f6;
            margin-top: 1.5rem;
        }
        
        /* زر الإرسال */
        .stButton button {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border-radius: 10px;
            padding: 12px 30px;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(16, 185, 129, 0.3);
        }
        
        /* الشريط الجانبي */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        }
        
        /* علامات التبويب */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f5f9;
            border-radius: 8px 8px 0 0;
            padding: 12px 24px;
            font-weight: 600;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #3b82f6;
            color: white;
        }
        
        /* تذييل الصفحة */
        .footer {
            text-align: center;
            color: #64748b;
            margin-top: 3rem;
            padding: 1.5rem;
            font-size: 0.9rem;
            border-top: 1px solid #e2e8f0;
        }
        
        /* مؤشر التحميل */
        .stSpinner > div {
            border: 3px solid #f3f4f6;
            border-radius: 50%;
            border-top: 3px solid #3b82f6;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* تحسينات للنصوص */
        h1, h2, h3 {
            color: #1e293b;
        }
        
        /* تحسينات للرسائل */
        .stAlert {
            border-radius: 10px;
        }
        
        /* أيقونات */
        .icon {
            font-size: 1.5rem;
            margin-left: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ===== الشريط الجانبي =====
with st.sidebar:
    st.markdown("<div style='text-align:center; margin-bottom:2rem;'>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Scale_of_justice.png", width=80)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ الإعدادات")
    
    # خيارات للنظام
    model_type = st.selectbox(
        "نوع النموذج:",
        ["النموذج الأساسي", "نموذج متقدم", "نموذج سريع"]
    )
    
    detail_level = st.selectbox(
        "مستوى التفاصيل:",
        ["مختصر", "متوسط", "مفصل"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 إحصائيات النظام")
    
    # إحصائيات وهمية
    col1, col2 = st.columns(2)
    with col1:
        st.metric("الاستعلامات اليومية", "24")
        st.metric("معدل الدقة", "94%")
    
    with col2:
        st.metric("الاستعلامات الشهرية", "720")
        st.metric("وقت الاستجابة", "1.2s")
    
    st.markdown("---")
    st.markdown("### ℹ️ حول النظام")
    st.info("""
    ALIS هو مساعد ذكي يستخدم تقنية RAG 
    للإجابة على الأسئلة المتعلقة بالأراضي 
    والتشريعات الأردنية.
    """)

# ===== رأس الصفحة =====
st.markdown("""
    <div class="header">
        <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
                <div class="title-text">🤖 ALIS - Jordan RAG Assistant</div>
                <div class="subtitle-text">نظام ذكي للإجابة على الأسئلة المتعلقة بالأراضي والتشريعات الأردنية</div>
            </div>
            <div style="font-size:2rem;">🇯🇴</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ===== علامات التبويب =====
tab1, tab2, tab3 = st.tabs(["🏠 الرئيسية", "📚 الأسئلة الشائعة", "ℹ️ معلومات"])

with tab1:
    # ===== بطاقة الترحيب =====
    st.markdown("""
    <div class="welcome-card">
        <h2 style="margin-top:0;">👋 أهلاً وسهلاً بك في <span style="color:#3b82f6;">ALIS</span></h2>
        <p style="font-size:1.2rem; margin-bottom:1.5rem;">مساعدك الذكي للإجابة على أسئلتك حول التشريعات والأراضي الأردنية</p>
        <div style="display:flex; justify-content:center; gap:1rem; flex-wrap:wrap;">
            <span style="background:#e0f2fe; color:#0369a1; padding:6px 12px; border-radius:20px;">⚖️ تشريعات</span>
            <span style="background:#dcfce7; color#166534; padding:6px 12px; border-radius:20px;">🏞️ أراضي</span>
            <span style="background:#fef3c7; color#92400e; padding:6px 12px; border-radius:20px;">📄 وثائق</span>
            <span style="background:#f3e8ff; color#6b21a8; padding:6px 12px; border-radius:20px;">🔍 بحث</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== حاوية الإدخال =====
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    st.markdown("### ✍️ أدخل سؤالك:")
    
    query = st.text_area(
        " ",
        height=140,
        placeholder="مثال: ما هي رسوم تسجيل قطعة أرض؟ أو ما هي إجراءات نقل ملكية العقار؟",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        send = st.button("🚀 إرسال السؤال", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ===== المعالجة =====
    if send:
        if not query.strip():
            st.warning("⚠️ الرجاء إدخال سؤال قبل الإرسال.")
        else:
            with st.spinner("🔍 جاري البحث عن الإجابة، الرجاء الانتظار..."):
                try:
                    # هذه الدوال تحتاج إلى تنفيذها حسب نظامك
                    # client = connect_to_db()
                    # answer = IntelligentRAGSystem(query, client)
                    # client.close()
                    
                    # نموذج إجابة وهمية للعرض
                    answer = """
                    وفقاً للمادة 25 من قانون الأراضي الأردني، فإن رسوم تسجيل قطعة أرض تُحدد بناءً على:
                    
                    1. مساحة الأرض: 0.5% من قيمة الأرض للمساحات التي تقل عن 1000 متر مربع.
                    2. موقع الأرض: تختلف الرسوم بين المناطق حسب التصنيف.
                    3. نوع الاستخدام: سكني، تجاري، زراعي.
                    
                    يجب تقديم الطلب إلى دائرة الأراضي والمساحة في المنطقة التابعة لها الأرض، 
                    مع المستندات المطلوبة بما في ذلك صك الملكية والهوية الشخصية.
                    
                    لمزيد من التفاصيل، يمكنك زيارة الموقع الرسمي لدائرة الأراضي والمساحة الأردنية.
                    """
                    
                    st.success("✅ تم العثور على الإجابة:")
                    st.markdown(f"<div class='answer-card'>{answer}</div>", unsafe_allow_html=True)
                    
                    # قسم التقييم
                    st.markdown("---")
                    st.markdown("#### 📊 كيف كانت تجربتك مع هذه الإجابة؟")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("👍 مفيدة", use_container_width=True):
                            st.success("شكراً لتقييمك!")
                    with col2:
                        if st.button("👎 غير مفيدة", use_container_width=True):
                            st.info("شكراً للملاحظة، سنعمل على تحسين الإجابات.")
                    with col3:
                        if st.button("📋 نسخ الإجابة", use_container_width=True):
                            st.info("تم نسخ الإجابة إلى الحافظة")
                    
                except Exception as e:
                    st.error(f"حدث خطأ أثناء معالجة سؤالك: {e}")

with tab2:
    st.markdown("### 📚 الأسئلة الشائعة")
    
    # تنظيم الأسئلة الشائعة في أكورديون
    with st.expander("ما هي إجراءات تسجيل قطعة أرض جديدة؟", expanded=False):
        st.markdown("""
        - تقديم طلب التسجيل إلى دائرة الأراضي والمساحة
        - إرفاق المستندات المطلوبة (صك الملكية، الهوية الشخصية)
        - دفع الرسوم المقررة
        - انتظار الفحص الميداني من قبل المهندس المسؤول
        - استلام سند التسجيل النهائي
        """)
    
    with st.expander("ما هي المستندات المطلوبة لنقل ملكية عقار؟"):
        st.markdown("""
        - سند الملكية الأصلي
        - هوية البائع والمشتري
        - عقد البيع الموقع من الطرفين
        - شهادة عدم الممانعة من الدائرة البلدية
        - شهادة عدم وجود ديون بلدية
        """)
    
    with st.expander("كيف يمكنني الاستعلام عن مخططات الأراضي؟"):
        st.markdown("""
        - زيارة موقع دائرة الأراضي والمساحة الإلكتروني
        - استخدام خدمة الاستعلام عن المخططات
        - إدخال رقم القطعة والمحافظة
        - الحصول على المعلومات المطلوبة
        """)
    
    with st.expander("ما هي رسوم التسجيل للعقارات التجارية؟"):
        st.markdown("""
        تختلف رسوم التسجيل للعقارات التجارية حسب:
        - قيمة العقار
        - موقع العقار
        - مساحة العقار
        - نوع النشاط التجاري
        
        بشكل عام، تتراوح بين 1% إلى 3% من قيمة العقار.
        """)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ℹ️ معلومات عن النظام")
        st.info("""
        **ALIS** هو نظام ذكي يستخدم تقنية 
        **الاسترجاع المعزز بالتوليد (RAG)** 
        لتقديم إجابات دقيقة ومحدثة حول:
        
        - التشريعات الأردنية
        - قوانين الأراضي والعقارات
        - الإجراءات القانونية
        - المستندات المطلوبة
        
        يعتمد النظام على أحدث تقنيات الذكاء الاصطناعي
        لضمان دقة المعلومات وسهولة الوصول إليها.
        """)
    
    with col2:
        st.markdown("### 📞 اتصل بنا")
        st.info("""
        للاستفسارات أو المساعدة التقنية:
        
        **البريد الإلكتروني:** support@alis.gov.jo  
        **هاتف:** 065000000  
        **ساعات العمل:** 8:00 ص - 4:00 م  
        **أيام العمل:** الأحد - الخميس
        """)

# ===== تذييل الصفحة =====
st.markdown("---")
st.markdown("""
    <div class="footer">
        تم تطوير النظام بواسطة مشروع <strong>ALIS</strong> - الذكاء الاصطناعي للأراضي والمساحة 🇯🇴<br>
        جميع الحقوق محفوظة © 2023
    </div>
""", unsafe_allow_html=True)







