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







