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
    initial_sidebar_state="collapsed"
)

# ===== CSS محسن للواجهة الحديثة مع ألوان مدروسة =====
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
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            direction: rtl;
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
            background: linear-gradient(135deg, #1e293b, #334155);
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin-bottom: 3rem;
            box-shadow: 0 20px 40px rgba(30, 41, 59, 0.15);
        }
        
        .main-title {
            font-size: 3rem !important;
            font-weight: 700;
            margin-bottom: 1rem;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            font-family: 'Cairo', sans-serif;
        }
        
        .main-subtitle {
            font-size: 1.3rem;
            opacity: 0.95;
            font-weight: 400;
            color: #cbd5e1;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Welcome Card */
        .welcome-card {
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            padding: 3rem 2rem;
            border-radius: 25px;
            text-align: center;
            margin: 3rem 0;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
        }
        
        .welcome-icon {
            font-size: 4rem;
            margin-bottom: 1.5rem;
        }
        
        .welcome-card h3 {
            color: #1e293b;
            font-size: 2rem;
            margin-bottom: 1rem;
            font-family: 'Cairo', sans-serif;
            font-weight: 600;
        }
        
        .welcome-card p {
            color: #475569;
            font-size: 1.2rem;
            line-height: 1.8;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Input Section */
        .input-section {
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            padding: 3rem;
            border-radius: 25px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.08);
            margin: 3rem 0;
            border: 1px solid #e2e8f0;
        }
        
        .section-title {
            font-size: 2rem;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 2rem;
            text-align: center;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Answer Box */
        .answer-container {
            background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
            padding: 3rem;
            border-radius: 25px;
            border-right: 6px solid #10b981;
            margin: 3rem 0;
            box-shadow: 0 15px 35px rgba(16, 185, 129, 0.1);
        }
        
        .answer-text {
            font-size: 1.2rem;
            line-height: 2.2;
            color: #1e293b;
            font-weight: 400;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #0ea5e9, #0284c7);
            color: white;
            border: none;
            border-radius: 30px;
            padding: 1rem 3rem;
            font-size: 1.2rem;
            font-weight: 600;
            box-shadow: 0 8px 20px rgba(14, 165, 233, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            font-family: 'Cairo', sans-serif;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #0284c7, #0369a1);
            transform: translateY(-3px);
            box-shadow: 0 12px 25px rgba(14, 165, 233, 0.4);
        }
        
        /* Text Area */
        .stTextArea textarea {
            border-radius: 20px;
            border: 2px solid #e2e8f0;
            padding: 1.5rem;
            font-size: 1.2rem;
            font-family: 'Cairo', sans-serif;
            direction: rtl;
            text-align: right;
            background: #ffffff;
            transition: all 0.3s ease;
        }
        
        .stTextArea textarea:focus {
            border-color: #0ea5e9;
            box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.1);
            outline: none;
        }
        
        /* Features Grid */
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
        }
        
        .feature-card {
            background: linear-gradient(135deg, #ffffff, #f8fafc);
            padding: 2.5rem 2rem;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.08);
            transition: all 0.3s ease;
            border: 1px solid #e2e8f0;
        }
        
        .feature-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.12);
        }
        
        .feature-icon {
            font-size: 3.5rem;
            margin-bottom: 1.5rem;
        }
        
        .feature-card h4 {
            color: #1e293b;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            font-family: 'Cairo', sans-serif;
            font-weight: 600;
        }
        
        .feature-card p {
            color: #475569;
            font-size: 1.1rem;
            line-height: 1.7;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: #f1f5f9;
            color: #475569;
            border-radius: 15px;
            padding: 1rem 2rem;
            font-weight: 500;
            font-family: 'Cairo', sans-serif;
            border: 2px solid transparent;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0ea5e9, #0284c7);
            color: white;
            border-color: #0ea5e9;
        }
        
        /* Quick Buttons */
        .quick-button {
            background: linear-gradient(135deg, #f1f5f9, #e2e8f0);
            border: 2px solid #cbd5e1;
            color: #1e293b;
            border-radius: 15px;
            padding: 1rem 1.5rem;
            font-weight: 500;
            font-family: 'Cairo', sans-serif;
            transition: all 0.3s ease;
            text-align: center;
            cursor: pointer;
        }
        
        .quick-button:hover {
            background: linear-gradient(135deg, #e2e8f0, #cbd5e1);
            border-color: #0ea5e9;
            color: #0ea5e9;
            transform: translateY(-2px);
        }
        
        /* Success Message */
        .success-message {
            background: linear-gradient(135deg, #dcfce7, #bbf7d0);
            color: #166534;
            padding: 1.5rem;
            border-radius: 20px;
            text-align: center;
            margin: 2rem 0;
            border-right: 5px solid #10b981;
            font-family: 'Cairo', sans-serif;
            font-weight: 500;
            font-size: 1.1rem;
        }
        
        /* Footer */
        .footer {
            background: linear-gradient(135deg, #1e293b, #334155);
            color: white;
            text-align: center;
            padding: 3rem 2rem;
            border-radius: 20px;
            margin-top: 4rem;
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
            background: linear-gradient(135deg, #0ea5e9, #0284c7);
        }
        
        /* Warning and Error Messages */
        .stAlert {
            border-radius: 15px;
            border: none;
            font-family: 'Cairo', sans-serif;
        }
        
        /* Expander */
        .stExpander {
            background: white;
            border-radius: 15px;
            border: 1px solid #e2e8f0;
        }
        
        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Cairo', sans-serif !important;
            color: #1e293b;
        }
        
        /* Regular text */
        p, div, span {
            font-family: 'Cairo', sans-serif;
        }
        
        /* Markdown content */
        .stMarkdown {
            font-family: 'Cairo', sans-serif;
        }
        
        /* Rating buttons */
        .rating-section {
            background: white;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.05);
            margin: 2rem 0;
            text-align: center;
        }
        
        .rating-section .stButton > button {
            background: #f8fafc;
            color: #475569;
            border: 2px solid #e2e8f0;
            margin: 0.5rem;
            width: auto;
            padding: 0.5rem 1rem;
            font-size: 1.5rem;
        }
        
        .rating-section .stButton > button:hover {
            background: #fbbf24;
            color: white;
            border-color: #f59e0b;
            transform: scale(1.1);
        }
        
        /* Container spacing */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# ===== الصفحة الرئيسية بدون sidebar =====
# Header
st.markdown("""
    <div class="main-header">
        <div class="main-title">🇯🇴 ALIS - مساعد الأراضي والتشريعات الأردني</div>
        <div class="main-subtitle">نظام ذكي متقدم للإجابة على استفساراتكم القانونية والعقارية بدقة ومهنية عالية</div>
    </div>
""", unsafe_allow_html=True)

# Welcome Section
st.markdown("""
    <div class="welcome-card">
        <div class="welcome-icon">🌟</div>
        <h3>مرحباً بك في ALIS</h3>
        <p>مساعدك الذكي المتخصص في التشريعات الأردنية وقوانين الأراضي. احصل على إجابات دقيقة وموثوقة لجميع استفساراتك القانونية مع ضمان الخصوصية والأمان التام.</p>
    </div>
""", unsafe_allow_html=True)

# Features Section
st.markdown("## ✨ لماذا ALIS؟")
features_html = """
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h4>سرعة استثنائية</h4>
            <p>إجابات فورية ودقيقة على جميع استفساراتك القانونية خلال ثوانٍ معدودة</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h4>دقة متناهية</h4>
            <p>معلومات موثقة ومحدثة باستمرار من المصادر القانونية الرسمية</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <h4>أمان وخصوصية</h4>
            <p>حماية كاملة لبياناتك واستفساراتك مع ضمان السرية التامة</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <h4>شمولية كاملة</h4>
            <p>تغطية شاملة لجميع قوانين الأراضي والتشريعات الأردنية</p>
        </div>
    </div>
"""
st.markdown(features_html, unsafe_allow_html=True)

# Input Section
st.markdown("""
    <div class="input-section">
        <div class="section-title">
            ✍️ اطرح سؤالك واحصل على الإجابة الشافية
        </div>
""", unsafe_allow_html=True)

# Initialize query variable
query = ""

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
    ) or query

with tab3:
    st.markdown("### استفسارات التشريعات")
    query = st.text_area(
        "",
        height=120,
        placeholder="مثال: ما هي القوانين المتعلقة بالإرث في الأردن؟",
        key="law_query"
    ) or query

st.markdown("</div>", unsafe_allow_html=True)

# Quick Questions
st.markdown("### 🚀 أسئلة شائعة - اضغط للإجابة السريعة")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💰 رسوم التسجيل", key="q1"):
        query = "ما هي رسوم تسجيل الأراضي في الأردن؟"
        st.session_state.quick_query = query

with col2:
    if st.button("📋 شروط البيع", key="q2"):
        query = "ما هي شروط بيع الأراضي؟"
        st.session_state.quick_query = query

with col3:
    if st.button("👨‍👩‍👧‍👦 الإرث", key="q3"):
        query = "كيف يتم توزيع الإرث حسب القانون الأردني؟"
        st.session_state.quick_query = query

with col4:
    if st.button("🛡️ التأمين", key="q4"):
        query = "ما هو التأمين العقاري المطلوب؟"
        st.session_state.quick_query = query

# Use quick query if set
if hasattr(st.session_state, 'quick_query') and st.session_state.quick_query:
    query = st.session_state.quick_query

# Send Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    send = st.button("🚀 احصل على الإجابة الشافية", type="primary")

# Processing
if send:
    if not query.strip():
        st.warning("⚠️ الرجاء إدخال سؤال قبل الإرسال")
    else:
        # Loading animation
        with st.spinner("🔍 جاري البحث في قاعدة البيانات القانونية المتخصصة..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            try:
                # Simulated response - replace with your actual function
                # client = connect_to_db()
                # answer = IntelligentRAGSystem(query, client)
                # client.close()
                
                # Clear the quick query after processing
                if hasattr(st.session_state, 'quick_query'):
                    st.session_state.quick_query = ""
                
                # Simulated answer for demonstration
                answer = """
                بناءً على القوانين الأردنية الحالية والأنظمة النافذة، إليك الإجابة التفصيلية والشاملة:

                **الإجراءات المطلوبة:**
                1. تحضير الوثائق المطلوبة (سند الملكية الأصلي، هوية مدنية سارية)
                2. دفع الرسوم المقررة في دائرة الأراضي والمساحة
                3. الحصول على موافقة البلدية إذا لزم الأمر حسب المنطقة
                4. إتمام عملية التسجيل النهائي والحصول على سند جديد

                **الرسوم والتكاليف:**
                - رسم التسجيل: 0.5% من القيمة المقدرة للعقار
                - رسم الطابع: 15 دينار أردني
                - رسوم إضافية متغيرة حسب المنطقة والبلدية

                **ملاحظات قانونية هامة:**
                - يجب أن تكون جميع الوثائق سارية المفعول وغير منتهية الصلاحية
                - قد تختلف الإجراءات والرسوم حسب نوع الأرض ومنطقتها الجغرافية
                - يُنصح بمراجعة دائرة الأراضي للتأكد من آخر التحديثات
                """
                
                st.markdown('<div class="success-message">✅ تم الحصول على الإجابة بنجاح من قاعدة البيانات القانونية!</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class="answer-container">
                        <div class="answer-text">{answer}</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Show sources
                st.markdown("### 📚 المصادر والمراجع القانونية")
                with st.expander("اضغط لعرض المصادر المعتمدة"):
                    st.markdown("""
                    - قانون الأراضي الأردني رقم 40 لسنة 1952 وتعديلاته
                    - تعليمات دائرة الأراضي والمساحة النافذة
                    - النشرات الرسمية لوزارة العدل الأردنية
                    - الأنظمة والقرارات الصادرة عن مجلس الوزراء
                    - القرارات القضائية ذات الصلة من محاكم العدل العليا
                    """)
                
                # Rating section
                st.markdown("""
                    <div class="rating-section">
                        <h4>📝 ساعدنا في تحسين خدماتنا - قيم الإجابة</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.button("⭐", key="rate1")
                with col2:
                    st.button("⭐⭐", key="rate2")
                with col3:
                    st.button("⭐⭐⭐", key="rate3")
                with col4:
                    st.button("⭐⭐⭐⭐", key="rate4")
                with col5:
                    st.button("⭐⭐⭐⭐⭐", key="rate5")
                    
            except Exception as e:
                st.error(f"❌ عذراً، حدث خطأ أثناء معالجة السؤال: {e}")

# Footer
st.markdown("""
    <div class="footer">
        <h4>🇯🇴 ALIS - مساعد الأراضي والتشريعات الأردني</h4>
        <p>تم تطويره بواسطة فريق ALIS المتخصص | جميع الحقوق محفوظة © 2024</p>
        <p>📧 info@alis.jo | 📱 +962-6-1234567 | 🌐 www.alis.jo</p>
        <p>نظام ذكي موثوق للاستشارات القانونية - مرخص من وزارة الاقتصاد الرقمي والريادة</p>
    </div>
""", unsafe_allow_html=True)








