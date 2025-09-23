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







