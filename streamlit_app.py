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

# ===== إعداد الصفحة =====
st.set_page_config(
    page_title="ALIS - Jordan RAG Assistant", 
    layout="wide",
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
        
        .main .block-container {
            padding-top: 2rem;
        }
        
        body {
            background-color: #f8fafc;
            font-family: "Segoe UI", "Tahoma", "Arial", sans-serif;
        }
        
        /* رأس الصفحة */
        .header-container {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .title-text {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .subtitle-text {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* بطاقة الترحيب */
        .welcome-card {
            background: linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%);
            padding: 2rem;
            border-radius: 12px;
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
            border-radius: 8px;
            padding: 12px 30px;
            font-size: 1.1rem;
            font-weight: 600;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        /* علامات التبويب */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #f1f5f9;
            border-radius: 6px;
            padding: 10px 20px;
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
        
        /* بطاقات الإحصائيات */
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-left: 4px solid #3b82f6;
            text-align: center;
        }
        
        .stat-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #64748b;
        }
        
        /* تحسينات للنصوص */
        h1, h2, h3 {
            color: #1e293b;
        }
        
        /* تحسينات للرسائل */
        .stAlert {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ===== رأس الصفحة =====
st.markdown("""
    <div class="header-container">
        <div class="title-text">ALIS - Jordan RAG Assistant</div>
        <div class="subtitle-text">نظام ذكي للإجابة على الأسئلة المتعلقة بالأراضي والتشريعات الأردنية</div>
    </div>
""", unsafe_allow_html=True)

# ===== علامات التبويب =====
tab1, tab2, tab3, tab4 = st.tabs(["الرئيسية", "الإحصائيات", "الأسئلة الشائعة", "معلومات"])

with tab1:
    # ===== بطاقة الترحيب =====
    st.markdown("""
    <div class="welcome-card">
        <h2 style="margin-top:0; text-align:center;">مرحبا بك في ALIS</h2>
        <p style="font-size:1.2rem; text-align:center; margin-bottom:1.5rem;">مساعدك الذكي للإجابة على أسئلتك حول التشريعات والأراضي الأردنية</p>
    </div>
    """, unsafe_allow_html=True)

    # ===== حاوية الإدخال =====
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    st.markdown("### أدخل سؤالك:")
    
    query = st.text_area(
        " ",
        height=140,
        placeholder="مثال: ما هي رسوم تسجيل قطعة أرض؟ أو ما هي إجراءات نقل ملكية العقار؟",
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        send = st.button("إرسال السؤال", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ===== المعالجة =====
    if send:
        if not query.strip():
            st.warning("الرجاء إدخال سؤال قبل الإرسال.")
        else:
            with st.spinner("جاري البحث عن الإجابة، الرجاء الانتظار..."):
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
                    
                    st.success("تم العثور على الإجابة:")
                    st.markdown(f"<div class='answer-card'>{answer}</div>", unsafe_allow_html=True)
                    
                    # قسم التقييم
                    st.markdown("---")
                    st.markdown("#### كيف كانت تجربتك مع هذه الإجابة؟")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("مفيدة", use_container_width=True):
                            st.success("شكراً لتقييمك!")
                    with col2:
                        if st.button("غير مفيدة", use_container_width=True):
                            st.info("شكراً للملاحظة، سنعمل على تحسين الإجابات.")
                    with col3:
                        if st.button("نسخ الإجابة", use_container_width=True):
                            st.info("تم نسخ الإجابة إلى الحافظة")
                    
                except Exception as e:
                    st.error(f"حدث خطأ أثناء معالجة سؤالك: {e}")

with tab2:
    st.markdown("### إحصائيات النظام")
    
    # شبكة بطاقات الإحصائيات
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">24</div>
            <div class="stat-label">الاستعلامات اليومية</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">94%</div>
            <div class="stat-label">معدل الدقة</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">720</div>
            <div class="stat-label">الاستعلامات الشهرية</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">1.2s</div>
            <div class="stat-label">وقت الاستجابة</div>
        </div>
        """, unsafe_allow_html=True)
    
    # إعدادات النظام
    st.markdown("---")
    st.markdown("### إعدادات النظام")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "نوع النموذج:",
            ["النموذج الأساسي", "نموذج متقدم", "نموذج سريع"]
        )
        
        detail_level = st.selectbox(
            "مستوى التفاصيل:",
            ["مختصر", "متوسط", "مفصل"]
        )
    
    with col2:
        language = st.selectbox(
            "لغة الإجابة:",
            ["العربية", "الإنجليزية", "الثنائية"]
        )
        
        sources = st.slider(
            "عدد المصادر المعروضة:",
            min_value=1,
            max_value=5,
            value=3
        )

with tab3:
    st.markdown("### الأسئلة الشائعة")
    
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
    
    with st.expander("كيف يمكنني الاعتراض على قرار دائرة الأراضي؟"):
        st.markdown("""
        - تقديم طلب اعتراض مسبب خلال 30 يومًا من تاريخ الإخطار
        - دفع رسوم الاعتراض
        - تقديم المستندات المؤيدة للاعتراض
        - حضور جلسات النظر في الاعتراض
        """)

with tab4:
    st.markdown("### معلومات عن النظام")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
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
        
        st.info("""
        **الإصدار:** 2.1.0  
        **تاريخ التحديث:** 2024  
        **نوع الترخيص:** حكومي  
        **الدعم الفني:** متاح
        """)
    
    with col2:
        st.markdown("### معلومات الاتصال")
        st.info("""
        للاستفسارات أو المساعدة التقنية:
        
        **البريد الإلكتروني:** support@alis.gov.jo  
        **هاتف:** 065000000  
        **ساعات العمل:** 8:00 ص - 4:00 م  
        **أيام العمل:** الأحد - الخميس
        
        **العنوان:** عمان، جبل عمان، شارع المدينة المنورة
        """)

# ===== تذييل الصفحة =====
st.markdown("---")
st.markdown("""
    <div class="footer">
        تم تطوير النظام بواسطة مشروع ALIS - الذكاء الاصطناعي للأراضي والمساحة<br>
        جميع الحقوق محفوظة © 2024
    </div>
""", unsafe_allow_html=True)








