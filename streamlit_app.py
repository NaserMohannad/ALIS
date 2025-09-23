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
    page_title="ALIS - مساعد الأراضي والتشريعات الأردني",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== حالة الإدخال =====
if "query" not in st.session_state:
    st.session_state.query = ""

def set_query(text: str):
    st.session_state.query = text

# ===== CSS احترافي هادئ =====
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');

        * { direction: rtl; text-align: right; }
        .stApp { background: #f6f7f9; }
        .block-container { padding: 2rem 1rem; max-width: 1200px; }

        /* رأس الصفحة */
        .hero {
            background: #0f172a;
            color: #fff;
            border-radius: 20px;
            padding: 2.5rem 2rem;
            box-shadow: 0 16px 40px rgba(15, 23, 42, 0.25);
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.06);
            font-family: 'Cairo', sans-serif;
        }
        .hero h1 {
            font-weight: 700;
            font-size: 2.2rem;
            margin: 0 0 0.5rem 0;
            letter-spacing: 0;
        }
        .hero p {
            margin: 0;
            color: #cbd5e1;
            font-size: 1.05rem;
            line-height: 1.8;
        }

        /* بطاقة المقدمة */
        .intro {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1.75rem;
            margin: 1.25rem 0 2rem 0;
            box-shadow: 0 10px 24px rgba(0,0,0,0.04);
            font-family: 'Cairo', sans-serif;
        }
        .intro h3 {
            margin: 0 0 0.75rem 0;
            color: #0f172a;
            font-weight: 700;
            font-size: 1.25rem;
        }
        .intro p {
            margin: 0;
            color: #475569;
            font-size: 1.05rem;
        }

        /* قسم الإدخال */
        .input-wrap {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 22px rgba(0,0,0,0.04);
            font-family: 'Cairo', sans-serif;
        }
        .section-title {
            color: #0f172a;
            font-weight: 700;
            font-size: 1.25rem;
            margin-bottom: 1rem;
        }
        .stTextArea textarea {
            border-radius: 14px;
            border: 1.5px solid #e5e7eb;
            padding: 1.25rem;
            font-size: 1.05rem;
            background: #fbfbfc;
            transition: all .2s ease;
            min-height: 140px;
        }
        .stTextArea textarea:focus {
            border-color: #2563eb;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.12);
            outline: none;
            background: #fff;
        }
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            padding: 0.9rem 1.25rem;
            font-weight: 700;
            border: 1px solid transparent;
            background: #1d4ed8;
            color: #fff;
            box-shadow: 0 8px 20px rgba(29, 78, 216, 0.25);
            transition: transform .06s ease, box-shadow .2s ease, background .2s ease;
            font-family: 'Cairo', sans-serif;
            font-size: 1.05rem;
        }
        .stButton > button:hover {
            background: #1e40af;
            box-shadow: 0 10px 24px rgba(30, 64, 175, 0.28);
        }
        .stButton > button:active { transform: translateY(1px); }

        /* أزرار الأسئلة الشائعة */
        .quick-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.75rem;
            margin-top: 0.75rem;
        }
        .quick-btn {
            display: block;
            width: 100%;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            color: #0f172a;
            border-radius: 12px;
            padding: 0.85rem 1rem;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: background .15s ease, border-color .15s ease, color .15s ease, transform .06s ease;
            font-family: 'Cairo', sans-serif;
            text-align: center;
        }
        .quick-btn:hover { background: #eef2f7; border-color: #cbd5e1; }
        .quick-btn:active { transform: translateY(1px); }

        /* صندوق الإجابة بلون مغاير */
        .answer {
            background: #f4f7ff; /* أزرق باهت مريح */
            border: 1px solid #dbe6ff;
            border-right: 5px solid #3b82f6;
            border-radius: 14px;
            padding: 1.25rem 1.25rem;
            margin-top: 1rem;
            box-shadow: 0 8px 22px rgba(59, 130, 246, 0.10);
            font-family: 'Cairo', sans-serif;
        }
        .answer p, .answer div { margin: 0; color: #0f172a; line-height: 1.95; font-size: 1.05rem; }

        /* الفوتر */
        .footer {
            background: #0f172a;
            color: #e2e8f0;
            border-radius: 16px;
            padding: 1.75rem;
            margin-top: 2rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.06);
            font-family: 'Cairo', sans-serif;
        }
        .footer p { margin: .25rem 0; }
    </style>
""", unsafe_allow_html=True)

# ===== رأس الصفحة =====
st.markdown("""
    <div class="hero">
        <h1>ALIS - مساعد الأراضي والتشريعات الأردني</h1>
        <p>نظام ذكي متقدم للإجابة على استفساراتكم القانونية والعقارية بدقة ومهنية، مع واجهة عربية مُحكمة واتجاه نص من اليمين لليسار.</p>
    </div>
""", unsafe_allow_html=True)

# ===== مقدّمة مختصرة =====
st.markdown("""
    <div class="intro">
        <h3>مرحبًا بك في ALIS</h3>
        <p>يوفر لك ALIS إجابات موثوقة ومبسطة حول قوانين وتشريعات الأراضي في الأردن. اكتب سؤالك بوضوح وستحصل على رد مُفصل مدعوم بالمراجع.</p>
    </div>
""", unsafe_allow_html=True)

# ===== قسم الإدخال =====
st.markdown('<div class="input-wrap">', unsafe_allow_html=True)
st.markdown('<div class="section-title">اكتب سؤالك</div>', unsafe_allow_html=True)

# صندوق الإدخال
st.session_state.query = st.text_area(
    label="",
    value=st.session_state.query,
    key="query",
    height=160,
    placeholder="مثال: ما هي الإجراءات والرسوم المطلوبة لتسجيل قطعة أرض؟",
)

# الأسئلة الشائعة — تملأ صندوق الإدخال مباشرة
st.markdown("**أسئلة شائعة**")
qcols = st.columns(1)
with qcols[0]:
    st.markdown(
        """
        <div class="quick-grid">
            <a class="quick-btn" href="#" onclick="return false;">رسوم تسجيل الأراضي</a>
            <a class="quick-btn" href="#" onclick="return false;">شروط بيع الأراضي ونقل الملكية</a>
            <a class="quick-btn" href="#" onclick="return false;">متطلبات الحصول على سند ملكية جديد</a>
            <a class="quick-btn" href="#" onclick="return false;">الرسوم والضرائب عند تحويل الملكية</a>
        </div>
        """,
        unsafe_allow_html=True
    )
# ملاحظة: الروابط أعلاه للعرض فقط. لضبط القيمة فعليًا نستخدم أزرار Streamlit التالية:

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.button("رسوم تسجيل الأراضي", use_container_width=True,
              on_click=set_query, args=("ما هي رسوم تسجيل الأراضي في الأردن؟",))
with c2:
    st.button("شروط بيع الأراضي ونقل الملكية", use_container_width=True,
              on_click=set_query, args=("ما هي شروط بيع الأراضي وإجراءات نقل الملكية؟",))
with c3:
    st.button("سند ملكية جديد", use_container_width=True,
              on_click=set_query, args=("ما متطلبات إصدار سند ملكية جديد لقطعة أرض؟",))
with c4:
    st.button("الرسوم والضرائب عند التحويل", use_container_width=True,
              on_click=set_query, args=("ما الرسوم والضرائب المترتبة عند تحويل ملكية أرض؟",))

st.markdown("</div>", unsafe_allow_html=True)  # إغلاق input-wrap

# ===== زر الإرسال =====
st.markdown("")
send_col = st.columns([1, 2, 1])[1]
with send_col:
    send = st.button("احصل على الإجابة", type="primary")

# ===== المعالجة =====
if send:
    if not st.session_state.query.strip():
        st.warning("يرجى إدخال سؤال قبل الإرسال.")
    else:
        with st.spinner("جاري البحث في المصادر القانونية المعتمدة..."):
            progress_bar = st.progress(0)
            for i in range(40):
                time.sleep(0.02)
                progress_bar.progress(min(i + 1, 100))

            try:
                # استبدل هذا الجزء باستدعاءاتك الفعلية
                # client = connect_to_db()
                # answer = IntelligentRAGSystem(st.session_state.query, client)
                # client.close()

                answer = """
                بناءً على القوانين والأنظمة النافذة في الأردن، يمكن تلخيص الإجراءات والرسوم كما يلي:

                **الإجراءات:**
                1) تجهيز الوثائق الرسمية (سند ملكية، هوية سارية، وأي موافقات لازمة حسب المنطقة).
                2) تقديم الطلب لدى دائرة الأراضي والمساحة المختصة.
                3) تسديد الرسوم المحددة بحسب نوع المعاملة وقيمة العقار.
                4) استلام السند أو القيد بعد إتمام المعاملة.

                **رسوم متعارف عليها (قد تتغير حسب التصنيف والمنطقة):**
                - رسم التسجيل: نسبة مئوية من القيمة المقدرة للعقار.
                - رسوم طوابع وأتعاب إدارية.
                - رسوم خدمات بلدية (إن وُجدت).

                **تنبيه مهم:**
                تتغير النسب والرسوم بتحديث التعليمات؛ يُنصح بمراجعة دائرة الأراضي للتأكد من آخر القرارات المعتمدة.
                """

                st.markdown('<div class="answer">', unsafe_allow_html=True)
                st.markdown(answer, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # مصادر
                st.markdown("#### المصادر والمراجع")
                with st.expander("عرض المراجع"):
                    st.markdown("""
                    - قانون الأراضي والأملاك الأردني وتعديلاته  
                    - تعليمات ونشرات دائرة الأراضي والمساحة  
                    - الأنظمة والقرارات ذات الصلة الصادرة عن الجهات الرسمية  
                    """)

            except Exception as e:
                st.error(f"حدث خطأ أثناء المعالجة: {e}")

# ===== الفوتر =====
st.markdown("""
    <div class="footer">
        <p>ALIS - مساعد الأراضي والتشريعات الأردني</p>
        <p>جميع الحقوق محفوظة © 2024</p>
        <p>info@alis.jo | +962-6-1234567 | www.alis.jo</p>
    </div>
""", unsafe_allow_html=True)









