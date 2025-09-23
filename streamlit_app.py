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
st.set_page_config(page_title="ALIS - Jordan RAG Assistant 🇯🇴", layout="wide")

# ===== CSS لتجميل الواجهة =====
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
            font-family: "Segoe UI", Arial, sans-serif;
        }
        .title-text {
            font-size: 28px !important;
            font-weight: bold;
            color: #2c3e50;
        }
        .subtitle-text {
            color: #6c757d;
            font-size: 15px;
        }
        .welcome-box {
            background: linear-gradient(135deg, #e3f2fd, #ffffff);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
        }
        .answer-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            direction: rtl;
            line-height: 1.8;
            font-size: 16px;
        }
        .footer {
            text-align: center;
            color: gray;
            margin-top: 30px;
            font-size: 13px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 25px;
            font-size: 16px;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# ===== الشعار والعنوان =====
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/8/89/Scale_of_justice.png", width=80)

with col2:
    st.markdown("<div class='title-text'>🤖 ALIS - Jordan RAG Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle-text'>نظام ذكي للإجابة على الأسئلة المتعلقة بالأراضي والتشريعات الأردنية</div>", unsafe_allow_html=True)

st.markdown("---")

# ===== الترحيب =====
st.markdown("""
<div class="welcome-box">
    👋 أهلاً وسهلاً بك في <strong>ALIS</strong><br>
    مساعدك الذكي للإجابة على أسئلتك حول التشريعات والأراضي الأردنية 🇯🇴
</div>
""", unsafe_allow_html=True)

# ===== الإدخال =====
st.markdown("### ✍️ أدخل سؤالك:")
query = st.text_area(
    "📝 اكتب سؤالك هنا:",
    height=120,
    placeholder="مثال: ما هي رسوم تسجيل قطعة أرض؟"
)

# ===== زر الإرسال =====
send = st.button("🚀 إرسال السؤال")

# ===== المعالجة =====
if send:
    if not query.strip():
        st.warning("⚠ الرجاء إدخال سؤال.")
    else:
        with st.spinner("🔍 جاري المعالجة، الرجاء الانتظار..."):
            try:
                client = connect_to_db()
                answer = IntelligentRAGSystem(query, client)
                client.close()
                st.success("✅ تم الحصول على الإجابة:")
                st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"حدث خطأ أثناء الإجابة: {e}")

# ===== تذييل الصفحة =====
st.markdown("---")
st.markdown(
    "<div class='footer'>تم تطوير النظام بواسطة مشروع <strong>ALIS</strong> - الذكاء الاصطناعي للأراضي والمساحة 🇯🇴</div>",
    unsafe_allow_html=True
)







