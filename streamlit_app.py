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


st.set_page_config(page_title="Jordan RAG Assistant 🇯🇴", layout="wide")
st.title("🤖 Jordan RAG Assistant")
st.write("أدخل سؤالك المتعلق بالأراضي والتشريعات:")

query = st.text_input("اكتب سؤالك هنا:")

if st.button("إرسال"):
    if query.strip() == "":
        st.warning("⚠ الرجاء إدخال سؤال")
    else:
        with st.spinner("🔍 جاري البحث ..."):
            client = connect_to_db()
            answer = IntelligentRAGSystem(query, client)
            client.close()
        st.success("✅ تم الحصول على الإجابة:")
        st.write(answer)


