import streamlit as st
import pandas as pd
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from PIL import Image
import pytesseract
from fpdf import FPDF
from io import BytesIO
import textwrap
from tavily import TavilyClient

# -- SETTINGS & INIT --
st.set_page_config(page_title="GuardianAssets AI", layout="wide")
st.title("📄 GuardianAssets AI - AI Powered Security Bot")
st.markdown("Upload Architecture Diagrams and get AI-Powered Security Requirements!")

# ---
client_tavily = TavilyClient("tvly-dev-xxxxxxxx")



# --- Sidebar: API Key & Platform Selection ---
st.sidebar.header("🔧 Settings")
user_api_key = st.sidebar.text_input("🔑 Enter your OpenAI API Key", type="password")
cloud_choice = st.sidebar.radio("Choose Cloud Provider:", ["AWS", "Azure", "GCP"], horizontal=True)
uploaded_file_kb = st.sidebar.file_uploader("📁 Upload custom datasheet [RAG] file based on choice of cloud", type=["csv"])

if not user_api_key:
    st.sidebar.warning("Please enter your OpenAI API key.")
    st.stop()
else:
    client = OpenAI(api_key=user_api_key)

@st.cache_data(show_spinner=False)
def load_csv_to_chunks(uploaded_file):
    df = pd.read_csv(uploaded_file)
    row_texts = df.astype(str).apply(lambda row: ' | '.join(row), axis=1).tolist()
    return row_texts

@st.cache_resource(show_spinner=False)
def create_chroma_db(cloud_choice, chunks):
    chroma_client = chromadb.Client(Settings(persist_directory="./chroma_data_store"))
    #chroma_client = chromadb.Client(Settings(persist_directory=f"./chroma_data_store_{cloud_choice.lower()}"))    
    collection_name = f"{cloud_choice.lower()}_knowledge_assets"
    collection = chroma_client.get_or_create_collection(name=collection_name)
    for i, chunk in enumerate(chunks):
        embedding = client.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding
        collection.add(ids=[f"chunk-{i+1}"], documents=[chunk], embeddings=[embedding])
    return collection

if uploaded_file_kb:
    row_texts = load_csv_to_chunks(uploaded_file_kb)
    collection = create_chroma_db(cloud_choice, row_texts)
else:
    st.info("👆 Please upload a CSV file to continue.")
    st.stop()

def extract_image_text(img_file):
    try:
        return pytesseract.image_to_string(Image.open(img_file))
    except Exception as e:
        st.error(f"Image text extraction failed: {e}")
        return ""

uploaded_file = st.file_uploader("📁 Upload Architecture Diagram to analyse", type=["png", "jpg", "jpeg"])
image_text = ""
if uploaded_file:
    with st.spinner("Extracting text from image..."):
        image_text = extract_image_text(uploaded_file)
    if not image_text.strip():
        st.warning("No text detected in diagram. Please try a clearer image.")
        st.stop()

def query_collection(collection, query_text, client, top_k=3):
    query_emb = client.embeddings.create(input=query_text, model="text-embedding-3-small").data[0].embedding
    result = collection.query(query_embeddings=[query_emb], n_results=top_k)
    return result["documents"][0] if result["documents"] else []

def ask_chatgpt(context, query, user_cloud_choice, client):
    messages = [
        {"role": "system", "content": f"You are a {user_cloud_choice} cloud architecture expert."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]
    response = client.chat.completions.create(model="gpt-4", messages=messages)
    return response.choices[0].message.content

def create_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
   # Split content into paragraphs
    paragraphs = text.split('\n')
    
    for para in paragraphs:
        if para.strip() == "":
            # Add a blank line for empty paragraphs
            pdf.ln(10)
        else:
            wrapped = textwrap.fill(para.strip(), width=100)
            pdf.multi_cell(0, 10, txt=wrapped)
            pdf.ln(2)  # Add small space between paragraphs

    pdf_bytes = BytesIO()
    pdf_content = pdf.output(dest='S').encode('latin1')  # Get as string, then encode
    pdf_bytes.write(pdf_content)
    pdf_bytes.seek(0)  # Reset buffer pointer
    return pdf_bytes

def web_search(text):
    response = client_tavily.search(
        query=f"{text}",max_results=10
    )
    content = ""
    for r in response.get('results'):
        content +=r['content']

    return content

tab1, tab2, tab3 = st.tabs(["📊 Cloud Assets", "📁 Current News ", "⚙️ Security Requirements "])

if image_text and uploaded_file_kb:
    top_chunks = query_collection(collection, image_text, client)

    if st.button("⚡ Identify Cloud Resources"):
        with st.spinner("Identification in progress..."):
            prompt = f"As a expert {cloud_choice} architect,list down the cloud resources only for {cloud_choice} along with the description."
            identified_resources = ask_chatgpt("\n".join(top_chunks), prompt, cloud_choice, client)
            with tab1:
                st.write(identified_resources)
                pdf_bytes = create_pdf(identified_resources)
                #st.write(pdf_bytes)
                st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_bytes,
                        file_name="cloud_resources_response.pdf",
                        mime="application/pdf"
                    )
