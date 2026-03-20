#pip install streamlit google-genai
# to run the application, use: streamlit run appmemory.py
import streamlit as st
import os
import time
from google import genai
from google.genai import types
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Business Assistant Pro", layout="centered")
st.title("🚀 Business Intelligence Bot")
st.markdown("Chat with **Documents**, analyze **Images**, and save your work.")

# --- Sidebar ---
with st.sidebar:
    st.header("1. Setup")
    api_key = st.text_input("Gemini API Key", type="password")
    
    st.header("2. Knowledge Base")
    uploaded_doc = st.file_uploader("Upload Document (PDF/TXT)", type=['pdf', 'txt'])
    uploaded_img = st.file_uploader("Upload Image for Analysis", type=['png', 'jpg', 'jpeg'])
    
    if st.button("Clear All"):
        st.session_state.chat_history = []
        st.session_state.store_name = None
        st.rerun()

# --- Initialize Session States ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "store_name" not in st.session_state:
    st.session_state.store_name = None

# --- Helper: Setup RAG ---
def setup_rag(file_obj, api_key):
    client = genai.Client(api_key=api_key)
    temp_path = f"temp_{file_obj.name}"
    with open(temp_path, "wb") as f:
        f.write(file_obj.getbuffer())
    
    with st.status("Indexing document for RAG..."):
        store = client.file_search_stores.create(config={'display_name': 'Business-Store'})
        client.file_search_stores.upload_to_file_search_store(
            file=temp_path,
            file_search_store_name=store.name,
        )
        time.sleep(5) # Minimal wait for indexing
    os.remove(temp_path)
    return store.name

# --- Main Logic ---
if api_key:
    client = genai.Client(api_key=api_key)

    # Handle Doc Indexing
    if uploaded_doc and st.session_state.store_name is None:
        st.session_state.store_name = setup_rag(uploaded_doc, api_key)

    # Display Chat History
    for msg in st.session_state.chat_history:
        role = "user" if msg.role == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.parts[0].text)

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare content list (Text + Image if provided)
        current_parts = [types.Part.from_text(text=prompt)]
        
        if uploaded_img:
            img = Image.open(uploaded_img)
            current_parts.append(img)
            st.image(img, caption="Image uploaded for context", width=300)

        # Generate Response
        with st.chat_message("assistant"):
            tools = []
            if st.session_state.store_name:
                tools.append(types.Tool(file_search=types.FileSearch(
                    file_search_store_names=[st.session_state.store_name]
                )))

            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=st.session_state.chat_history + [types.Content(role="user", parts=current_parts)],
                config=types.GenerateContentConfig(tools=tools) if tools else None
            )
            
            answer = response.text
            st.markdown(answer)

        # Save History
        st.session_state.chat_history.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
        st.session_state.chat_history.append(types.Content(role="model", parts=[types.Part.from_text(text=answer)]))

    # --- Download Feature ---
    if st.session_state.chat_history:
        chat_text = ""
        for m in st.session_state.chat_history:
            role_label = "User" if m.role == "user" else "Assistant"
            chat_text += f"{role_label}: {m.parts[0].text}\n\n"
        
        st.download_button(
            label="📥 Download Chat Log",
            data=chat_text,
            file_name="business_chat_log.txt",
            mime="text/plain"
        )
else:
    st.warning("Enter your API Key to start.")