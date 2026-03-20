#pip install streamlit google-genai
# to run the application, use: streamlit run app.py

import streamlit as st
import os
import time
from google import genai
from google.genai import types

# --- Page Config ---
st.set_page_config(page_title="Chat With your Document!!", layout="centered")
st.title("📄 Chat with your Document")
st.markdown("Upload a PDF or text file to start a grounded conversation.")

# --- Sidebar for Config ---
with st.sidebar:
    st.header("Setup")
    # This looks for a secret named 'GEMINI_API_KEY' in the cloud settings
    #api_key = "AIzaSyDmEmGjVgYDZ4Y_o6ewABtQgc9UQRd-wE0"
    api_key = st.text_input("Enter Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload your document", type=['pdf', 'docx', 'txt'])
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Initialize Session States ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "store_name" not in st.session_state:
    st.session_state.store_name = None

# --- Helper Function: Setup RAG Store ---
def setup_rag(file_obj, api_key):
    client = genai.Client(api_key=api_key)
    
    # Save uploaded file temporarily to disk
    temp_path = f"temp_{file_obj.name}"
    with open(temp_path, "wb") as f:
        f.write(file_obj.getbuffer())
    
    with st.status("Indexing document..."):
        # Create Store
        store = client.file_search_stores.create(config={'display_name': 'Streamlit-Store'})
        
        # Upload
        operation = client.file_search_stores.upload_to_file_search_store(
            file=temp_path,
            file_search_store_name=store.name,
        )
        
        while not operation.done:
            time.sleep(2)
            operation = client.operations.get(operation)
            
    os.remove(temp_path) # Clean up local temp file
    return store.name

# --- Main Logic ---
if api_key and uploaded_file:
    # 1. Process File once
    if st.session_state.store_name is None:
        st.session_state.store_name = setup_rag(uploaded_file, api_key)
        st.success("Document Indexed!")

    # 2. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Chat Input
    if prompt := st.chat_input("Ask something about the document"):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            client = genai.Client(api_key=api_key)
            
            # Send prompt with File Search Tool
            response = client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(file_search=types.FileSearch(
                        file_search_store_names=[st.session_state.store_name]
                    ))]
                )
            )
            
            full_response = response.text
            st.markdown(full_response)
            
        # Add bot message to state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("Please enter your API Key and upload a file in the sidebar to begin.")