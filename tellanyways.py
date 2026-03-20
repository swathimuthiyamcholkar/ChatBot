# Install required packages:
# pip install -q -U google-genai requests
#pip install -q -U google-genai
#pip install -q -U google-generativeai
import os
import time
import requests
from google import genai
from google.genai import types


# -----------------------------
# 1. CONFIGURATION
# -----------------------------
FILE_ID = "1g2dJjG-1LikDs2sFt_KU_UmneSPPGIXT"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
file_path = r".\Data\Design Pattern.pdf"  # change extension if needed
api_key = "AIzaSyDmEmGjVgYDZ4Y_o6ewABtQgc9UQRd-wE0"

QUERY = "explain singleton design pattern with example"


# -----------------------------
# 2. DOWNLOAD FILE
# -----------------------------
def download_file(url, output_path):
    print("Downloading file...")
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("Failed to download file. Check sharing permissions.")

    with open(output_path, "wb") as f:
        f.write(response.content)

    print("Download complete.\n")


# -----------------------------
# 3. RAG PIPELINE
# -----------------------------
def run_rag_query(file_path, user_query):
    api_key = "AIzaSyDmEmGjVgYDZ4Y_o6ewABtQgc9UQRd-wE0"  #os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)

    print(f"--- Processing Document: {file_path} ---")

    # Create search store
    search_store = client.file_search_stores.create(
        config={'display_name': 'Docs-Store'}
    )

    try:

        #print(dir(client))
        # Upload file
        operation = client.file_search_stores.upload_to_file_search_store(
            file=file_path,
            file_search_store_name=search_store.name,
            config={'display_name': 'UploadedDocument'}
        )

        # Wait with timeout
        print("Indexing document...")
        timeout = 260  # seconds
        start_time = time.time()

        while not operation.done:
            if time.time() - start_time > timeout:
                raise TimeoutError("Indexing took too long.")

            time.sleep(2)
            #print(dir(client))
            operation = client.operations.get(operation)

        print("Document ready for questions.\n")

        # Run query
        # models  = client.models.list()  # Warm up the client
        # for m in models:
        #     print(m.name)
        #models/gemini-1.5-flash
        user_query = QUERY.strip()
        print(f"Running query: {user_query}")
        response = client.models.generate_content(
            model="models/gemini-flash-latest", #
            contents=user_query,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[search_store.name]
                        )
                    )
                ]
            )
        )

        print("----- RESULT -----")
        print(f"Question:\n{user_query}")
        print("\nAnswer:\n", response.text)
    except ValueError as e:
        ("Error:", e)
    finally:
        # Cleanup
        print("\nCleaning up resources...")
        client.file_search_stores.delete(
            name=search_store.name,
            config={'force': True}
        )
        print("Done.")


# -----------------------------
# 4. MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    try:
        #download_file(DOWNLOAD_URL, file_path)
        
        run_rag_query(file_path, QUERY)

    except Exception as e:
        print("Error:", e)