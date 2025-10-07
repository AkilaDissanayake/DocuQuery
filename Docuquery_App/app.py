import os
import streamlit as st
import openai
import atexit
from dotenv import load_dotenv

from document_processor import extract_text_from_file  #Extracts text from PDF/TXT
from text_splitter import chunk_text                   #Splits text into chunks
from embeddings_manager import EmbeddingsManager       #Create and manage text embeddings
from qa_engine import QAEngine                         #Builds prompts and interacts with the LLM


#Constants
CACHE_PATH = "embeddings_cache.npz"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1200                                     #Number of characters per chunk
CHUNK_OVERLAP = 200                                   #Overlap between chunks
TOP_K = 4                                             #Number of top similar chunks to retrieve
TEMPERATURE = 0.0                                     #Creativity level
MAX_TOKENS = 500                                      #Max tokens to generate per answer

#Clear cache files when closing the app
def clear_on_exit():
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

atexit.register(clear_on_exit) 

#Load environment variables(Access api key securely)
load_dotenv()

st.set_page_config(page_title="DocuQuery", layout="wide")
st.title("Document-based Question Answering")



#Sidebar to access settings to be used
st.sidebar.title("API key")
api_key = st.sidebar.text_input("OpenAI API Key",type="password",value=os.environ.get("OPENAI_API_KEY", ""))

#Saved to tempory environ variable to use at runtime
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key
else:
    st.warning("Please enter your OpenAI API Key in the sidebar to start using the app.")
#Permantly save API key on env file 
with open(".env", "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")

#Drop down menue to select model        
st.sidebar.title("Model Selection")
LLM_MODEL = st.sidebar.selectbox(
    "Select LLM Model",
    options=["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4o", "gpt-4-mini"],
    index=0  #Default model index
)
st.sidebar.write(f"Current Model: `{LLM_MODEL}`")


#Initialize Streamlit session state
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if api_key: #create only whe api available
 if "emb_mgr" not in st.session_state:
    st.session_state["emb_mgr"] = EmbeddingsManager(model_name=EMBED_MODEL,api_key=api_key)
if "embeddings_ready" not in st.session_state:
    st.session_state["embeddings_ready"] = False
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


#File uploader to upload txt/PDF
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files to ask questions",
    type=["pdf", "txt"],
    accept_multiple_files=True  #To upload multiple files
)




# Start processing (Embeddings are not created for each upload-reduce token usage)

if st.button("Start processing"):

    #Extract text
    if uploaded_files:
        combined_texts = []

        #Extract text from each uploaded file
        for f in uploaded_files:
            b = f.read()
            text= extract_text_from_file(b, filename=f.name)
            st.success(f"{f.name} length: {len(text)}")
            combined_texts.append((f.name, text))
            

        st.success(f"Extracted text from {len(uploaded_files)} file(s).")

        # Create chunks
        chunks =[] 
        for fname, text in combined_texts:
            if not text:              #Skip files without text
                continue
            local_chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            for chunk_text_item in local_chunks:
              chunks.append(chunk_text_item)
        st.session_state["chunks"] = chunks    

        #Create embeddings (create only whe upload button pressed since indented)
        if not st.session_state["embeddings_ready"]:
            with st.spinner("Creating embeddings..."):
                emb_mgr = st.session_state["emb_mgr"]
                try:
                    emb_mgr.create_embeddings(st.session_state["chunks"])
                except openai.OpenAIError as e:  # Catches all OpenAI API errors
                    st.error(f"OpenAI API error: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"Unexpected error during embedding: {e}")
                    st.stop()

                st.session_state["embeddings_ready"] = True
                emb_mgr.save_npz(CACHE_PATH)
            st.success("Embeddings created.")



#Question input
st.markdown("---")
st.header("Ask questions about the uploaded documents")
question = st.text_area("Enter your question here", height=160)


if st.button("Get Answer"):
    if not st.session_state["chunks"]:
        st.error("Upload files first.")
    elif not st.session_state["embeddings_ready"]:
        st.error("Embeddings are not ready yet.")
    elif not question.strip():
        st.error("Please type a question.")
    else:
        Answer_pressed=True
        emb_mgr = st.session_state["emb_mgr"]
        with st.spinner("Retrieving relevant chunks..."):
            top_idx_scores = emb_mgr.retrieve(query=question, k=TOP_K) #find the most relevant chunks to question
            retrieved = []
            for idx, score in top_idx_scores:
                chunk_text_item = st.session_state["chunks"][idx]
                retrieved.append((chunk_text_item,score))

        qa = QAEngine(llm_model=LLM_MODEL,api_key=api_key)
        prompt = qa.build_prompt(question, retrieved)   #Combines the question and the retrieved relevant chunks
        with st.spinner("Generating answer from LLM..."):
            try:
                answer = qa.generate_answer(
                    prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )

                # Display answer
                st.subheader("Answer")
                st.write(answer)
                st.session_state["chat_history"].append({"question": question,"answer": answer}) #Store to history
                
            #Handle possible errors
            except openai.OpenAIError as e:
                st.error(f"LLM request failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

st.markdown("---")


#Show chat history (below answer section)
if st.session_state.get("chat_history"):
    st.markdown("### Chat History")
    #Last added not printed since it is displaying now
    for chat in reversed(st.session_state["chat_history"][:-1]):#print in reverse to get last question first 
        st.markdown(f"**Q:** {chat['question']}")
        st.markdown(f"**A:** {chat['answer']}")
        st.markdown("---")

  



# Future optimization:
  #Remove session data for files no longer uploaded
  #Keep embeddings/chunks for files still present
  #Identify only new files that need chunking & embedding
  #This will reduce token usage for repeated queries

  #Show user the relavant answer page to question
