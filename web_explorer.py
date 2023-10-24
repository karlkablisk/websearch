import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever

import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

#os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # Get it at https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials
#os.environ["GOOGLE_CSE_ID"] = "YOUR_CSE_ID" # Get it at https://programmablesearchengine.google.com/
#os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
#os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY" 

st.set_page_config(page_title="HORO AI Search Engine", page_icon="🌐")

#models
turbo = "gpt-3.5-turbo"
turbo_16k = "gpt-3.5-turbo-16k"
gpt4 = "gpt-4"

def settings():

    # Vectorstore
    import faiss
    from langchain.vectorstores import FAISS 
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.docstore import InMemoryDocstore  
    embeddings_model = OpenAIEmbeddings()  
    embedding_size = 1536  
    index = faiss.IndexFlatL2(embedding_size)  
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


    # LLM
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model_name=turbo, temperature=0, streaming=True)

    # Search
    from langchain.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()   

    # Initialize 
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm, 
        search=search, 
        num_search_results=1
    )

    return web_retriever, llm

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


st.sidebar.image("img/horofox.png")
st.header("`HORO AI Search Engine`")
st.info("Horo has been uploaded to the internet. Now she helps anyone answer questions by exploring, reading, and summarizing web pages.")

# Make retriever and llm
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input 
question = st.text_input("`Ask a question:`")

if question:

    # Generate answer (w/ citations)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Write answer and sources
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])
    answer.info('`Answer:`\n\n' + result['answer'])
    st.info('`Sources:`\n\n' + result['sources'])

# Code to save files


def mkdir_p(sftp, remote_directory):
  """Change to this directory, recursively making new folders if needed.
    Returns True if any folders were created."""
  if remote_directory == '/':
    # absolute path so change directory to root
    sftp.chdir('/')
    return
  if remote_directory == '':
    # top-level relative directory must exist
    return
  try:
    sftp.chdir(remote_directory)  # sub-directory exists
  except IOError:
    dirname, basename = os.path.split(remote_directory.rstrip('/'))
    mkdir_p(sftp, dirname)  # make parent directories
    sftp.mkdir(basename)  # sub-directory missing, so created it
    sftp.chdir(basename)
    return True


def upload_to_server(filename, uid, remote_folder):
  try:
    print("Attempting to connect...")

    # Initialize the SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Connection details: replace with your actual details
    ssh.connect(hostname='',
                username='',
                password=os.environ['SSH_PASS'])
    print("SSH Connection Successful.")

    # Open the SFTP session
    sftp = ssh.open_sftp()

    # User-specific details
    user_name = get_user_name()
    #print(f"UID: {uid}")  # Display UID
    #print(f"Username: {user_name}")  # Display Username

    # Remote directory path: replace with your actual path
    remote_directory = f"/applications/site/public_html/{remote_folder}/"

    # Ensure the remote directory exists
    mkdir_p(sftp, remote_directory)

    # Check if remote directory exists, if not create it
    try:
      sftp.stat(remote_directory)
    except FileNotFoundError:
      sftp.mkdir(remote_directory)

    # Full path to the file on the remote server
    remote_path = remote_directory + filename

    # Upload the file
    sftp.put(filename, remote_path)
    print("File uploaded successfully.")

    # Close the SFTP and SSH sessions
    sftp.close()
    ssh.close()

  except Exception as e:
    print(f"An error occurred: {e}")

