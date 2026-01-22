import os
import google.generativeai as genai
from pdfextractor import text_extractor
import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter



# lets configure the models here

# First lets configure the model

gemini_key = os.getenv('Gemini_API_Key')
genai.configure(api_key=gemini_key)
model = genai.GenerativeModel('gemini-2.5-flash-lite')


# configure Embeding model

embedding_model = HuggingFaceBgeEmbeddings(model_name='all-MiniLM-L6-v2')


# lets cretae the main page
st.title(':orange[CHATBOT:] :blue[AI Assisted chatbot using RAG]')

tips = '''
Follow the steps to use the application:-
* Upload your pdf Document is sidebar.
* Write a query aand start the chat
'''
st.text(tips)

# Lets create the sidebar

st.sidebar.title('grren[Upload Your File]')
st.sidebar.subheader('Uplaod pdf file only.')
pdf_file = st.sidebar.file_uploader('Upload here ',type=['pdf'])

if pdf_file:
    st.sidebar.success('File uploaded successfully')
    
    
    file_text = text_extractor(pdf_file)
    
    # step1: Chunking
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    chunks = splitter.split_text(file_text)
    
    # step2:create the vector database
    vector_store = FAISS.from_texts(chunks,embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={'k':3})
    
    def generate_content(query):
        # step3: Retriever (r)
        retrieved_docs = retriever.invoke(query)
        context = '\n'.join([d.page_content for d in retrieved_docs])
        
        # step4: Augmenting(A)
        
        augmented_prompt =f'''
        <Role> You are a helpful assistant using RAG.
        <Goal> Answer the question asked by the user. Here is the question {query}
        <Context> Here are  the documents retrieved from the vector database to support the answer which you have to generate{context}


        '''
        
        # step5:Generate(G)
        
        response  = model.generate_content(augmented_prompt)
        return response.text
        
        
    # Create chatbot in order to start the conversation
    # TO initialize a chat, create History if not already created
    if 'history' not in st.session_state:
        st.session_state.history = []
        
        
    # Display the history
    for msg in st.session_state.history:
        if msg['role']=='user':
            st.info(f':green[User:] :blue[{msg['text']}]')
        else:
            st.warning(f':orange[CHATBOT:] :blue[{msg['text']}]')
            
    # Input from the user using streamlit form
    
    with st.form('chatbot form',clear_on_submit=True):
        user_query = st.text_area('What do You want to Know...')
        send = st.form_submit_button('Send')
        
    # Start the conversaation and append  output and query in history
    
    if user_query and send:
        st.session_state.history.append({'role':'user','text':user_query})
        st.session_state.history.append({'role':'Chatbot','text':generate_content(user_query)})
        st.rerun()