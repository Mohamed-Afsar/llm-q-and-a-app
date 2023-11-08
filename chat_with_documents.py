import os
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Function that loads a document
def load_document(file):
    name, extension = os.path.splitext(file)
    print(f'Loading {file}...')
    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported')
        return None

    data = loader.load()
    return data

# Chunking
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(data)
    return chunks

# Calculating embedding cost
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return  total_tokens, total_tokens / 1000 * 0.0004

# Embedding
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

# Ask and get an answer
def ask_and_get_answer(vector_store, question, k=3):
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import RetrievalQA

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(seatch_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.run(question)
    return answer

# UI
if __name__ == '__main__':
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    st.image('img.png')
    st.subheader('LLM Question-Answering Application 🤖')

    with st.sidebar:
        api_key = st.text_input('OpenAI APK key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'txt', 'docx'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512)
        k_val = st.number_input('k:', min_value=1, max_value=20, value=3)
        add_data = st.button('Add Data')

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embdedding the file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size} and Chunks: {len(chunks)}')

                tokens_count, cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: {cost:.4f}')

                vector_store = create_embeddings(chunks)

                st.session_state.vs = vector_store
                st.success('File successfully uploaded, chunked and embedded!')

    question = st.text_input('Ask a question about the content of your file:')

    if question:
        if 'vs' in st.session_state:
            st.write(f'k: {k_val}')
            answer = ask_and_get_answer(st.session_state.vs, question, k_val)
            st.text_area('LLM Answer: ', value=answer)

    st.divider()
    if 'history' not in st.session_state:
        st.session_state.history = ''

    value = f'Q: {question}\nA: {answer}'
    st.session_state.history = f'{value}\n{"-" * 50}\n{st.session_state.history}'
    st.text_area(label='Q&A History', value=st.session_state.history, key='qa_history_text_area', height=400)
