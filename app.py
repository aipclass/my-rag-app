import streamlit as st
import os
import arxiv
# [最终修复 1]: 导入官方推荐的 HuggingFaceEndpoint 替代 HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables. On Streamlit Cloud, this reads from Secrets.
load_dotenv()

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI论文搜索与问答机器人", page_icon=" C", layout="wide")
st.title(" C AI论文搜索与问答机器人")
st.write("在这里，您可以搜索arXiv上的论文，并与选定的论文进行智能对话。")

# --- 2. Directory Path Definition ---
PDF_SAVE_PATH = "downloaded_papers"
if not os.path.exists(PDF_SAVE_PATH):
    os.makedirs(PDF_SAVE_PATH)

# --- 3. Cached Data Processing Function ---
@st.cache_resource
def get_retriever_and_metadata(_paper_id):
    print(f"--- [Cache Miss] Building retriever for paper {_paper_id} ---")
    client = arxiv.Client()
    search = arxiv.Search(id_list=[_paper_id])
    paper = next(client.results(search))
    
    pdf_filename = f"{paper.entry_id.split('/')[-1]}.pdf"
    local_pdf_path = os.path.join(PDF_SAVE_PATH, pdf_filename)
    if not os.path.exists(local_pdf_path):
        paper.download_pdf(dirpath=PDF_SAVE_PATH, filename=pdf_filename)
    
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    
    loader = PyMuPDFLoader(local_pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 6})

    return retriever, paper, local_pdf_path

# --- Session State Initialization ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'search'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )

# --- Application Flow ---

if st.session_state.stage == 'search':
    st.header("1. 搜索论文")
    query = st.text_input("输入您想搜索的论文关键词", key="search_query")
    if st.button(" C 搜索"):
        if query:
            with st.spinner("正在arXiv上搜索..."):
                client = arxiv.Client()
                search = arxiv.Search(query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance)
                results = list(client.results(search))
                if results:
                    st.session_state.search_results = results
                    st.session_state.stage = 'select'
                    st.rerun()
                else:
                    st.error("没有找到相关的论文，请换个关键词试试。")
        else:
            st.warning("请输入搜索关键词。")

elif st.session_state.stage == 'select':
    st.header("2. 选择一篇论文进行对话")
    if 'search_results' in st.session_state and st.session_state.search_results:
        for paper in st.session_state.search_results:
            st.subheader(paper.title)
            st.write(f"**作者**: {', '.join(author.name for author in paper.authors)}")
            st.write(f"**摘要**: {paper.summary[:300]}...")
            paper_id = paper.entry_id.split('/')[-1]
            if st.button(f" C 与这篇论文对话", key=f"select_{paper_id}"):
                st.session_state.selected_paper_id = paper_id
                st.session_state.stage = 'chat'
                st.session_state.messages = []
                st.session_state.memory.clear()
                st.rerun()
    if st.button("返回搜索"):
        st.session_state.pop('search_results', None)
        st.session_state.stage = 'search'
        st.rerun()

elif st.session_state.stage == 'chat':
    paper_id = st.session_state.selected_paper_id

    try:
        retriever, paper_metadata, downloaded_pdf_path = get_retriever_and_metadata(paper_id)

        # [最终修复 2]: 使用 HuggingFaceEndpoint 替代 HuggingFaceHub
        # 它更现代，且能正确处理新版 huggingface_hub 库
        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen1.5-7B-Chat",
            temperature=0.3,
            max_new_tokens=2048,
            # 它会自动从环境变量(Secrets)中读取HUGGINGFACEHUB_API_TOKEN
        )
        
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.memory,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template="""[任务指令]
                    你是一个顶级的AI学术研究员，你的任务是基于下方提供的“[论文相关内容]”，以一种深刻、专业且富有洞察力的口吻，详细回答用户的“[问题]”。
                    [知识范围]: 你的所有回答必须严格来源于下方提供的“[论文相关内容]”。绝对禁止使用任何外部知识或进行无根据的猜测。
                    [约束条件]: 如果内容片段确实无法支撑回答，就直截了当地说：“这篇论文的相关部分未讨论此问题。”
                    ---
                    [论文相关内容]: {context}
                    ---
                    [问题]: {question}
                    [你的专家级分析回答]:
                    """,
                    input_variables=["context", "question"]
                )
            },
            return_source_documents=True
        )

        st.header(f"3. 正在与论文对话: {paper_metadata.title}")
        with open(downloaded_pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 下载当前论文PDF",
                data=pdf_file,
                file_name=os.path.basename(downloaded_pdf_path),
                mime="application/octet-stream"
            )
        st.divider()

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("请就这篇论文提问：")
        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.spinner("模型正在检索与思考中..."):
                result = rag_chain.invoke({"question": user_question})
                ai_response = result['answer']
                
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                    with st.expander("查看本次回答引用的原文片段"):
                        for doc in result.get('source_documents', []):
                            st.markdown(f"> {doc.page_content}\n\n_(来源: PDF 第 {doc.metadata.get('page', 'N/A')} 页)_")

        if st.button(" C 返回论文选择列表"):
            st.session_state.stage = 'select'
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.session_state.pop('selected_paper_id', None)
            st.rerun()

    except Exception as e:
        st.error(f"处理对话时发生错误: {e}")
        if st.button("返回重试"):
            get_retriever_and_metadata.clear()
            st.rerun()
