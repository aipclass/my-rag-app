import streamlit as st
import os
import arxiv
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="AI论文搜索与问答机器人", page_icon=" C", layout="wide")
st.title(" C AI论文搜索与问答机器人")
st.write("在这里，您可以搜索arXiv上的论文，并与选定的论文进行智能对话。")

PDF_SAVE_PATH = "downloaded_papers"
if not os.path.exists(PDF_SAVE_PATH):
    os.makedirs(PDF_SAVE_PATH)

@st.cache_resource
def setup_pipelines(_paper_id):
    print(f"--- 正在为论文 {_paper_id} 构建RAG流水线 ---")

    client = arxiv.Client()
    search = arxiv.Search(id_list=[_paper_id])
    paper = next(client.results(search))
    
    pdf_filename = f"{paper.entry_id.split('/')[-1]}.pdf"
    local_pdf_path = os.path.join(PDF_SAVE_PATH, pdf_filename)
    if not os.path.exists(local_pdf_path):
        paper.download_pdf(dirpath=PDF_SAVE_PATH, filename=pdf_filename)
    print(f"--- 论文PDF '{pdf_filename}' 已就绪 ---")

    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    loader = PyMuPDFLoader(local_pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 6})
    repo_id = "Qwen/Qwen1.5-7B-Chat"
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.3, "max_length": 2048}
    )
    
    qa_template = """[任务指令]
    你是一个顶级的AI学术研究员...
    """
    QA_PROMPT = PromptTemplate(
        template=qa_template, input_variables=["context", "question"]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    print("--- RAG流水线构建完成 ---")
    
    # 【最终修正点 1】: 返回所有需要的信息，而不仅仅是chain
    return rag_chain, paper, local_pdf_path

# --- Session State and App Flow (无需改动) ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'search'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

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
                st.rerun()
    if st.button("返回搜索"):
        st.session_state.pop('search_results', None)
        st.session_state.stage = 'search'
        st.rerun()

elif st.session_state.stage == 'chat':
    paper_id = st.session_state.selected_paper_id

    if st.session_state.rag_chain is None:
        with st.status(f"正在准备与论文 {paper_id} 的对话环境...", expanded=True) as status:
            try:
                # 【最终修正点 2】: 接收所有返回值，并全部存入session_state
                rag_chain, paper_metadata, downloaded_pdf_path = setup_pipelines(paper_id)
                st.session_state.rag_chain = rag_chain
                st.session_state.paper_metadata = paper_metadata
                st.session_state.downloaded_pdf_path = downloaded_pdf_path
                
                status.update(label=" C 环境准备完成！", state="complete", expanded=False)

            except Exception as e:
                status.update(label=f"环境准备失败: {e}", state="error")
                st.session_state.stage = 'select'
                if st.button("返回重试"):
                    st.rerun()

    if st.session_state.rag_chain:
        st.header(f"3. 正在与论文对话: {st.session_state.paper_metadata.title}")

        with open(st.session_state.downloaded_pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 下载当前论文PDF",
                data=pdf_file,
                file_name=os.path.basename(st.session_state.downloaded_pdf_path),
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
                result = st.session_state.rag_chain({"question": user_question})
                ai_response = result['answer']
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                    with st.expander("查看本次回答引用的原文片段"):
                        for doc in result.get('source_documents', []):
                            st.markdown(
                                f"> {doc.page_content}\n\n_(来源: PDF 第 {doc.metadata.get('page', 'N/A')} 页)_")

    if st.button(" C 返回论文选择列表"):
        st.session_state.stage = 'select'
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.session_state.pop('selected_paper_id', None)
        st.session_state.pop('paper_metadata', None)
        st.session_state.pop('downloaded_pdf_path', None)
        st.rerun()
