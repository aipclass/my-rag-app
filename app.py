import streamlit as st
import os
import arxiv
# [Cloud Change]: 导入 HuggingFaceHub 以便通过API调用模型
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- 部署准备: 加载环境变量 ---
# 在本地，这会加载 .env 文件。在Streamlit Cloud上，它会读取您设置的Secrets。
load_dotenv()

# --- 1. 页面配置 ---
st.set_page_config(page_title="AI论文搜索与问答机器人", page_icon=" C", layout="wide")
st.title(" C AI论文搜索与问答机器人")
st.write("在这里，您可以搜索arXiv上的论文，并与选定的论文进行智能对话。")

# --- 2. 文件夹路径定义 ---
# 在云端临时文件系统中使用一个简单的文件夹名即可
PDF_SAVE_PATH = "../project/downloaded_papers"
if not os.path.exists(PDF_SAVE_PATH):
    os.makedirs(PDF_SAVE_PATH)


# --- 3. 核心功能函数 (缓存以提高性能) ---
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
    print(f"--- 正在从Hub加载Embedding模型: {embedding_model_name} ---")
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)
    print("--- Embedding模型加载完毕 ---")

    loader = PyMuPDFLoader(local_pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("--- 正在为论文创建向量索引... ---")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("--- 向量索引创建完成！ ---")

    retriever = vectorstore.as_retriever(search_kwargs={'k': 6})

    # ===================================================================
    # --- START DIAGNOSTIC BLOCK ---
    # ===================================================================
    st.info("--- 正在执行LLM初始化诊断 ---")

    # 1. 检查环境变量是否被应用成功读取
    api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if api_token:
        st.success("✅ 诊断 1/3: 成功从Secrets中读取到 HUGGINGFACEHUB_API_TOKEN。")
        # 为安全起见，只显示部分token
        st.write(f"Token 片段: `{api_token[:5]}...{api_token[-5:]}`")
    else:
        st.error("🚨 诊断 1/3: 关键失败！未能从Secrets中读取到 HUGGINGFACEHUB_API_TOKEN！请检查Secrets的名称拼写。")
        st.stop()  # 如果没有token，直接停止运行

    # 2. 尝试初始化HuggingFaceHub对象
    repo_id = "Qwen/Qwen1.5-7B-Chat"
    llm = None  # 先声明变量
    try:
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.3, "max_length": 2048}
        )
        st.success("✅ 诊断 2/3: HuggingFaceHub 对象初始化成功！")
        st.write(f"LLM 对象类型: `{type(llm)}`")
    except Exception as e:
        st.error(f"🚨 诊断 2/3: 关键失败！在初始化 HuggingFaceHub 对象时发生错误: {e}")
        st.stop()

    # 3. 检查内部客户端是否存在 (AttributeError的直接原因)
    if hasattr(llm, 'client') and llm.client is not None:
        st.success("✅ 诊断 3/3: 内部 `llm.client` 对象存在且不为空。")
    else:
        st.warning("⚠️ 诊断 3/3: 警告！内部 `llm.client` 对象缺失或为空！这可能是版本不兼容导致的。")

    st.info("--- LLM初始化诊断结束 ---")
    # ===================================================================
    # --- END DIAGNOSTIC BLOCK ---
    # ===================================================================

    qa_template = """[任务指令]
    你是一个顶级的AI学术研究员...
    """  # (模板内容保持不变)
    QA_PROMPT = PromptTemplate.from_template(qa_template)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True
    )

    print("--- RAG流水线构建完成 ---")
    return rag_chain

# --- Session State and App Flow (这部分代码和您本地成功运行的版本完全一样，无需改动) ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'search'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

if st.session_state.stage == 'search':
    st.header("1. 搜索论文")
    query = st.text_input("输入您想搜索的论文关键词 (例如: 'large language model')", key="search_query")
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
                status.write(" C 正在下载论文PDF...")
                client = arxiv.Client()
                search = arxiv.Search(id_list=[paper_id])
                paper_metadata = next(client.results(search))
                pdf_filename = f"{paper_metadata.entry_id.split('/')[-1]}.pdf"
                downloaded_pdf_path = os.path.join(PDF_SAVE_PATH, pdf_filename)
                if not os.path.exists(downloaded_pdf_path):
                    paper_metadata.download_pdf(dirpath=PDF_SAVE_PATH, filename=pdf_filename)
                st.session_state.paper_metadata = paper_metadata
                st.session_state.downloaded_pdf_path = downloaded_pdf_path

                status.write(f" C 正在构建RAG流水线...")
                st.session_state.rag_chain = setup_pipelines(paper_id)

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
