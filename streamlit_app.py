import streamlit as st
import os
import arxiv
# --- 【修改点 1】: 导入新的HuggingFaceHub和环境变量加载库 ---
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- 部署准备: 加载环境变量 (本地测试用) ---
# 在本地开发时，可以创建一个 .env 文件存放HUGGINGFACEHUB_API_TOKEN
# 在Streamlit Cloud部署时，则需要配置Secrets
load_dotenv()

# --- 1. 页面配置 ---
st.set_page_config(page_title="arXiv RAG 问答机器人 (云端版)", page_icon="☁️", layout="wide")
st.title("☁️ arXiv RAG 问答机器人 (云端版)")
st.write("本应用结合云端Qwen模型API与RAG技术，允许您与指定的arXiv论文进行多轮对话。")

# --- 2. 文件夹路径定义 ---
PDF_SAVE_PATH = "./downloaded_papers"
if not os.path.exists(PDF_SAVE_PATH):
    os.makedirs(PDF_SAVE_PATH)


# --- 3. 核心功能函数 (缓存以提高性能) ---
@st.cache_resource
def setup_pipelines(paper_id="2307.09288"):
    print("--- 正在执行一次性初始化 (此部分将被缓存) ---")

    # --- 下载论文部分 ---
    search = arxiv.Search(id_list=[paper_id])
    paper = next(search.results())
    pdf_filename = f"{paper.entry_id.split('/')[-1]}.pdf"
    local_pdf_path = os.path.join(PDF_SAVE_PATH, pdf_filename)
    if not os.path.exists(local_pdf_path):
        paper.download_pdf(dirpath=PDF_SAVE_PATH, filename=pdf_filename)

    # --- 嵌入与文档处理 ---
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_name)

    loader = PyMuPDFLoader(local_pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # --- 【修正点】: 将创建retriever的代码合并为正确的一行 ---
    retriever = vectorstore.as_retriever(search_kwargs={'k': 6})

    # --- LLM调用部分 ---
    repo_id = "Qwen/Qwen1.5-7B-Chat"
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.3, "max_length": 2048}
    )

    # --- Prompt和Chain的其余部分 ---
    qa_template = """[任务指令]
    你是一个顶级的AI学术研究员，你的任务是基于下方提供的“[论文相关内容]”，以一种深刻、专业且富有洞察力的口吻，详细回答用户的“[问题]”。

    [角色扮演要求]
    - **身份**: 你已经精读并完全理解了这篇论文。你的回答应该体现出这一点，避免使用“根据提供的资料...”这类拉开距离的表述。
    - **语气**: 专业、自信、分析性强。像一个真正的专家在进行学术交流。
    - **知识范围**: 你的所有回答必须严格来源于下方提供的“[论文相关内容]”。绝对禁止使用任何外部知识或进行无根据的猜测。

    [回答构建指南]
    1.  **对于总结/分析类问题**: 请综合所有相关内容片段，提炼出核心论点、方法和结论，并用自己的话进行有条理的组织和阐述。
    2.  **对于事实查询类问题**: 请直接从内容中定位并抽取出具体的数字、名称或事实，并清晰地呈现出来。
    3.  **约束条件**: 如果内容片段确实无法支撑回答，就直截了当地说：“这篇论文的相关部分未讨论此问题。”

    ---
    [论文相关内容]:
    {context}
    ---

    [问题]: {question}

    [你的专家级分析回答]:
    """
    QA_PROMPT = PromptTemplate.from_template(qa_template)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=True
    )

    print("--- 初始化完成 ---")
    return rag_chain, paper, local_pdf_path

# --- 【修改点 3】: 使用Streamlit Secrets获取API令牌 ---
# 检查API令牌是否存在，提供更友好的提示
if 'HUGGINGFACEHUB_API_TOKEN' not in os.environ and 'HUGGINGFACEHUB_API_TOKEN' not in st.secrets:
    st.error("部署错误：Hugging Face API令牌未配置！请在Streamlit Cloud的Secrets中设置它。")
    st.image("https://i.imgur.com/i2uCIgV.png")  # 添加一个示例图片指导用户
    st.stop()

# --- 主程序和界面显示代码保持不变 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("正在初始化问答机器人，首次启动可能需要一些时间..."):
    try:
        rag_chain, paper_metadata, downloaded_pdf_path = setup_pipelines()
        st.success("初始化完成！现在可以开始提问了。")
    except Exception as e:
        st.error(f"初始化失败，错误信息: {e}")
        st.stop()

st.subheader("当前论文信息")
st.markdown(
    f"**标题:** {paper_metadata.title}\n\n**作者:** {', '.join(author.name for author in paper_metadata.authors)}\n\n**发布日期:** {paper_metadata.published.strftime('%Y-%m-%d')}")
with open(downloaded_pdf_path, "rb") as pdf_file:
    st.download_button(label="📥 下载论文PDF", data=pdf_file, file_name=os.path.basename(downloaded_pdf_path),
                       mime="application/octet-stream")
st.divider()

st.subheader("对话记录")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_question = st.chat_input("请就这篇论文提问：")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("模型正在检索与思考中..."):
        try:
            result = rag_chain({"question": user_question})
            ai_response = result['answer']

            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            with st.chat_message("assistant"):
                st.markdown(ai_response)
                with st.expander("查看本次回答引用的原文片段"):
                    if 'source_documents' in result and result['source_documents']:
                        for i, doc in enumerate(result['source_documents']):
                            st.markdown(
                                f"**片段 {i + 1} (来自 PDF 第 {doc.metadata.get('page', 'N/A')} 页):**\n> {doc.page_content}")
                    else:
                        st.write("警告：本次回答未能从论文中检索到任何相关的原文片段。")
        except Exception as e:
            st.error(f"获取回答时发生错误: {e}")