import streamlit as st
import os
import arxiv
import requests
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
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import List, Optional, Any

# Load environment variables. On Streamlit Cloud, this reads from Secrets.
load_dotenv()

# --- 0. Minimal HF Inference API LLM Wrapper (to avoid InferenceClient.post issues) ---
class HfInferenceLLM(LLM):
    """Lightweight LLM using Hugging Face Inference API via requests.

    This avoids version-mismatch issues around huggingface_hub's InferenceClient.post.
    """

    repo_id: str
    temperature: float = 0.3
    max_new_tokens: int = 2048
    timeout: float = 60.0

    @property
    def _llm_type(self) -> str:
        return "hf-inference-api"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError(
                "Missing HUGGINGFACEHUB_API_TOKEN. Please set it in environment/Secrets."
            )

        url = f"https://api-inference.huggingface.co/models/{self.repo_id}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            if resp.status_code == 404:
                raise RuntimeError(
                    f"HF Inference API 404: 模型未找到或未启用推理API -> {self.repo_id}. "
                    "请确认模型ID正确，或在Secrets中设置 HF_MODEL_ID 指向可用模型。"
                )
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"HF Inference API request failed: {e}")

        try:
            data = resp.json()
        except Exception:
            return resp.text

        # HF responses could be list[{'generated_text': ...}] or dict with error
        if isinstance(data, list) and data and isinstance(data[0], dict):
            text = data[0].get("generated_text")
            if text is not None:
                return text
        if isinstance(data, dict):
            # Some models return {'generated_text': '...'} directly
            if "generated_text" in data:
                return str(data["generated_text"])  # type: ignore
            if "error" in data:
                raise RuntimeError(f"HF Inference API error: {data['error']}")

        # Fallback to best-effort string conversion
        return str(data)

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

        # 使用自定义的 HF Inference API 包装器以规避 InferenceClient.post 兼容性问题
        # 允许通过环境变量 HF_MODEL_ID 覆盖默认模型；默认选择更易于在免费Inference API上可用的较小模型
        selected_model = os.getenv("HF_MODEL_ID", "mistralai/Mistral-7B-Instruct-v0.2")
        llm = HfInferenceLLM(
            repo_id=selected_model,
            temperature=0.3,
            max_new_tokens=2048,
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
        st.caption(f"当前模型: {selected_model}")
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
                            st.markdown(
                                f"> {doc.page_content}\n\n_(来源: PDF 第 {doc.metadata.get('page', 'N/A')} 页)_")

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


