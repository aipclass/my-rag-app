import streamlit as st
import os
import arxiv
from dotenv import load_dotenv
import requests

# --- 直接用官方 SDK（zhipuai>=2.x），不再依赖 langchain_glm 以减少安装问题 ---
try:
    from zhipuai import ZhipuAI  # type: ignore
except Exception:
    ZhipuAI = None

# --- LangChain 核心组件 ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from typing import List, Optional, Any

# 加载环境变量 (在Streamlit Cloud上会自动读取Secrets)
load_dotenv()


# --- 0. 最小HF Inference API封装：作为智谱不可用时的自动回退 ---
class HfInferenceLLM(LLM):
    repo_id: str
    temperature: float = 0.3
    max_new_tokens: int = 2048
    timeout: float = 60.0
    fallback_models: Optional[List[str]] = None

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
            raise RuntimeError("缺少 HUGGINGFACEHUB_API_TOKEN，请在Secrets中配置。")

        url = f"https://api-inference.huggingface.co/models/{self.repo_id}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code == 404:
            fallback_env = os.getenv("HF_FALLBACK_MODELS")
            fallbacks = (
                [m.strip() for m in fallback_env.split(",") if m.strip()]
                if fallback_env
                else (self.fallback_models or [
                    "google/flan-t5-base",
                    "google/flan-t5-small",
                    "google/flan-t5-large",
                ])
            )
            for fb in fallbacks:
                fb_url = f"https://api-inference.huggingface.co/models/{fb}"
                fb_resp = requests.post(fb_url, headers=headers, json=payload, timeout=self.timeout)
                if fb_resp.ok:
                    self.repo_id = fb
                    try:
                        data = fb_resp.json()
                    except Exception:
                        return fb_resp.text
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        text = data[0].get("generated_text")
                        if text is not None:
                            return text
                    if isinstance(data, dict) and "generated_text" in data:
                        return str(data["generated_text"])  # type: ignore
                    return str(data)
            raise RuntimeError("HF Inference API 404：选定与回退模型均不可用。")

        if resp.status_code == 429:
            raise RuntimeError("HF Inference API 429：频率限制，请稍后再试。")

        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            return resp.text
        if isinstance(data, list) and data and isinstance(data[0], dict):
            text = data[0].get("generated_text")
            if text is not None:
                return text
        if isinstance(data, dict):
            if "generated_text" in data:
                return str(data["generated_text"])  # type: ignore
            if "error" in data:
                raise RuntimeError(f"HF Inference API error: {data['error']}")
        return str(data)


# --- 1. 页面配置 ---
st.set_page_config(page_title="AI论文搜索与问答机器人", page_icon=" C", layout="wide")
st.title(" C AI论文搜索与问答机器人")
st.write("在这里，您可以搜索arXiv上的论文，并与选定的论文进行智能对话。")

# --- 2. 定义路径 ---
PDF_SAVE_PATH = "downloaded_papers"
if not os.path.exists(PDF_SAVE_PATH):
    os.makedirs(PDF_SAVE_PATH)


# --- 3. 缓存的数据处理函数 (核心功能不变) ---
@st.cache_resource
def get_retriever_and_metadata(_paper_id):
    """
    下载论文PDF，加载、切分、向量化，并创建检索器。
    利用Streamlit的缓存避免重复计算。
    """
    print(f"--- [Cache Miss] 正在为论文 {_paper_id} 构建检索器 ---")
    client = arxiv.Client()
    search = arxiv.Search(id_list=[_paper_id])
    paper = next(client.results(search))

    # 使用健壮的方式获取 arXiv 短ID，兼容旧式ID（如 "cs/0506025"）与新式ID
    try:
        short_id = paper.get_short_id()  # type: ignore[attr-defined]
    except Exception:
        short_id = paper.entry_id.split('abs/')[-1]
    pdf_filename = f"{short_id}.pdf"
    local_pdf_path = os.path.join(PDF_SAVE_PATH, pdf_filename)

    # 如果本地不存在PDF，则下载
    if not os.path.exists(local_pdf_path):
        paper.download_pdf(dirpath=PDF_SAVE_PATH, filename=pdf_filename)

    # 1. 加载文档
    loader = PyMuPDFLoader(local_pdf_path)
    docs = loader.load()

    # 2. 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. 向量化并创建FAISS索引（改为 fastembed，避免安装大型PyTorch 依赖）
    embeddings = FastEmbedEmbeddings()
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    # 4. 创建检索器
    retriever = vectorstore.as_retriever(search_kwargs={'k': 6})

    return retriever, paper, local_pdf_path


# --- Session State 初始化 ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'search'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )

# --- 选择论文的回调，避免循环变量绑定问题，确保点击哪一项就进入哪一项 ---
def _on_select_paper(paper_id: str) -> None:
    # 选择新论文时清理缓存，避免因缓存键异常导致始终展示同一篇论文
    try:
        get_retriever_and_metadata.clear()
    except Exception:
        pass
    st.session_state.selected_paper_id = paper_id
    st.session_state.stage = 'chat'
    st.session_state.messages = []
    st.session_state.memory.clear()

# --- 应用流程控制 ---

# ================= 阶段1: 搜索论文 =================
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

# ================= 阶段2: 选择论文 =================
elif st.session_state.stage == 'select':
    st.header("2. 选择一篇论文进行对话")
    if 'search_results' in st.session_state and st.session_state.search_results:
        # 使用可预测的 key 与 on_click 回调，避免 for 循环闭包导致始终取第一项
        for idx, paper in enumerate(st.session_state.search_results):
            # 兼容旧式与新式 arXiv ID，避免因丢失分类前缀导致始终打开同一篇论文
            try:
                paper_id = paper.get_short_id()  # type: ignore[attr-defined]
            except Exception:
                paper_id = paper.entry_id.split('abs/')[-1]
            st.subheader(paper.title)
            st.write(f"**作者**: {', '.join(author.name for author in paper.authors)}")
            st.write(f"**摘要**: {paper.summary[:300]}...")
            st.button(
                " C 与这篇论文对话",
                key=f"select_btn_{idx}_{paper_id}",
                on_click=_on_select_paper,
                kwargs={"paper_id": paper_id},
            )
    if st.button("返回搜索"):
        st.session_state.pop('search_results', None)
        st.session_state.stage = 'search'
        st.rerun()

# ================= 阶段3: 与论文对话 =================
elif st.session_state.stage == 'chat':
    paper_id = st.session_state.get('selected_paper_id')
    if not paper_id:
        st.warning("未检测到已选择的论文，请先返回列表重新选择。")
        if st.button("返回论文列表"):
            st.session_state.stage = 'select'
            st.rerun()
        st.stop()

    try:
        retriever, paper_metadata, downloaded_pdf_path = get_retriever_and_metadata(paper_id)

        # --- 优先使用智谱AI（官方 SDK + 简单适配器）；若不可用则回退 HF 免费模型 ---
        llm = None
        current_model_label = ""
        zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
        zhipu_model = os.getenv("ZHIPU_MODEL", "glm-4-flash")
        if ZhipuAI is not None and zhipuai_api_key:
            class ZhipuLLMAdapter(LLM):
                model: str
                api_key: str
                temperature: float = 0.3

                @property
                def _llm_type(self) -> str:
                    return "zhipu-adapter"

                def _call(self, prompt: str, stop=None, run_manager=None, **kwargs):
                    client = ZhipuAI(api_key=self.api_key)
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                    )
                    # 兼容 SDK 返回
                    try:
                        return resp.choices[0].message.content  # type: ignore
                    except Exception:
                        return str(resp)


            llm = ZhipuLLMAdapter(model=zhipu_model, api_key=zhipuai_api_key, temperature=0.3)
            current_model_label = f"ZhipuAI {zhipu_model}"
        if llm is None:
            # 回退：使用HF免费模型
            selected_model = os.getenv("HF_MODEL_ID", "google/flan-t5-base")
            llm = HfInferenceLLM(
                repo_id=selected_model,
                temperature=0.3,
                max_new_tokens=512,
                fallback_models=[
                    "google/flan-t5-base",
                    "google/flan-t5-small",
                    "google/flan-t5-large",
                ]
            )
            current_model_label = f"HF {selected_model}"

        # 创建对话检索链 (后续逻辑不变)
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

        # --- 聊天界面 ---
        st.header(f"3. 正在与论文对话: {paper_metadata.title}")

        # 显示当前实际使用的模型
        st.caption(f"当前模型: {current_model_label}")

        with open(downloaded_pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 下载当前论文PDF",
                data=pdf_file,
                file_name=os.path.basename(downloaded_pdf_path),
                mime="application/octet-stream"
            )
        st.divider()

        # 显示历史消息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 接收用户输入
        user_question = st.chat_input("请就这篇论文提问：")
        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # 调用RAG链获取回答
            with st.spinner("模型正在检索与思考中..."):
                result = rag_chain.invoke({"question": user_question})
                ai_response = result['answer']

                st.session_state.messages.append({"role": "assistant", "content": ai_response})

                # 显示AI回答和引用的原文
                with st.chat_message("assistant"):
                    st.markdown(ai_response)
                    with st.expander("查看本次回答引用的原文片段"):
                        for doc in result.get('source_documents', []):
                            st.markdown(
                                f"> {doc.page_content}\n\n_(来源: PDF 第 {doc.metadata.get('page', 'N/A')} 页)_")

        # 返回按钮
        if st.button(" C 返回论文选择列表"):
            st.session_state.stage = 'select'
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.session_state.pop('selected_paper_id', None)
            st.rerun()

    except Exception as e:
        st.error(f"处理对话时发生错误: {e}")
        if st.button("返回重试"):
            # 清理缓存并重试
            get_retriever_and_metadata.clear()
            st.rerun()
