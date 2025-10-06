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

# [结构性修复 1]: 创建一个只负责数据处理和向量化的缓存函数
# 这个函数返回的对象（retriever, paper, path）都是可以被安全缓存的
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

# --- Session State and App Flow ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'search'
if 'messages' not in st.session_state:
    st.session_state.messages = []
# [结构性修复 2]: 我们不再在session_state中缓存整个chain，只在需要时构建
# if 'rag_chain' not in st.session_state:
#     st.session_state.rag_chain = None

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

    # [结构性修复 3]: 构建完整的对话界面
    try:
        # 1. 获取缓存好的数据处理器（retriever）和元数据
        with st.spinner("正在准备论文数据..."):
            retriever, paper_metadata, downloaded_pdf_path = get_retriever_and_metadata(paper_id)
        st.success("环境准备完成!")

        # 2. 在这里，每次进入对话时，都创建一个新的、带有“鲜活”网络连接的LLM和Chain
        # 这个创建过程非常快，不会影响用户体验
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
        # 为本次对话创建一个新的记忆和RAG链
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key='answer'
        )
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True
        )

        # 3. 显示对话界面
        st.header(f"3. 正在与论文对话: {paper_metadata.title}")
        with open(downloaded_pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 下载当前论文PDF",
                data=pdf_file,
                file_name=os.path.basename(downloaded_pdf_path),
                mime="application/octet-stream"
            )
        st.divider()

        # 4. 处理对话逻辑
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("请就这篇论文提问：")
        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.spinner("模型正在检索与思考中..."):
                # 使用刚刚创建的、鲜活的rag_chain
                result = rag_chain({"question": user_question, "chat_history": st.session_state.messages[:-1]})
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
            st.session_state.pop('selected_paper_id', None)
            st.rerun()

    except Exception as e:
        st.error(f"环境准备失败: {e}")
        if st.button("返回重试"):
            st.rerun()
