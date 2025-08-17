from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_pinecone import PineconeVectorStore

# 자동 히스토리 관리용
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv() 
# 세션 히스토리 저장소
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    index_name = "tax-index-markdown"
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    # 필요 시 k 조정
    retriever = database.as_retriever(search_kwargs={"k": 1})
    return retriever


def get_llm():
    llm = ChatUpstage()
    return llm


def get_dictionary_chain():
    llm = get_llm()
    dictionary = ["사람을 나타내는 표현은 모두 거주자로 변경해주세요"]

    prompt = ChatPromptTemplate.from_template(
        f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        변경할 필요가 없다고 판단되면, 질문 원문을 그대로 반환하세요.

        추가로 초과 금액은 단순하게 사용자가 제시한 금액에서 기준 금액을 뺀 값으로 계산해주세요.

        사전: {dictionary}
        질문: {{question}}
        """
        )

    return prompt | llm | StrOutputParser()

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_qa_chain():
    llm = get_llm()
    history_aware_retriever = get_history_retriever()

    # 2) 최종 답변 프롬프트 (stuff) – {context} + {input}
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "너는 소득세 도우미야. 아래 컨텍스트만 사용해 간결하고 정확히 답해.\n"
                "<context>\n{context}\n</context>",
            ),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ]
    )
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=answer_prompt)

    # 3) 최종 RAG 체인  (⚠️ 인자명: combine_docs_chain)
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain,
    )
    return rag_chain


def get_ai_message(user_question: str, session_id: str = "abc123"):
    """
    - 사전 변환 → 히스토리 인지형 RAG 스트리밍
    - 문자열 제너레이터를 반환 (Streamlit의 st.write_stream에 바로 전달 가능)
    """
    # 1) 질문 정규화
    normalized_q = get_dictionary_chain().invoke({"question": user_question})

    # 2) 체인 + 자동 히스토리 래핑
    rag_chain = get_qa_chain()
    rag_with_history = RunnableWithMessageHistory(
        rag_chain,                            # 실제 실행할 체인 (여기서는 RAG 체인)
        get_session_history,                  # 세션별 히스토리 불러오는 함수
        input_messages_key="input",           # 입력 프롬프트에서 사용자가 친 질문이 담기는 키
        history_messages_key="chat_history",  # 대화 히스토리(대화 로그)를 넘길 때 쓰는 키
        output_messages_key="answer",         # 체인 실행 후 나오는 답변이 저장되는 키
    )


    # 3) 'answer'만 스트리밍
    return rag_with_history.pick("answer").stream(
        {"input": normalized_q},
        config={"configurable": {"session_id": session_id}},
    )