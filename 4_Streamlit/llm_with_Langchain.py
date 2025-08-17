
import os
from langchain_upstage import UpstageEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_upstage import ChatUpstage
from langchain.chains import RetrievalQA
from langchain import hub

load_dotenv()

def get_retriever():
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    index_name = 'tax-index-markdown'
    
    # 데이터베이스가 이미 존재하는 경우 DB불러오기
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    retriever = database.as_retriever(search_kwargs={"k": 1})
    return retriever


def get_llm():
    llm = ChatUpstage()
    return llm


def get_dictionary_chain():
    """
    Create a chain that modifies user questions based on a predefined dictionary.
    """
    llm = get_llm()
    dictionary = ["사람을 나타내는 표현은 모두 거주자로 변경해주세요"]
    prompt = ChatPromptTemplate.from_template(f"""
                            사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
                            변경할 필요가 없다고 판단되면, 사용자 질문을 변경하지 않아도 됩니다.
                            그런 경우에는 질문만 return 해주세요.

                            사전: {dictionary}
                            질문 : {{question}}
                            """)
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()
    
    # 미리 정의된 RAG용 프롬포트를 불러오기
    prompt = hub.pull("rlm/rag-prompt")

    # 직장인으로 입력받는 경우, 거주자로 변경하는 chain을 추가한다.
    qa_chain = RetrievalQA.from_chain_type(
        llm, 
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


def get_ai_message(user_question):
    dictionary_chain = get_dictionary_chain()
    qa_chain = get_qa_chain()

    tax_chain = {"query": dictionary_chain} |  qa_chain
    ai_message = tax_chain.invoke({
        "question": user_question
    })["result"]
    return ai_message
