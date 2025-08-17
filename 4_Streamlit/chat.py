import streamlit as st
# from llm_with_History import get_ai_message
from llm_with_History_Fewshoty import get_ai_message

st.set_page_config(page_title="Chat Application", page_icon="📈")

st.title("소득세 챗봇")
st.caption("소득세 관련 질문을 해보세요!")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 세션에 저장된 메시지를 반복하고 각 메시지 역할에 따라 메시지 표시
print(f"before: {st.session_state.messages}")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input("소득세와 관련된 궁금한 내용들을 말씀해주세요!"):
    # 채팅이 입력되었을 때 상단에 메시지가 노출되도록 함.
    # "user", "assistant", "ai", "human" 가 있는데 그중 user 을 사용함.
    with st.chat_message("user"):
        st.write(user_question)

    # 채팅 메시지를 Session State에 저장
    st.session_state.messages.append({"role": "user", "content": user_question})


    # 로딩중인 부분을 보여주기
    with st.spinner("AI 응답을 생성하는 중..."):
        ai_message = get_ai_message(user_question)

        # with st.chat_message("ai"):
        #     st.write(ai_message)
        #     st.session_state.messages.append({"role": "ai", "content": ai_message})
        with st.chat_message("ai"):
            final_text = st.write_stream(ai_message)
            # 채팅 메시지를 Session State에 저장
    st.session_state.messages.append({"role": "ai", "content": final_text})