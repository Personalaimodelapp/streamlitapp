import streamlit as st
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

import time
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun

class ThrottledDuckDuckGoSearch(DuckDuckGoSearchRun):
    def run(self, query: str,**kwargs) -> str:
        time.sleep(2)  # 2-second delay to avoid rate limits
        return super().run(query)

load_dotenv()

#groq_api_key = os.getenv('GROQ_API_KEY')


arxiv_wrapper = ArxivAPIWrapper(top_k_result = 1,doc_content_chars_max = 200)
arxiv = ArxivQueryRun(api_wrapper =arxiv_wrapper)


wiki_wrapper = WikipediaAPIWrapper(top_k_result = 1, doc_content_chars_max =200)
wiki = WikipediaQueryRun(api_wrapper= wiki_wrapper)


search = ThrottledDuckDuckGoSearch(name= 'search',handle_parsing_errors=True)

st.title("Langchain-  chat with search")

st.sidebar.title("settings")
groq_api_key = st.sidebar.text_input("Enter your Groq API key: ", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {'role':"assistant","content":"Hi, I'm chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:= st.chat_input(placeholder="What is machine learning"):
    st.session_state.messages.append({"role":"user","content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key= groq_api_key, model_name ="Llama3-8b-8192",streaming = True)
    tools =[search,arxiv,wiki]

    search_agent =initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors =True)


    with st.chat_message('assitant'):
        st_cb =StreamlitCallbackHandler(st.container(),expand_new_thoughts= False)
        response = search_agent.run(st.session_state.messages,callbacks= [st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)

