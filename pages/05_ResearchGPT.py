import streamlit as st
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper

st.set_page_config(
    page_title="ResearchGPT",
    page_icon="✏"
)

st.title("ResearchGPT")

st.markdown("""
### Welcome!
            
Use this chatbot to ask questions to an AI about your files!
            
Upload your Google API Key!
""")

@tool(description="Search in DuckDuckGo.")
def get_research_ddg(query):
    ddg = DuckDuckGoSearchAPIWrapper()
    return ddg.run(query)

@tool(description="Search in Wikipedia.")
def get_research_wiki(query):
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

tools = [get_research_ddg, get_research_wiki]
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

with st.sidebar:
    api_key = st.text_input("Your Google API Key")

if api_key:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a helpful assistant.
         
        You should try to search in Wikipedia or DuckDuckGo. If you find a website in
        DuckDuckGo, you should enter the website and extract it's content."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    message = st.chat_input("Ask anything to search...")
    if message:
        send_message(message, "human")
        
        with st.chat_message("ai"):
            message_box = st.empty()

            with st.spinner("Searching..."):
                response = agent_executor.invoke({"input": message})
                message_box.markdown(response["output"])
                save_message(response["output"], "ai")
else:
    st.session_state["messages"] = []