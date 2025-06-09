import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import AsyncChromiumLoader, SitemapLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

url = "https://developers.cloudflare.com/sitemap-0.xml"

answers_prompt = ChatPromptTemplate.from_template("""
    Using ONLY the following context answer the user's question. If you can't
    just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5. 0 being not helpful to
    the user and 5 being helpful to the user.
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
                                                  
    Context: {context}
    Question: {question}
""")

def get_answers(inputs):
    docs = inputs['docs']
    question = inputs['question']
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #         "question": question,
    #         "context": doc.page_content
    #     })
    #     answers.append(result.content)
    # st.write(answers)
    return {
        "question": question,
        "answers": [
            {
            "answer": answers_chain.invoke(
                {"question": question, "context": doc.page_content,}
            ).content,
            "source": doc.metadata["source"],
            "date": doc.metadata["lastmod"],
            } for doc in docs
        ]
    }

choose_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    Use ONLY the following pre-existing answers to answer the user's question.

    Use the answers that have the highest score (more helpful) and
    favor the most recent ones.

    Cite sources. Do not modify the source, keep it as a link.

    Answers: {answers}
    """
    ),
    ("human", "{question}")
])

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(f"{answer['answer']}\nSource: {answer['source']}\nDate: {answer['date']}\n" for answer in answers)
    return choose_chain.invoke({
        "question": question,
        "answers": condensed,
    })

def parse_page(soup):
    header = soup.find("header")
    nav = soup.find("nav")
    aside = soup.find("aside")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if nav:
        nav.decompose()
    if aside:
        aside.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")

@st.cache_resource(show_spinner="Loading website...")
def load_website(url, api_key):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/ai-gateway\/).*",
            r"^(.*\/vectorize\/).*",
            r"^(.*\/workers-ai\/).*",
        ],
        parsing_function=parse_page,
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    return vector_store.as_retriever()

st.set_page_config(
    page_title="SiteGPT",
    page_icon="💻"
)

html2text_transformer = Html2TextTransformer()

st.markdown("""
    # SiteGPT

    Ask questions about the content of Cloudflare.
            
    Start by uploading your Google API Key on the sidebar.
""")

with st.sidebar:
    api_key = st.text_input("Your Google API Key")

if api_key:
    llm = ChatGoogleGenerativeAI(
        model= "gemini-1.5-flash",
        google_api_key= api_key,
        temperature= 0.1,
    )

    retriever = load_website(url, api_key)
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = {
            "docs": retriever,
            "question": RunnablePassthrough(),
        } | RunnableLambda(get_answers) | RunnableLambda(choose_answer)

        result = chain.invoke(query)
        st.write(result.content)