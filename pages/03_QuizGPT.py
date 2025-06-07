import streamlit as st
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)

output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    os.makedirs(".cache/quiz_files", exist_ok=True)
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    loader = UnstructuredFileLoader(file_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100
    )
    docs = loader.load_and_split(text_splitter=splitter)
    return docs

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, quiz_diff):
    question_chain = {
        "context": format_docs,
        "difficulty": lambda x: quiz_diff
    } | question_prompt | llm
    chain = {"context": question_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    return retriever.get_relevant_documents(term)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

question_prompt = ChatPromptTemplate.from_template(
    """
        You are a helpful assistant that is role playing as a teacher.

        Based ONLY on the following context make 10 questions to test the user's
        knowledge about the text.

        All questions should be {difficulty} in difficulty.

        Each question should have 4 answers, three of them must be incorrect and
        one should be correct.

        Use (o) to signal the correct answer.

        Question examples:

        Question: What is the color of the ocean?
        Answers: Red | Yellow | Green | Blue (o)

        Question: What is the capital of Georgia?
        Answers: Baku | Tbilisi(o) | Manila | Beirut

        Question: When was Avatar released?
        Answers: 2007 | 2001 | 2009(o) | 1998

        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o) | Painter | Actor | Model

        Your turn!

        Context: {context}
    """)
formatting_prompt = ChatPromptTemplate.from_template(
        """
        You are a powerful formatting algorithm.

        You format exam questions into JSON format.
        Answers with (o) are the correct ones.

        Example Input:

        Question: What is the color of the ocean?
        Answers: Red | Yellow | Green | Blue (o)

        Question: What is the capital of Georgia?
        Answers: Baku | Tbilisi(o) | Manila | Beirut

        Question: When was Avatar released?
        Answers: 2007 | 2001 | 2009(o) | 1998

        Question: Who was Julius Caesar?
        Answers: A Roman Emperor(o) | Painter | Actor | Model

        Example Output:

        ```json
        {{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                    {{
                        "answer": "Red",
                        "correct": false
                    }},
                    {{
                        "answer": "Yellow",
                        "correct": false
                    }},
                    {{
                        "answer": "Green",
                        "correct": false
                    }},
                    {{
                        "answer": "Blue",
                        "correct": true
                    }},
                ]
            }},
            {{
                "question": "What is the capital of Georgia?",
                "answers": [
                    {{
                        "answer": "Baku",
                        "correct": false
                    }},
                    {{
                        "answer": "Tbilisi",
                        "correct": true
                    }},
                    {{
                        "answer": "Manila",
                        "correct": false
                    }},
                    {{
                        "answer": "Beirut",
                        "correct": false
                    }},
                ]
            }},
            {{
                "question": "When was Avatar released?",
                "answers": [
                    {{
                        "answer": "2007",
                        "correct": false
                    }},
                    {{
                        "answer": "2001",
                        "correct": false
                    }},
                    {{
                        "answer": "2009",
                        "correct": true
                    }},
                    {{
                        "answer": "1998",
                        "correct": false
                    }},
                ]
            }},
            {{
                "question": "Who was Julius Caesar?",
                "answers": [
                    {{
                        "answer": "A Roman Emperor",
                        "correct": true
                    }},
                    {{
                        "answer": "Painter",
                        "correct": false
                    }},
                    {{
                        "answer": "Actor",
                        "correct": false
                    }},
                    {{
                        "answer": "Model",
                        "correct": false
                    }},
                ]
            }},
        ]
        }}
        
        ```

        Your turn!
        Questions: {context}
        """
    )

with st.sidebar:
    docs = None
    choice = st.selectbox("Choose what you want to use.", (
        "File", "Wikipedia Article",
    ))

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx, .txt, or .pdf file",
            type= ["pdf", "txt", "docx"],   
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Name of the article")
        if topic:
            docs = wiki_search(topic)

    api_key = st.text_input("Your Google API Key")

if not (docs or api_key):
    st.markdown("""
        Welcome to QuizGPT.
                
        I will make a quiz from Wikipedia articles or files you upload to test
        your knowledge and help you study.
                
        Get started by uploading a file or searching on Wikipedia in the sidebar.
        Also upload your Google API key.
    """)
else:
    quiz_diff = st.selectbox("Choose the difficulty of the quiz.",(
        "Easy", "Hard"
    ))

    llm = ChatGoogleGenerativeAI(
        model= "gemini-1.5-flash",
        google_api_key= api_key,
        temperature= 0.1,
    )
    formatting_chain = formatting_prompt | llm

    response = run_quiz_chain(docs, topic if topic else file.name, quiz_diff)

    with st.form("questions_form"):
        correct = 0
        for question in response["questions"]:
            st.write(question["question"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )

            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct")
                correct += 1
            elif value is not None:
                st.error("Wrong")

        button = st.form_submit_button()

        if correct == 10:
            st.balloons()