from openai import OpenAI
import os
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from getpass import getpass

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass("OpenAI API key: ")

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  
)

dataset_name = "siulhin-vlad37/final"
page_content_column = "text"

loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

data = loader.load()

vectorstore = Chroma.from_documents(documents=data, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.get_relevant_documents(
    "What are the risks of cryptocurrency?"
)

print(retrieved_docs[0].page_content)

llm = ChatOpenAI(model_name="gpt-4", temperature=0)

prompt = hub.pull("rlm/rag-prompt")

print(
    prompt.invoke(
        {"context": "filler context", "question": "filler question"}
    ).to_string()
)

def format_docs(data):
    return "\n\n".join(doc.page_content for doc in data)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is Crypto Regulation?"):
    print(chunk, end="", flush=True)

condense_q_system_prompt = """Given a chat history and the most recent user query \
which which may relate to the chat history, formulate a standalone question \
without needing background from the chat history. Do NOT provide an answer, \
just reformulate it if necessary and otherwise return it as is."""
condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
condense_q_chain = condense_q_prompt | llm | StrOutputParser()

condense_q_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What is ICO?"),
            AIMessage(content="Initial Coin Offering"),
        ],
        "question": "What is meant by Initial",
    }
)

condense_q_chain.invoke(
    {
        "chat_history": [
            HumanMessage(content="What is ICO?"),
            AIMessage(content="Initial Coin Offering"),
        ],
        "question": "How do STO work",
    }
)

qa_system_prompt = """As an assistant tasked with responding to queries, \
utilize the given context to formulate an answer. \
If the answer is unknown, state your lack of knowledge. \
Provide a brief response, limiting it to no more than three sentences.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def condense_question(input: dict):
    if input.get("chat_history"):
        return condense_q_chain
    else:
        return input["question"]

rag_chain = (
    RunnablePassthrough.assign(context=condense_question | retriever | format_docs)
    | qa_prompt
    | llm
)

chat_history = []
while True:
    user_question = input("\nUser: ")
    if user_question.lower() in ["exit", "quit", "stop"]:
        print("\nExiting chatbot.")
        break

    ai_response = rag_chain.invoke({"question": user_question, "chat_history": chat_history})
    print("\nChatbot:", ai_response.content)

    chat_history.extend([HumanMessage(content=user_question), AIMessage(content=ai_response.content)])