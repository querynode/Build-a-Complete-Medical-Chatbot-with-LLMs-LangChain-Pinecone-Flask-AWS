from flask import Flask, render_template, request

from src.helper import download_hugging_face_embeddings

from langchain_pinecone import PineconeVectorStore

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

from src.prompt import system_prompt

from huggingface_hub import InferenceClient

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from typing import Any, List

import os


# Flask application
app = Flask(__name__)

load_dotenv()


# API Keys
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


# Hugging Face Chat Model

class HuggingFaceChatModel(BaseChatModel):

    client: Any
    model_name: str = "openai/gpt-oss-20b"
    max_tokens: int = 512
    temperature: float = 0.1

    @property
    def _llm_type(self) -> str:
        return "huggingface-gpt-oss-20b"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:

        hf_messages = []

        for message in messages:

            if message.type == "system":
                role = "system"

            elif message.type == "human":
                role = "user"

            elif message.type == "ai":
                role = "assistant"

            else:
                role = "user"

            hf_messages.append(
                {
                    "role": role,
                    "content": message.content
                }
            )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=hf_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        content = response.choices[0].message.content

        message = AIMessage(content=content)

        generation = ChatGeneration(message=message)

        return ChatResult(
            generations=[generation]
        )


# Download Hugging Face Embeddings

embeddings = download_hugging_face_embeddings()


# Connect to Pinecone

index_name = "medicalbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# Create Retriever

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# Create Hugging Face LLM

client = InferenceClient(
    api_key=HF_TOKEN
)

chatModel = HuggingFaceChatModel(
    client=client,
    model_name="openai/gpt-oss-20b",
    max_tokens=512,
    temperature=0.1
)


# Create Prompt

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# Create RAG Chain

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# Home Route

@app.route("/")
def index():
    return render_template("chat.html")


# Chat Route

@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    input = msg

    print(input)

    response = rag_chain.invoke(
        {
            "input": msg
        }
    )

    print("Response : ", response["answer"])

    return str(response["answer"])


# Run Flask Application

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )      