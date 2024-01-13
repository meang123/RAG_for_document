import os
import cohere
from typing import List, Optional


from pydantic import BaseModel, Field, validator

from langchain.vectorstores import Chroma


from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.schema.output_parser import StrOutputParser

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.retrievers.document_compressors import EmbeddingsFilter
from typing import ForwardRef
from pydantic import BaseModel
from cohere import Client
from langchain.retrievers.document_compressors import CohereRerank
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_extraction_chain_pydantic
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from pathlib import Path
from langchain.output_parsers import PydanticOutputParser
from langchain import PromptTemplate
from typing import Sequence
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.document_loaders import DirectoryLoader

# agent chat model + conversationMemory buffer
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from operator import itemgetter
from langchain.load.dump import dumps
from langchain.agents import Tool, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate

from langchain import OpenAI, LLMChain
from langchain.tools import DuckDuckGoSearchRun
from langchain import hub
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
import langchain
import ast

from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.agent_toolkits import create_retriever_tool

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chains import LLMSummarizationCheckerChain
from langchain.llms import OpenAI

from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.chains import LLMSummarizationCheckerChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import urllib.request

import urllib
import logging
from io import StringIO

import time

client_id = "qxlmu9xgv1" # 개발자센터에서 발급받은 Client ID 값
client_secret = "2zts3jk9K1a35OFtKVU6x1HRWyRs3R0qrAEu2f5m" # 개발자센터에서 발급받은 Client Secret 값
url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"

# "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"

os.environ["OPENAI_API_KEY"] = "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"


# os.environ["COHERE_API_KEY"]="kVLAwLqUGmxDNGwD6l13gRBaTswRPnXUPMuoPK8S"

from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-base-en"
encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity

model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)
o_llm = ChatOpenAI(temperature=0)
# from llama_hub.file.pdf.base import PDFReader
loader = DirectoryLoader('./data/', glob="./*.pdf", loader_cls=PyPDFLoader)
#PyPDFLoader
documents = loader.load()

#splitting the text into
pdf_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=120)
texts = pdf_text_splitter.split_documents(documents)

#cromdb 생성 및 임베딩 사용
persist_directory = 'together_db'

## Here is the nmew embeddings being used
embedding = model_norm

vectordb = Chroma.from_documents(documents=texts,
                                 embedding=embedding,
                                 persist_directory=persist_directory)
vectordb.persist()
vectordb = None
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 5}) # default

# for retrival pipeline


co=Client(os.environ["COHERE_API_KEY"])

#compressor = CustomCohereRerank(top_n=3,user_agent='langchain')#CohereRerank(top_n=2)
compressor = CohereRerank(top_n=3)
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
relevant_filter = EmbeddingsFilter(embeddings=embedding,similarity_threshold=0.71)

pipeline_compressor = DocumentCompressorPipeline(
    transformers=[compressor,relevant_filter,redundant_filter]
)



compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

# retrival + output parser : Name : ,,., Content:....
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers.multi_query import MultiQueryRetriever


multi_qa = MultiQueryRetriever.from_llm(retriever=compression_retriever,llm=o_llm)

class ListHandler(logging.Handler):
    def __init__(self, log_list):
        super().__init__()
        self.log_list = log_list

    def emit(self, record):
        self.log_list.append(self.format(record))

def papago(content,data):

    korText = urllib.parse.quote(content)
    time.sleep(0.001)
    
    data = data + korText

    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
    request.add_header("X-NCP-APIGW-API-KEY", client_secret)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()

    if (rescode == 200):

        response_body = response.read()
        time.sleep(0.001)

        return response_body.decode('utf-8')

    else:
        print("Error Code:" + rescode)



def generate_queries(question):
    
    text = multi_qa.generate_queries(question=question, run_manager = CallbackManagerForRetrieverRun(run_id = 'runid', handlers = [],inheritable_handlers = [] ) )
    
    return text 



search = DuckDuckGoSearchRun()

retrival_tool = create_retriever_tool(
    compression_retriever,
    "compression_retriever",
    "This is the pipline for retrival document.",
)
search_tool = Tool(
    name="duckduckgo Search",
    func=search.run,
    description="useful for when you need to answer questions about current events or the current state of the world",
)

tools = [retrival_tool,search_tool]
prompt = hub.pull("hwchase17/react-chat-json")
chat_model = ChatOpenAI(temperature=0,model="gpt-4-1106-preview",max_tokens=1000)
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])




# We need some extra steering, or the chat model forgets how to respond sometimes


TEMPLATE_TOOL_RESPONSE = """TOOL RESPONSE:
---------------------
{observation}

USER'S INPUT
--------------------

Okay, so what is the response to my last comment? If using information obtained from the tools you must mention it explicitly without mentioning the tool names - I have forgotten all TOOL RESPONSES! Remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else - even if you just want to respond to the user. Do NOT respond with anything except a JSON snippet no matter what!"""

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_messages(
            x["intermediate_steps"], template_tool_response=TEMPLATE_TOOL_RESPONSE
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | chat_model_with_stop
    | JSONAgentOutputParser()
)


memory = ConversationBufferWindowMemory(k=10,memory_key="chat_history", return_messages=True)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

