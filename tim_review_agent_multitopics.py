#!/usr/bin/env python
# coding: utf-8

# In[1]:


from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import ChatOpenAI
from langsmith import traceable
from langchain.schema import Document
from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal, List, NotRequired, Annotated, TypedDict
from operator import add

import json

from langgraph.constants import Send

from IPython.display import Image, display, Markdown

from dotenv import load_dotenv
load_dotenv()


# In[2]:


from langchain_google_vertexai import VertexAIEmbeddings
embeddings = VertexAIEmbeddings(model="text-embedding-005")


# In[3]:


from langchain_chroma import Chroma
persist_directory = 'docs/chroma_from_web/'
vector_store = Chroma(persist_directory = persist_directory,embedding_function = embeddings)


# In[4]:


vector_store.similarity_search_with_relevance_scores("What's innovative entrepreneurship?", k=5)


# In[5]:


llm = ChatVertexAI(model="gemini-2.0-flash-exp", temperature=0)
#llm = ChatOpenAI(temperature=0, model_name="gpt-4o")


# In[6]:


query_or_respond_system_message = """
Your are an expert on technology innovation management.
You are given a query by the user specified in the user message.
You are to retrieve information related to the query from around 700 articles from the Tim Review Journal.
The retrieval is done using a vector similarity search.
You should decide whether a research is required.
If research is required, handle the query properly.
If more than one topics are involved in the query, no matter how the question is asked, you must divide the query into subtopics. This division rule is very important, take special care of it. 
If there is authors, time, etc. involved, your should specify them in every subquery separately.
If the user does not provide a query related to technology innovation management, you should response with an empty list.
"""

generate_tool_call_system_message = """ 
Generate a tool call based on the query.
Carefully choose the filters agrument to retrieve the information that is most relevant to the query.
The content under the key "author" in metadata contains all the authors of the documents. 
"""

generate_system_message = """ 
Your are an expert on technology innovation management.
You are given a query by the user specified in the user message and some documents retrieved from the Tim Review Journal.
Examine every document carefully and choose choose those you think are relevant to the query. Answer the query based on the chosen documents. 
Generally you should confine your answer to the documents that you have chosen, but you can add information from outside of the documents if the information is considered common sense. 
If the documents do not provide enough information to answer the query, you should clearly state it.
You should use APA 7th edition format in-line citation and generate a reference list at the end of the answer unless the user specifys otherwise.
Generate Markdown formatted answer. 
Use Markdown format for the answer. Use single * for bold and single _ for italics.
When a reference list is generated, make the heading Reference bold.

<query>
{query}
</query>

<documents>
{documents}
</documents>
"""

steer_back_system_prompt = """
Now the user has asked a question that's not related to technology innovation management. Say something to steer the user back on track.
"""


# In[7]:


class DividedQuery(BaseModel):
    query: str = Field(description="Query for the subtopic.")
    
class DividedQueryList(BaseModel):
    divided_queries: List[DividedQuery] = Field(description="A list of queries for the subtopics.") 
    
class RetrieveState(TypedDict):
    subquery: str
    
class RetrieveInput(BaseModel):
    query: str = Field(description="The query to retrieve the answer for in cosine similarity.")
    #filter: Dict[Literal["author", "title", "year", "month"], str | int] = {}
    filter_author: Optional[str] = Field(default=None, description="Filter property, the authors of the document")
    filter_title: Optional[str] = Field(default=None, description="Filter property, the title of the document")
    filter_year: Optional[int] = Field(default=None, description="Filter property, the year of the document. Should be an integer")
    filter_month: Optional[int] = Field(default=None, description="Filter property, the month of the document. Should be an integer")

class ChosenDocument(BaseModel):
    indeces: List[int] = Field(description="A list of the indeces of the documents that are chosen to be used for the answer. The index start from 0.")
    documents: List[str] = Field(description="A list of the strings of the oringinal text of the documents that are chosen to be used for the answer. The index start from 0.")
    
class TimReviewState(MessagesState):
    query: str
    subqueries: Annotated[list, add]


# In[27]:


@traceable
def query_or_respond(state: TimReviewState):
    """Decide whether research is required and rewrite the query if necessary."""
    system_prompt = SystemMessage(content=query_or_respond_system_message)
    human_prompt = HumanMessage(content=state['query'])
    query_llm = llm.with_structured_output(DividedQueryList)
    response = query_llm.invoke([system_prompt, human_prompt])
    return {"subqueries": response.divided_queries}

@traceable
def generate_retrieval_tool_call(state: RetrieveState):
    system_prompt = SystemMessage(content=generate_tool_call_system_message)
    tools = [retrieve]
    retrieve_llm = llm.bind_tools(tools)
    query = state["query"].query
    response = retrieve_llm.invoke([system_prompt] + [HumanMessage(content=query)])

    return {"messages": [response]}

@traceable
def should_research(state: TimReviewState):

    if len(state["subqueries"]):
        return [Send("generate_retrieval_tool_call", {"query": q}) for q in state["subqueries"]]
    else: 
        return "steer_back"


@traceable
def multiple_retrieve(state: TimReviewState):
    tool_calls_messages = []
    for message in reversed(state["messages"]):
        if message.tool_calls:
            tool_calls_messages.append(message)
        else:
            break
    tool_calls_messages = tool_calls_messages[::-1]
    tool_messages = []
    for tool_call in tool_calls_messages:
        tool_result = retrieve.invoke(tool_call.tool_calls[0]["args"])
        tool_message = ToolMessage(
            content=str(tool_result), 
            name="retrieve", 
            tool_call_id=tool_call.tool_calls[0]["id"]
            )
        tool_messages.append(tool_message)
    
    return {"messages": tool_messages}

def steer_back(state: TimReviewState):
    response = llm.invoke([SystemMessage(content=steer_back_system_prompt), HumanMessage(content=" ")])
    return {"messages": [response]}
    
@tool("retrieve", args_schema=RetrieveInput)
def retrieve(query: str, filter_author: str = None, filter_title: str = None, filter_year: int = None, filter_month: int = None):
    """
    Retrieve information from the literatures related to a query.
    
    param query: The query to search for in cosine similarity.
    param filter: The filter to apply to the search. The filter is a dictionary with the following keys:
        - author: The authors of the document
        - title: The title of the document
        - year: The year of the document
        - month: The month of the document. Format: mm/yyyy
        
    return: A list of the retrieved documents    
    
    """

    filter = {}
    if filter_author: filter["author"] = filter_author
    if filter_title: filter["title"] = filter_title
    if filter_year: filter["year"] = filter_year
    if filter_month: filter["month"] = filter_month
    if len(filter.keys()):
        retrieved_docs = vector_store.similarity_search(query, k=5, filter=filter)
    else: retrieved_docs = vector_store.similarity_search(query, k=5)
    print('*************** Retrieval ***************')
    print(f'Query: {query}')
    print(f'Number of documents retrieved: {len(retrieved_docs)}')
    print('Documents:')
    for document in retrieved_docs:
        print(document)
    return retrieved_docs


def generate(state: TimReviewState):
    """Generate answer."""
    # Get generated ToolMessages
    tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            tool_messages.append(message)
        else:
            break
    tool_messages = tool_messages[::-1]
    
    docs = []
    for tool_message in tool_messages:
        if type(tool_message.content) == str:
            docs.extend(eval(tool_messages[-1].content))
    for doc in docs:
        doc.metadata.pop("abstract")
    
    system_prompt = generate_system_message.format(query=state["query"], documents=docs)

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=" ")])

    return {"messages": [response]}


# In[668]:





# In[28]:


def get_graph():
    graph_builder = StateGraph(TimReviewState)

    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("generate_retrieval_tool_call", generate_retrieval_tool_call)
    graph_builder.add_node("steer_back", steer_back)
    graph_builder.add_node("multiple_retrieve", multiple_retrieve)
    graph_builder.add_node("generate", generate)

    graph_builder.add_edge(START, "query_or_respond")
    graph_builder.add_conditional_edges("query_or_respond", should_research, ["generate_retrieval_tool_call", "steer_back"])
    graph_builder.add_edge("generate_retrieval_tool_call", "multiple_retrieve")
    graph_builder.add_edge("multiple_retrieve", "generate")
    graph_builder.add_edge("steer_back", END)
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()   


# In[29]:


if __name__ == "__main__": 
    graph = get_graph()
    display(Image(graph.get_graph().draw_mermaid_png()))


# In[30]:


if __name__ == "__main__": 
    response = graph.invoke({"query": "what is technology entrepreneurship and business model?"})
    print('\n'+"*"*80+'\n')
    display(Markdown(response['messages'][-1].content))


# In[31]:


#print(response['messages'][-1].content)


# In[14]:





# In[ ]:




