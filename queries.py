import os
import requests
import json

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_NUM_QUERY = 7

def make_embedding(prompt: str) -> list[float]:
    """
    Create the embedding with load balancing endpoint given a prompt.

    Parameters
    prompt (str): prompt from user
    
    Returns
    list[float]: a list represent the embeddings
    """
    headers = {'Content-Type': 'application/json'}
    body = {
        "input": prompt
    }
    res = requests.post(os.getenv("EMBEDDING_ENDPOINT"), headers=headers, json=body)
    return json.loads(res.text)["data"][0]["embedding"]


def make_vectorized_query(prompt: str, k: int) -> VectorizedQuery:
    """
    Create the vectorized query given a prompt and # of neighbor

    Parameters
    prompt (str): prompt from user
    k (int): number of nearest neighbor
    
    Returns
    VectorizedQuery: a vector query => allow hybrid search
    """
    return VectorizedQuery(vector=make_embedding(prompt),
                           k_nearest_neighbors=k,
                           kind="vector",
                           exhaustive=True,
                           fields="Embedding")


def query_ai_search(question: str, num_citations: dict) -> list[dict]:
    """
    Queries the AI Search for citations. "Retrieval" part of RAG pattern.

    Parameters
    question (str): question from user
    num_citations (dict): Number of citations per source type. Should have the format:
                          {'overview': n_citations_overview,
                           'policy'  : n_citations_policy,
                           'saq'     : n_citations_saq}
    
    Returns
    list[dict]: a list of dictionary represent citations
    """
    # Without trimming surrounding double quotes, text search will be performed by exact string rather than intelligent keyword search
        # although semantic search is the same with or without this adjustment
    if question[0] == question[-1] == '"':
        question = question[1:-1]

    citations = []

    # Search overview document
    citations += query_unique_doc_from_index(question=question, 
                                             index_name=os.getenv("SEARCH_INDEX_OVERVIEW"),
                                             num_unique_doc=num_citations['overview'])

    # Search policy documents
    citations += query_unique_doc_from_index(question=question, 
                                             index_name=os.getenv("SEARCH_INDEX_POLICY"),
                                             num_unique_doc=num_citations['policy'])

    # Search previous SAQs
    citations += query_unique_doc_from_index(question=question, 
                                             index_name=os.getenv("SEARCH_INDEX_SAQS"),
                                             num_unique_doc=num_citations['saq'])

    return citations


def query_unique_doc_from_index(question: str, index_name: str, num_unique_doc: int, filter: str | None = None) -> list[dict]:
    """
    Queries the a specific AI Search index for citations

    Parameters
    question (str): question from user
    index_name (str): name of index
    num_unique_doc (int): number of unique doc
    filter (str ! None, default to None): any filter if needed
    
    Returns
    list[dict]: a list of dictionary represent citations
    """
    fields_to_include = json.loads(os.getenv("FIELDS_TO_INCLUDE"))

    citations = []
    unique_files = set()

    if num_unique_doc > 0:
        search_client = SearchClient(
            endpoint = f"https://{os.getenv("SEARCH_SERVICE")}.search.windows.net",
            index_name = index_name,
            credential = AzureKeyCredential(os.getenv("AI_SEARCH_API_KEY")))
    
        for i in range(MAX_NUM_QUERY):
            # we first query the citations (ordered by similarity score)
            citations_iter = search_client.search(question,
                                                  top = num_unique_doc*(i + 1),
                                                  filter = filter,
                                                  vector_queries=[make_vectorized_query(question, num_unique_doc*(i + 1))])

            for c in citations_iter:
                if c["FileName"] not in unique_files:
                    # new unique file => add to citations + unique_files
                    citations.append({key: val for key, val in c.items() if key in fields_to_include})
                    unique_files.add(c["FileName"])
                if len(citations) >= num_unique_doc:
                    break

            if len(citations) >= num_unique_doc:
                break
        
    return citations

def query_llm(question: str, citations: list[dict]) -> str:
    """
    Queries Azure OpenAI for summary of retrieved citations. "Generation" part of RAG pattern.

    Parameters:
    question (str): query provided by user
    citations (list[dict]): list of citation dictionaries retrieved by query_ai_search

    Returns:
    str: message if post response had proper formatting, 
         error string if system unexpected format or POST call failed
    """
    headers = {'Content-Type': 'application/json'}

    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)
    logger.info(f"Current directory: {current_dir}")
    # Construct the full path to the file
    file_path = os.path.join(current_dir, f'{os.getenv("SYSTEM_PROMPT_FNAME")}.txt')
    
    # prepare RAG prompt
    try:
        with open(file_path, 'r') as f:
            system_prompt = f.read()
    except FileNotFoundError as e:
        print("File not found. Please check the file path:", e)
    except IOError as e:
        print("An error occurred while reading the file:", e)
    source_text = ""

    for i, output in enumerate(citations):
        source_text += f"Document {i+1}: {output["Content"]}\n"

    # TEST CODE
    # Initialize source_text
    # source_text = "OVERVIEW DOCUMENT/S:\n"

    # Iterate over the citations and append to source_text
    # for i, output in enumerate(citations):
    #     if i in range(0,num_citations['overview']):
    #         source_text += f"Document: {output['Content']}\n"
    #     elif i in range(num_citations['overview'], num_citations['overview']+num_citations['policy']):
    #         if i == range(num_citations['overview'], num_citations['overview']+num_citations['policy'])[0]:
    #             source_text += f"\n-----\nPOLICY DOCUMENT/S:\n"
    #         source_text += f"Document: {output['Content']}\n"
    #     elif i in range(num_citations['overview']+num_citations['policy'], num_citations['overview']+num_citations['policy']+num_citations['saq']):
    #         if i ==range(num_citations['overview']+num_citations['policy'], num_citations['overview']+num_citations['policy']+num_citations['saq'])[0]:
    #             source_text += "\n-----\nRETURNED SURVEY QUESTIONNAIRES:\n"
    #         source_text += f"Document: {output['Content']}\n"

    body = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt + source_text
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "temperature": float(os.getenv("LLM_TEMPERATURE"))
        # 'top_p': float(os.getenv("LLM_TOP_P")),
    }

    # call LLM
    try:
        response = requests.post(os.getenv("LLM_ENDPOINT"), headers=headers, json=body)
        response.raise_for_status() # for some reason necessary to raise HTTPError
        response_dict = json.loads(response.text)
        
        # gets answer from reponse object, with some error handling
        if ('choices' in response_dict and 
            isinstance(response_dict['choices'], list) and
            'message' in response_dict['choices'][0] and
            'content' in response_dict['choices'][0]['message']):
            answer = response_dict['choices'][0]['message']['content']
            return answer
        else:
            return "The response dictionary was in an unexpected format, please try again."
    except requests.exceptions.HTTPError as e:
        response_dict = json.loads(response.text)
        if response_dict.get("error", {}).get("code", "") == "context_length_exceeded":
            return f"Sorry, the question and documents exceed the maximum allowed length. Please shorten the question or request fewer documents."
        else:
            return f"Error: status code {response.status_code}; {e}"