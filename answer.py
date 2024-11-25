from flask import current_app as app
from queries import query_ai_search, query_llm
import pandas as pd
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import time # For timing the processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

file_path = 'sample_sheet.xlsx'

def answer(file_path):
    start_time = time.time()
    logger.info("Reading file...")
    # Read the uploaded file into bytes
    df = pd.read_excel(file_path).iloc[:28]
    logger.info(f"File read succesfully: {file_path}")
    
    try:
        # Process the "question" column row by row
        for index, row in df.iterrows():
            logger.info(f"Answering question at row {index+1}")
            question = row['Question']
            logger.info(f"Question: {question}")
        
            num_citations = {
                    'overview': 1,
                    'policy'  : 3,
                    'saq'     : 3
                }

            # Query the AI Search service for similar documents
            citations = query_ai_search(question, num_citations)

            # Invoke the LLM model to answer the question
            llm_response = query_llm(question, citations)

            # Insert the LLM response into the "LLM Answer" column
            df.at[index, 'POC LLM Answer'] = llm_response

            formatted_citations = ''
            for item in citations:
                formatted_citations += f"{item['FileLinkingUrl']}\n"
            df.at[index, 'POC Citations'] = formatted_citations
            logger.info(f"Done answering question at row {index+1}")
        
    except Exception as e:
        logger.error(f"Error in processing file: {e}")  # Debug print
        
    
    logger.info(f"Done answering all questions")
    # Convert the dataframe to a downloadable file
    df.to_excel('result_sheet_revised_prompt_test2.xlsx', index=False)

    # Check if the file processing was successful
    if df['POC LLM Answer'].notnull().all():
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        elapsed_time_formatted = time.strftime("%M:%S", time.gmtime(elapsed_time))
        logger.info(f"Elapsed time: {elapsed_time_formatted} to process {len(df)} questions.")
        return 'File processed successfully.'
    else:
        return 'Error in processing file. Please try again.'

if __name__ == '__main__':
    answer(file_path)