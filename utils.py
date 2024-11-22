from typing import Dict, List, Union, Tuple, Any, Optional
from collections import OrderedDict
import traceback
import json
from orichain import validate_gen_response_request, validate_result_request
from orichain.lang_detect import LanguageDetection
from fastapi import Request
import asyncio
import parameters
import chromadb
from orichain.embeddings import EmbeddingModels
from orichain.knowledge_base import KnowledgeBase
import pandas as pd
import os
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import time

embedding_model = EmbeddingModels(
    model_name = parameters.embedding,
    aws_secret_key = parameters.aws_secret_key,
    aws_access_key = parameters.aws_access_key,
    aws_region = parameters.aws_region
)

# Embedding function for chromadb
class MyEmbeddingFunction(chromadb.EmbeddingFunction):

        """
        This is a custom function that generates embeddings for text data using the given model.
        """
        def __call__(self, Docs: chromadb.Documents) -> chromadb.Embeddings:

            """
            This function generates embeddings for a list of text documents using the given model.
            Args:
                Docs (chromadb.Documents): A list of text documents.
            Returns:
                chromadb.Embeddings: A list of embeddings (numerical representations) for the input text documents.
            """
            embeddings = [embedding_model(text=chunk) for chunk in Docs]
            return embeddings


lang_detect = LanguageDetection()


async def generative_request_validation(
    user_message,
    metadata,
    prev_pairs,
    prev_chunks,
) -> Union[Dict, None]:
    validation_check = await validate_gen_response_request(
        user_message=user_message,
        metadata=metadata,
        prev_pairs=prev_pairs,
        prev_chunks=prev_chunks,
    )

    # Add more validation steps if needed...

    return validation_check


async def results_request_validation(
    user_message,
    bot_message,
    intent,
    brand_entity,
    universal_ner,
) -> Union[Dict, None]:
    validation_check = await validate_result_request(
        user_message=user_message,
        bot_message=bot_message,
        intent=intent,
        brand_entity=brand_entity,
        universal_ner=universal_ner,
    )

    # Add more validation steps if needed...

    return validation_check


async def generative_prompt(retrived_chunks: List[str]) -> str:
    if len(retrived_chunks)>0:
        knowledge_source = "\n\n".join(retrived_chunks)
    else:
        knowledge_source = "NO DATA"

    system_prompt = parameters.system_prompt.format(knowledge_source=knowledge_source)

    return system_prompt


async def results_processsing(
    user_message: str,
    bot_message: str,
    request: Request,
    llm: Any,
    intent: Optional[Dict] = {},
    brand_entity: Optional[Dict] = {},
    universal_ner: Optional[Dict] = {},
    language_detection: Optional[Dict] = {},
) -> Dict:
    try:
        tasks = OrderedDict()
        whole_chat = f"The converstation between User and Bot is given below:\nBot: {bot_message}\nUser: {user_message}"
        custom_whole_chat = user_message

        if intent.get("trigger"):
            tasks["intent"] = llm(
                request = request,
                user_message = custom_whole_chat,
                system_prompt = parameters.intent_prompt,
                sampling_paras = {"temperature": 0.0, "max_tokens": 100},
                do_json = True,
            )

        # Change it according to the brand usecase
        if brand_entity.get("trigger"):
            brand_ner_details = "\n".join(brand_entity.get("list_of_entities"))
            key_names = "\n".join(brand_entity.get("list_of_entities"))

            tasks["brand_entity"] = llm(
                request=request,
                user_message=whole_chat,
                system_prompt=parameters.brand_entity_prompt.format(
                    brand_ner_details=brand_ner_details, key_names=key_names
                ),
                # sampling_paras={"temperature": 0.0, "max_tokens": 4096},
                do_json=True,
            )

        if universal_ner.get("trigger"):
            ner_details = ""

            modifications_of_entities = universal_ner.get("modifications_of_entities")

            if not modifications_of_entities:
                modifications_of_entities = [None] * len(
                    universal_ner.get("list_of_entities")
                )

            for index, tuple_details in enumerate(
                zip(universal_ner.get("list_of_entities"), modifications_of_entities)
            ):
                ner_details += f"\n{tuple_details[0]}"
                if tuple_details[1]:
                    ner_details += f": {tuple_details[1]}"

            key_names = "\n".join(universal_ner.get("list_of_entities"))

            tasks["universal_ner"] = llm(
                request=request,
                user_message=whole_chat,
                system_prompt=parameters.universal_ner_prompt.format(
                    ner_details=ner_details.strip(), key_names=key_names
                ),
                # sampling_paras={"temperature": 0.0, "max_tokens": 4096},
                do_json=True,
            )

        if language_detection.get("trigger"):
            tasks["language_detection"] = lang_detect(user_message=user_message)

        if len(tasks) == 0:
            return {}

        results = await asyncio.gather(*tasks.values())

        # Create a dictionary to map task names to their results
        result_map = dict(zip(tasks.keys(), results))

        if result_map.get("intent") and "error" not in result_map.get("intent"):
            intent_result = json.loads(result_map.get("intent").get("response")).get(
                "intent", None
            )
            # validating intent
            if intent_result:
                if intent_result.strip().lower() not in parameters.all_intents:
                    intent_result = None


            result_map["intent"] = (
                {"name": intent_result} if intent_result else {"name": None}
            )

        if result_map.get("brand_entity") and "error" not in result_map.get(
            "brand_entity"
        ):
            result_map["brand_entity"] = json.loads(
                result_map.get("brand_entity").get("response")
            )

        if result_map.get("universal_ner") and "error" not in result_map.get(
            "universal_ner"
        ):
            result_map["universal_ner"] = json.loads(
                result_map.get("universal_ner").get("response")
            )

        for key, value in result_map.items():
            if not value:
                result_map[key] = None

        return result_map
    except Exception as e:
        exception_type = type(e).__name__
        exception_message = str(e)
        exception_traceback = traceback.extract_tb(e.__traceback__)
        line_number = exception_traceback[-1].lineno

        print(f"Exception Type: {exception_type}")
        print(f"Exception Message: {exception_message}")
        print(f"Line Number: {line_number}")
        print("Full Traceback:")
        print("".join(traceback.format_tb(e.__traceback__)))
        return {"error": 500, "reason": str(e)}
    

async def get_knowledge_base(collection_name : str) ->object:
    """_summary_

    Args:
        collection_name (str): Chromadb Collection name.

    Returns:
        object: _description_
    """
    knowledge_base_manager = KnowledgeBase(
        type = parameters.knowledge_base,
        collection_name = collection_name,
        path = parameters.chromadb_path,
        embedding_function = MyEmbeddingFunction()
    )
    return knowledge_base_manager


class Train_data (object):
    """
    Manages all the fuction related to retrival and vector database
    """
    class MyEmbeddingFunction(chromadb.EmbeddingFunction):
        """
        This is a custom function that generates embeddings for text data using the given model.
        """
        def __call__(self, Docs: chromadb.Documents) -> chromadb.Embeddings:
            """
            This function generates embeddings for a list of text documents using the given model.
            Args:
                Docs (chromadb.Documents): A list of text documents.
            Returns:
                chromadb.Embeddings: A list of embeddings (numerical representations) for the input text documents.
            """
            embeddings = [embedding_model(text=chunk) for chunk in Docs]
            return embeddings
    
    def __init__(
            self, 
            chromadb_path:str = parameters.chromadb_path, 
            train_data_path:str = parameters.train_data_path,
            vector_db_mapping:Dict = parameters.vector_db_mapping
        ) -> None:
        """ 
        >>> `train_data_path` should be in ``.xlsx`` extension. 
        """
        self.client = chromadb.PersistentClient(path=chromadb_path)
        self.knowledge_base_menu = vector_db_mapping
        self.training_data_path = train_data_path
        
    def train(self, train:bool = False) -> None:
        if train or not self.client.list_collections():
            # Need to allow the client to do a reset
            self.client.get_settings().allow_reset=True
            # This can not be revered
            self.client.reset()
            print("All the collections has been removed")
            # Create a temp dir for storing chromadb then delete after pushing to aws.
            temp_dir = '../Temp_chromadb'
            os.makedirs(temp_dir,exist_ok = True)
            print(f"Temperary Dir created at : {os.curdir}+/+{temp_dir}")
            excel_data = pd.read_excel(self.excel_sheet_path, sheet_name=None)
            # Get the sheet names
            sheet_names = list(excel_data.keys())
            for collection_name in (list(set(sheet_names) & set(list(self.knowledge_base_menu.values())))):
                print(f"Starting training for {collection_name}")
                collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.MyEmbeddingFunction(),
                    metadata={"hnsw:space": "cosine"})
                collection.add(
                    documents=excel_data[collection_name]['Details'].to_list(),
                    metadatas=excel_data[collection_name].to_dict(orient='records'),
                    ids=excel_data[collection_name].index.astype(str).to_list()
                )
            print("Data has been loaded succesfully")
        else:
            print("Data has been loaded succesfully")
            
    async def retrieve_chunks(
            self, 
            user_message : str, 
            prev_relevant_chunk : List[str], 
            rank: str,
            category : str,
            chunks_metadata : Dict
            )->Tuple[str,Dict]:
        """Retrive chunks from chromadb dynamically based on user_identity_tag mapped with specific keys.

        Args:
            user_message (str): user query
            prev_relevant_chunk (List[str]): previous retrived chunks
            user_identity_tag (str): a key sent by backend

        Returns:
            Tuple[str,Dict]: _description_
        """
        try: 
            user_rank = rank
            user_category = category
            if user_rank not in parameters.rank_mapping or user_rank is parameters.default_user_rank:
                return {"error":"invalid user rank"}
            elif user_category and user_category not in parameters.valid_user_categories:
                return {"error":"invalid user category"}
        
            # KNOWLEDGE BASE
            user_identity_tag = None
        
            if user_rank in parameters.rank_mapping:
                user_rank = parameters.rank_mapping.get(user_rank)
                if user_category:
                    user_identity_tag = f"{user_rank}_{user_category}"
                    print(user_identity_tag)
                else:
                    user_identity_tag = user_rank
            else:
                user_identity_tag = parameters.default_user_rank
            # Default to "cabin_documents" collection
            collection_name = self.knowledge_base_menu.get(user_identity_tag, "cabin_documents")
                    
            knowledge_base_manager = await get_knowledge_base(collection_name = collection_name) 
            query_embedding = await embedding_model(user_message = user_message)
            query_results = await knowledge_base_manager(
                user_message_vector = query_embedding,
                num_of_chunks=parameters.num_of_chunks
            )
            # print(query_results['metadatas'])
            if prev_relevant_chunk:
            # Adding corresponding data from both
                transformed_data = {
                    "details": query_results['documents'][0] + prev_relevant_chunk,
                    "metadata":{
                        "pdf_name": [str(item["PDF"]) for item in query_results['metadatas'][0]]+ chunks_metadata["pdf_name"],
                        "pdf_section": [str(item["Section"]) for item in query_results['metadatas'][0]]+ chunks_metadata["pdf_section"],
                        "pdf_page": [str(item["Page No."]) for item in query_results['metadatas'][0]]+ chunks_metadata["pdf_page"]
                    }
                }
                
            else:
                
                transformed_data = {
                    "details": query_results['documents'][0],
                    "metadata":{
                        "pdf_name": [str(item["PDF"]) for item in query_results['metadatas'][0]],
                        "pdf_section": [str(item["Section"]) for item in query_results['metadatas'][0]],
                        "pdf_page": [str(item["Page No."]) for item in query_results['metadatas'][0]]
                        }
                }
            matched_chunks = transformed_data['details']
            transformed_data['metadata']['matched_sentence'] = matched_chunks
            transformed_data["metadata"]['user_message_vector'] = query_embedding
            chunks_metadata = transformed_data['metadata']
            # Final prompt
            system_prompt= await generative_prompt(
            retrived_chunks = matched_chunks
            )
            return system_prompt,chunks_metadata
        
        except Exception as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            exception_traceback = traceback.extract_tb(e.__traceback__)
            line_number = exception_traceback[-1].lineno

            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Line Number: {line_number}")
            print("Full Traceback:")
            print("".join(traceback.format_tb(e.__traceback__)))
            return {"error": 500, "reason": str(e)}
        


def upload_folder_to_s3(s3_bucket, input_dir, s3_path):
    """
    Uploads all files from a local directory to an S3 bucket with a progress bar and elapsed time.

    Args:
        s3_bucket (boto3.s3.Bucket): The S3 bucket object where files will be uploaded.
        input_dir (str): The local directory containing files to upload.
        s3_path (str): The target S3 path (directory) where files will be uploaded.

    Raises:
        Exception: Raises an exception if the upload fails.
    """
    print("Uploading results to S3 initiated...")
    print(f"Local Source: {input_dir}")
    print(f"Destination S3 Path: {s3_path}")

    # Collect all files to upload
    files_to_upload = []
    for path, _, files in os.walk(input_dir):
        for file in files:
            local_file_path = os.path.join(path, file)
            relative_path = os.path.relpath(path, input_dir)
            s3_file_path = os.path.join(s3_path, relative_path, file).replace("\\", "/")
            files_to_upload.append((local_file_path, s3_file_path))
    
    total_files = len(files_to_upload)
    print(f"Total files to upload: {total_files}")

    start_time = time.time()

    try:
        for local_file_path, s3_file_path in tqdm(files_to_upload, desc="Uploading", unit="file"):
            # Upload file to S3
            s3_bucket.upload_file(local_file_path, s3_file_path)
        
        elapsed_time = time.time() - start_time
        print(f"\nUpload complete. Time elapsed: {elapsed_time:.2f} seconds.")

    except Exception as e:
        print(" ...Failed to upload! Quitting upload process.")
        print(e)
        raise e


s3_resource = boto3.resource('s3')
s3_bucket = s3_resource.Bucket("oriserve-dev-nlp")
s3_folder = "indigo/vector_database/chromadb"
local_dir = "/home/ubuntu/projects/chromadb"
# upload_folder_to_s3(s3bucket,local_dir ,s3_folder)

def download_s3_folder(bucket_name: str, s3_folder: str, local_dir: str = None) -> None:
    """
    Download contents of an S3 folder to a local directory.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_folder (str): Path of the folder in S3 to download.
        local_dir (str, optional): Local directory to save files. If None, uses current directory.

    Raises:
        FileNotFoundError: If the specified S3 folder does not exist.
        ClientError: If there's an issue connecting to S3 or downloading files.

    Returns:
        None
    """
    try:
        s3_resource = boto3.resource("s3")
        bucket = s3_resource.Bucket(bucket_name)

        print(f"Downloading files from S3 bucket '{bucket_name}', folder '{s3_folder}'")

        objects = list(bucket.objects.filter(Prefix=s3_folder))
        
        if not objects:
            raise FileNotFoundError(f"Folder '{s3_folder}' does not exist in bucket '{bucket_name}'")

        for obj in objects:
            # Skip if the object is a folder (ends with '/')
            if obj.key.endswith('/'):
                continue

            # Determine the local file path
            if local_dir:
                target = os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
            else:
                target = obj.key

            # Create local directory if it doesn't exist
            os.makedirs(os.path.dirname(target), exist_ok=True)

            # Download the file
            print(f"Downloading: {obj.key} to {target}")
            bucket.download_file(obj.key, target)

        print("Download completed successfully.")

    except ClientError as e:
        print(f"Error connecting to S3 or downloading files: {e}")
        raise e        
# download_s3_folder(bucket_name = 'oriserve-dev-nlp',s3_folder = s3_folder,local_dir = local_dir)