from typing import Dict, List, Union, Tuple, Optional,AsyncGenerator
from collections import OrderedDict
import traceback
import json
from fastapi.requests import Request
from orichain import validate_gen_response_request, validate_result_request
from orichain.lang_detect import LanguageDetection 
from fastapi import Request
import asyncio
import parameters
import re
from orichain.llm import LLM
from  orichain.embeddings import EmbeddingModels
from orichain.knowledge_base import KnowledgeBase


knowledge_base_manager = KnowledgeBase(
    type = parameters.knowledge_base,
    collection_name = parameters.collection_name,
    path = parameters.chromadb_path

)

embedding_model = EmbeddingModels(
    model_name=parameters.embedding,
    model_download_path=parameters.embedding_model_path,
)
llm = LLM(
    model_name=parameters.llm_name,
    api_key=parameters.openai_key,
)
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

    # Custom validation which is not in library, as of this version when developed.
    

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


async def generative_prompt(all_chunks : List[str]) -> str:
    """This function will return final system_prompt for bot,which can be directly used without any modification.

    Args:
        all_chunks (List[str]): List of chunks retrieved from vector database

    Returns:
        str: Final System Bot Prompt
    """
    if len(all_chunks)>0:
        knowledge_source = '' 
        for i,doc in enumerate(all_chunks):
            knowledge_source += f"source {i}: {doc} \n\n"
    else:
        knowledge_source = "NO DATA"

    system_prompt = parameters.system_prompt.format(knowledge_source = knowledge_source)
    return system_prompt


async def results_processsing(
    user_message: str,
    bot_message: str,
    request: Request,
    intent: Optional[Dict] = {},
    brand_entity: Optional[Dict] = {},
    universal_ner: Optional[Dict] = {},
    language_detection: Optional[Dict] = {},
) -> Dict:
    try:
        tasks = OrderedDict()
        whole_chat = f"Focus on Bot question for extracting entity. You should Always extract entity when bot question is related to asking user information like email,phone_number,name etc irrespective of valid or invalid. The converstation between User and Bot is given below:\nBot_Question: {bot_message}\nUser_Response: {user_message}"
        # As we are not taking context of chat history , we will only take current user query.
        custom_whole_chat_intent = user_message

        if intent.get("trigger"):
            tasks["intent"] = llm(
                request=request,
                user_message = custom_whole_chat_intent,
                system_prompt=parameters.intent_prompt,
                # sampling_paras={"temperature": 0.0, "max_tokens": 4096},
                do_json=True,
            )

        # Change it according to the brand usecase
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
            
            system_prompt=parameters.universal_ner_prompt.format(
            ner_details=ner_details.strip(), key_names=key_names)
            tasks["universal_ner"] = llm(
                request=request,
                user_message=whole_chat,
                system_prompt = system_prompt,
                # sampling_paras={"temperature": 0.0, "max_tokens": 4096},
                do_json=True,
            )

        if language_detection.get("trigger"):
            tasks["language_detection"] = LanguageDetection()(user_message = user_message)

        if len(tasks) == 0:
            return {}

        results = await asyncio.gather(*tasks.values())

        # Create a dictionary to map task names to their results
        result_map = dict(zip(tasks.keys(), results))

        if result_map.get("intent"):
            intent_result = json.loads(result_map.get("intent").get("response")).get(
                "intent", None
            )
            # List of all intents
            all_intents = parameters.all_intents
            result_map["intent"] = {"name":intent_result} if intent_result in all_intents else {"name":None}

        if brand_entity.get("trigger"):
            result_map["brand_entity"] = None

        if result_map.get("universal_ner"):
            result_map["universal_ner"] = json.loads(
                result_map.get("universal_ner").get("response")
            )
            print(result_map['universal_ner'])
            for key ,value in result_map['universal_ner'].items():
                value = [value] if value!='' else None
                result_map['universal_ner'][key] = value
            if not result_map['universal_ner']:
                for entity in universal_ner.get("list_of_entities"):
                    result_map['universal_ner'][entity] = None

        if result_map.get('language_detection'):
            # result_map['language_detection']['user_lang'].lower()
            # Making it fixed to english because bot will be only in english to avoid any wrong language prediction.
            result_map['language_detection']['user_lang'] = 'en'

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
    
#----------------------------------------------------------------------------------
## Custom Brand related code ##
async def get_expanded_query(user_message : str,chat_history : List[Dict],request : Dict)->str:
    """Asynch function for getting expanded query. 

    Args:
        llm (`object`): orichain llm object
        chat_history (`List[Dict]`): chat history
        request_body (`Dict`): user_request Body

    Returns:
        >>> str: Expanded query in the format of ``Ques : expanded_query``
    """
    try:
        prompt=parameters.expand_query_prompt
        response_text= await llm(
            request=request,
            user_message = user_message,
            system_prompt = prompt,
            chat_hist = chat_history,
            sampling_paras=parameters.sampling_paras,
            do_json = True
            )
        # print(response_text)
        # string response ,need to convert in dict as our prompt is to return json.
        response = response_text['response']
        response = json.loads(response)
        # print(type(response),response)
        res = response.get('rephrased_query')
        if res:
            if isinstance(res,list):
                # If not empty,takes 1st one .
                if len(res)>0:
                    user_mes_expanded = f"Ques : {res[0]}"
                else:
                    user_mes_expanded = f"Ques : {user_message}"
            # If String
            else:
                user_mes_expanded = f"Ques : {res}"
        # If res is None
        else:
            user_mes_expanded = f"Ques : {user_message}"
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
        # When try failed .
        user_mes_expanded = ""
    print(" Query Expanded : \n",user_mes_expanded)
    return user_mes_expanded



async def retrieval_with_query_expansion(user_message : str, 
                                        prev_conversation : List[Dict] ,
                                        request: Request, 
                                        top1_chunk :Optional[str]=[''],
                                        metadata : Dict ={}
                                        ) -> Tuple[List,str,str,str]:
    """This is a asynch parallel execution of retrieval with original query and rephrased query.
    then select only top 5 chunks from both.first priority will to the rephrased query then original query.

    Args:
        llm (object): _description_
        prev_conversation (List[Dict]): _description_
        request_body (Dict): _description_
        top1_chunk (str): _description_
        embedding_model (object): _description_
        knowledge_base_manager (object): _description_

    Returns:
        Tuple[List,str,str,str]: List[str] chunks ,chunks_types(metadata from chromadb) , rephrased_query and final system prompt generated by ```generative_prompt```.
    """
    # st_qe = time.time()
    if not top1_chunk:
        top1_chunk = []
    expanded_query = await get_expanded_query(
        user_message = user_message,
        chat_history = prev_conversation, 
        request = request
        )
    # print(f"Time in query Expansion : {time.time()-st_qe}")

    if len(expanded_query) > 0:
        expanded_query_embedding = await embedding_model(expanded_query)
        retrieval_task1 = asyncio.create_task(knowledge_base_manager(user_message_vector=expanded_query_embedding,num_of_chunks=parameters.num_of_chunks))
    else:
        retrieval_task1 = None

    user_message_embedding = await embedding_model(user_message)
    retrieval_task2 = asyncio.create_task(knowledge_base_manager(user_message_vector=user_message_embedding,num_of_chunks=parameters.num_of_chunks))

    # Gather the results of both retrieval tasks
    retrieval_results = await asyncio.gather(
        retrieval_task1,
        retrieval_task2
    )

    retrieved_chunks1 = []
    chunks_types1 = []
    if retrieval_task1:
        # Result of expanded query
        semantic_search_result1 = retrieval_results[0]
        # print(semantic_search_result1)
        if 'error' not in semantic_search_result1:
            chunks_types1 = [meta['query_type'] for meta in semantic_search_result1['metadatas'][0]]
            # print(chunks_retrieved_categories)
            retrieved_chunks1=semantic_search_result1['documents'][0] + top1_chunk
    # Result of original query
    semantic_search_result2 = retrieval_results[1]
    if 'error' not in semantic_search_result2:
        chunks_types2 = [meta['query_type'] for meta in semantic_search_result2['metadatas'][0]]
        retrieved_chunks2=semantic_search_result2['documents'][0]
    else:
        retrieved_chunks2 = []
        chunks_types2 = []

    # Using set for getting unique chunks from both retrieval
    all_chunks = retrieved_chunks1.copy()
    for i, j in enumerate(retrieved_chunks2):
        if j not in all_chunks:
            all_chunks.append(j)
            chunks_types1.append(chunks_types2[i])
    
    if len(all_chunks) > 5:
        all_chunks = all_chunks[:5]
        chunks_types1 = chunks_types1[:5]
    
    # Prompt finalizer- final step
    system_prompt= await generative_prompt(
        all_chunks = all_chunks
    )
    # Adding Follow up Question
    if metadata.get('followup_question'):
        follow_up_prompt = parameters.follow_up_prompt.format(follow_up_question = metadata.get('followup_question'))
        system_prompt+=follow_up_prompt
    # print(system_prompt)
    return all_chunks,chunks_types1,expanded_query,system_prompt


class GenAIResponse:
    """A class for generating AI responses based on a given client.
    """
    async def get_streaming(
            self, 
            chunks_types : List[str],
            request ,
            matched_sentence :List[str],
            system_prompt : str
            ) -> AsyncGenerator:
        """
        Gets the streaming response for a given query.

        Args:
        session_id (str): The session identifier.
        user_message_vector (List[float]): The user message vector.
        user_query (str): The user's query.
        chat_history (List[Dict]): History of previous chat messages.
        documents (List[str]): List of documents related to the query.

        Returns:
        AsyncGenerator: An async generator yielding the SSE response and event type.
        """
        request_json = await request.json()

        response = llm.stream(
            request=request,
            user_message = request_json['user_message'],
            matched_sentence = matched_sentence,
            system_prompt = system_prompt,
            chat_hist = request_json['prev_pairs'],
            sampling_paras=parameters.sampling_paras,
            do_json = True,
            do_sse = False
            )
        answer_end_tokens=[".\",\n"," \"","fallback",".\",\n"," \"",'"?\",\n"','"\",\n"'," "]
        unwanted_tokens = ["!\",\n","?\",\n"]
        fallback_start = False
        no_of_json_parse_tokens=5
        count=0
        async for part in response:
            content = part
            print("Token  :\n",content)
            # Count for controlling initial unwanted token to be stream.
            count+=1
            if content and content == 'fallback' or content == '."':
                fallback_start =True
                print("Stream Stopped here.",content)
            elif count>no_of_json_parse_tokens and content not in answer_end_tokens and fallback_start==False and content not in unwanted_tokens:
                sse_chunk = await self.format_sse(data = content,event='text')
                yield sse_chunk

            else:
                if isinstance(content,dict):
                    body = content
                    payload = {}
                    if body:
                        response = body['response']
                        json_response = json.loads(response)
                        answer=json_response.get('answer')
                        fallback=json_response.get('fallback',"false")
                        if not isinstance(fallback ,bool):
                            if fallback.strip().lower() in ["True","true"]:
                                fallback=True
                            else:
                                fallback=False
                    else:
                        answer = ''
                        fallback = False
                    # payload 
                    payload['response'] = answer
                    payload['message'] = request_json['user_message']
                    # response metadata
                    response_metadata = {}
                    if request_json.get('session_id','') :
                        response_metadata['session_id'] = request_json.get('session_id','')
                    response_metadata['fallback']=fallback
                    response_metadata['query_type'] = chunks_types
                    response_metadata['matched_sentence'] = matched_sentence
                    payload['metadata'] = response_metadata
                    body_response = await self.format_sse(data=payload,event='body')
                    yield body_response


    async def without_streaming(
            self, 
            chunks_types : List[str],
            request :Request,
            matched_sentence :List[str],
            system_prompt : str) -> str:
        """
        Generates response without streaming.

        Args:
            user_query (str): The user's query.
            chat_history (List[Dict]): History of previous chat messages.
            documents (List[str]): List of documents related to the query.

        Returns:
            str: The generated response.
        """
        request_json = await request.json()
        # Global payload to update iteratively
        payload = {}
        try:
            ## get response
            response_text= await llm(
                request=request,
                user_message = request_json['user_message'],
                matched_sentence = matched_sentence,
                system_prompt = system_prompt,
                chat_hist = request_json['prev_pairs'],
                sampling_paras=parameters.sampling_paras,
                do_json = True
                )
            json_response = response_text['response']
            print("Response : \n",json_response)
            json_response = json.loads(json_response)
            answer=json_response.get('answer')
            fallback=json_response.get('fallback',"false")
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

        if not isinstance(fallback ,bool):
            if fallback.strip().lower() in ['True','true']:
                fallback=True
            else:
                fallback=False
        #payload
        request_json = await request.json()
        payload['response'] = answer
        payload['message'] = request_json['user_message']
        response_metadata = {}
        if request_json.get('session_id','') :
            response_metadata['session_id'] = request_json.get('session_id','')
        response_metadata['fallback']=fallback
        response_metadata['query_type'] = chunks_types
        response_metadata['matched_sentence'] = matched_sentence
        payload['metadata'] = response_metadata
        return payload

    async def format_sse(self,data:str, event :str ) -> str:
        """
    Formats the data into a Server-Sent Events (SSE) format.

    Args:
        data (str): The data to be formatted.
        event (str, required): The event type, `event` or `body`.

    Returns:
        str: The formatted SSE data.
        """
        msg = f'data: {json.dumps(data)}\n\n'
        if event is not None:
            msg = f'event: {event}\n{msg}'
        return msg

