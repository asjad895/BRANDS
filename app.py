from typing import Dict
import traceback
import art

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import BackgroundTasks,Depends
from orichain.llm import LLM
import parameters
import utils

data_trainer = utils.Train_data(
    chromadb_path = parameters.chromadb_path,
    train_data_path = parameters.train_data_path,
    vector_db_mapping = parameters.vector_db_mapping
)

llm = LLM(
    model_name = parameters.llm_name,
    aws_access_key = parameters.aws_access_key,
    aws_secret_key = parameters.aws_secret_key,
    aws_region = parameters.aws_region,
)

app = FastAPI(redoc_url=None, docs_url=None)

@app.post("/generative_response")
async def generate(request: Request) -> Response:
    """This endpoint is responsible for generating a response based on the
    user's input message, optional previous conversation history, and any additional metadata.

    Arg:
    - user_message `(string, required)`: The user's input message.
    - metadata `(dict, optional)`: Additional metadata related to the request.
        - session_id `(string, optional)`: A unique identifier for the current session.
        - stream `(boolean, optional)`: Indicates whether the response should be streamed or not.
        - Any additional metadata specific to a brand or use case can be added as key-value pairs within this object.
    - prev_pairs `(array of dict, optional)`: An array representing the previous conversation history,
    where each object has the following structure:
        - role `(string, required)`: Either "user" or "assistant".
        - content `(string, required)`: The content of the message.
        - If prev_pairs is not provided or null, it will be treated as an empty conversation history.
        - The last object in the array should have a “role” set to "assistant".
    - prev_chunks `(array of dict, optional)`: An array containing the top retrieved chunk(s) from the last user query.
        - If prev_chunks is not provided or null, it will be treated as an empty array.

    Return:
    - response `(string)`: The generated response based on the user's input and optional previous conversation history.
    - message `(string)`: User text received in request data.
    - matched_sentence `(array of string)` list of matched sentences used by AI to answer the user’s query.
    - metadata `(object, optional)`: Any additional metadata related to the generated response. Keys might differ for different brands.
        - session_id  `(string)`: session ID in string.
        - user_lang `(string)`: user message’s language in string.
        - user_message_vector `(array of float)`: vector/list of float numbers

    - Status Codes:
        - `200` OK: The request was successful.
        - `4XX` Bad Request: The request body is invalid or missing required parameters.
        - `5XX` Internal Server Error: An unexpected error occurred on the server."""

    try:
        # Fetching data from the request recevied
        request_json = await request.json()

        # Fetching valid keys
        user_message = request_json.get("user_message")
        metadata = request_json.get("metadata")
        prev_pairs = request_json.get("prev_pairs")
        prev_chunks = request_json.get("prev_chunks")

        # Validating data
        validation_check = await utils.generative_request_validation(
            user_message=user_message,
            metadata=metadata,
            prev_pairs=prev_pairs,
            prev_chunks=prev_chunks,
        )
        if validation_check:
            return JSONResponse(validation_check)
        print("User Request : ",request_json)
        chunks_metadata = {
            'pdf_name':metadata.get('pdf_name',[]),
            'pdf_section':metadata.get('pdf-section',[]),
            'pdf_page':metadata.get('pdf_page',[])
        }

        system_prompt,chunks_metadata = await data_trainer.retrieve_chunks(
            user_message = user_message,
            prev_relevant_chunk = prev_chunks,
            rank = metadata.get('rank',None),
            category = metadata.get('category',None),
            chunks_metadata = chunks_metadata
        )

        # Streaming
        if metadata.get("stream"):
            return StreamingResponse(
                llm.stream(
                    request = request,
                    user_message = user_message,
                    matched_sentence = chunks_metadata['matched_sentence'],
                    system_prompt = system_prompt,
                    chat_hist = prev_pairs,
                    extra_metadata = chunks_metadata
                ),
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
                media_type="text/event-stream",
            )
        # Non streaming
        else:
            llm_response = await llm(
                request = request,
                user_message = user_message,
                matched_sentence = chunks_metadata['matched_sentence'],
                system_prompt = system_prompt,
                chat_hist = prev_pairs,
                extra_metadata = chunks_metadata
            )

            return JSONResponse(llm_response)

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
        return JSONResponse({"error": 500, "reason": str(e)})


@app.post("/results")
async def results(request: Request) -> Response:
    """
    This endpoint is responsible for processing user and bot message and detecting:
        - `intent`
        - `entities`
            - brand specific entities (`brand_entity`)
            - common entities like name, age etc. (`universal_ner`)
        - `language_detection`

    Args:
        request (`Request`): The incoming request object containing JSON data.

    The JSON payload should include:
        - user_message `(string, required)`: The user's input message.
        - bot_message `(string, required)`: The bot's response message.
        - intent `(object, optional)`: To check if intent needs to be detected based on the user's message.
        - brand_entity `(object, optional)`: Entity information specific to the brand.
        - universal_ner `(object, optional)`: Universal Named Entity Recognition results.
        - language_detection `(object, optional)`: To detect language used by user.

    Returns:
        Response: A JSON response containing the processed results or validation errors.

        The response structure includes:
        - Processed results based on the input data.
        - In case of an error, an object with 'error' and 'reason' keys.

    Status Codes:
        - 200 OK: The request was successful and results were generated.
        - 4XX Bad Request: The request body is invalid or missing required parameters.
        - 5XX Internal Server Error: An unexpected error occurred on the server.

    Raises:
        Exception: Any unexpected errors during processing are caught and logged.

    Note:
        - The function performs validation on the input data using utils.results_request_validation().
        - If validation fails, it returns the validation error response.
        - The main processing is done asynchronously using `utils.results_processsing()`.
        - In case of exceptions, it logs the error details and returns a 5XX error response.
    """
    try:
        # Fetching data from the request recevied
        request_json = await request.json()

        # Fetching valid keys
        user_message = request_json.get("user_message")
        bot_message = request_json.get("bot_message")
        intent = request_json.get("intent")
        brand_entity = request_json.get("brand_entity")
        universal_ner = request_json.get("universal_ner")
        language_detection = request_json.get("language_detection")

        # Validating data
        validation_check = await utils.results_request_validation(
            user_message=user_message,
            bot_message=bot_message,
            intent=intent,
            brand_entity=brand_entity,
            universal_ner=universal_ner,
        )
        if validation_check:
            return JSONResponse(validation_check)
        
        print("User request ",request_json)
        # Async processing all required details
        results = await utils.results_processsing(
            user_message = user_message,
            bot_message = bot_message,
            intent = intent,
            brand_entity = brand_entity,
            universal_ner = universal_ner,
            language_detection = language_detection,
            request = request,
            llm = llm,
        )

        return JSONResponse(results)

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
        return JSONResponse({"error": 500, "reason": str(e)})


print(art.text2art("Server has started!", font="small"))

async def train_data(request : Request,background_task : BackgroundTasks) -> JSONResponse:
    """_summary_

    Args:
        request (Request): _description_
        background_task (BackgroundTasks): _description_

    Returns:
        JSONResponse: _description_
    """
    # 
    request_data = await request.json()
    # data_trainer.train(train = True)
    background_task.add_task(data_trainer.train(train = True))
    return JSONResponse({"status":"Training has been started ,in a while it will complete."})
