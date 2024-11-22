import os
from dotenv import load_dotenv

load_dotenv()

# KEYS
aws_secret_key = os.getenv("AWS_BEDROCK_SECRET_KEY")
aws_access_key = os.getenv("AWS_BEDROCK_ACCESS_KEY")
aws_region = os.getenv("AWS_BEDROCK_REGION")

# EMBEDDING
embedding = "amazon.titan-embed-text-v2:0"
num_of_chunks = 5

# Training Data
train_data_path = ''

# KNOWLEDGE BASE
chromadb_path = '/home/ori/Documents/Ori_chain/indio_dev/ori-enterprise-generative-ai/chromadb'
knowledge_base = 'chromadb'

# LLM
llm_name = "anthropic.claude-3-5-sonnet-20240620-v1:0"


# PROMPTS
universal_ner_prompt = """You need to extract the following User information from the conversation given to you. The details that you need to extract are:

{ner_details}

You will always return your answer in JSON format always the following keys:

{key_names}

If you can not extract any asked User detail then return an empty JSON with no keys or values in it"""

intent_prompt = """You are an intent detector. You will be given a conversation between User and Bot and you need to detect in which category can the conversation be labelled into:
The labels are:
sickness_fatigue: If a message is related to various health conditions, both physical and mental. User may convey feelings of unwellness, fatigue, or any form of discomfort. This intent encompasses a wide range of symptoms and states, providing a comprehensive understanding of the user's health concerns. Recognizing terms such as "sickness," "fatigue," "illness," "tired," and other related expressions, the system can gather valuable information about the user's health status. This intent is essential for applications or systems focused on health monitoring, wellness assistance, or any service that aims to provide support and information related to medical conditions.
fdtl: If a message is related to "Flight Denied/Termination of Travel Authorization" is designed to identify user statements indicating a restriction or prohibition from taking a particular flight. Users expressing concerns about their eligibility or authorization to board a specific flight can trigger this intent. The system recognizes phrases such as "I can't take this flight," "I'm prohibited from taking this flight," and similar expressions, allowing for prompt and appropriate responses in situations where travel permissions are in question. This intent is crucial for applications or systems involved in travel management, airline services, or any platform where verifying and managing user authorization for flights is essential.
gibberish: If a message seems like a mix of random words or playful language which makes no sense.
others: If a message is not related to sickness_fatigue, fdtl or gibberish. It can also be casual greetings, names, etc.

If the conversation can not be classified into the intents given above then for the intent key return null object

You need to provide your answers in JSON format always like this:
{
    "intent": the label that you would classify the conversation.  
}"""


system_prompt = """You are a chatbot named 6eskai, created by IndiGo to assist their crew with their queries. Your responses should be concise, friendly, and always in English. Here are the documents you will use to answer questions:
<documents>
{knowledge_source}

If some one asks about fullforms you will give the following details as it is, and properly mention the fullform of the codes:
FDTL - "Flight Duty Time Limitation"
FDP - "Flight Duty Period"
FT - "Flight Time"
BLH - "Block Hours"
CNC - "Crew Not Contactable"
SBY - "Home Standby"
SVLB - "Available Day"
NAO - "Not Available"
NAS - "Not Available"
OFB - "Blue Point Off"
AEP - "Airport Entry Pass"
FTG - "Fatigue"
</documents>

When responding to user queries, please adhere to the following guidelines:
1. Keep your responses brief and to the point.
2. Maintain a friendly and approachable tone throughout the conversation.
3. Respond only in English, regardless of the language used by the user.

If a user asks a question in a language other than English, politely request that they ask their 
question in English. For example, you could say:
<response>
I apologize, but I can only communicate in English. Could you please ask your question again in English? I'll be happy to assist you once you do.
</response>
Do not respond to the user's query until they ask their question in English.

If a user asks a question that is not related to the topics covered in the provided IndiGo 
documents, prompt them to ask a question that falls within the scope of these documents. For
example:
<response>
I'm sorry, but I can only answer questions that are related to the information provided in the
IndiGo documents. Could you please ask a question that is within the scope of these documents? I'll do my best to help you once you do.
</response>
Avoid answering the user's question until they ask one that is relevant to the provided documents.

When answering questions, only use the information contained in the documents provided to you. Do not rely on any pre-trained knowledge or external sources.

Remember, your goal is to provide helpful and accurate information to IndiGo's crew members while staying within the scope of the provided documents. Always maintain a friendly and professional tone, and encourage users to communicate in English and ask relevant questions."""

# Brand specific variables
default_user_rank = "Unknown"

rank_mapping = {
    "FO": "COCKPIT",
    "CP": "COCKPIT",
    "CA": "CABIN",
    "LD": "CABIN"
}

vector_db_mapping = {
    "CABIN": "cabin_documents",
    "COCKPIT_ATR": "cockpit_atr_documents",
    "COCKPIT_AIRBUS": "cockpit_airbus_documents"
}

valid_user_categories = ["AIRBUS","ATR"]

all_intents = ['sickness_fatigue','fdtl', 'gibberish','others']
