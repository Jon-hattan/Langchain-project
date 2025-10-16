import os
from dotenv import load_dotenv

load_dotenv() #take environment variables: API key
model_name = os.getenv("MODEL_NAME")
llm_api_key = os.getenv("GROQ_API_KEY")

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
user_prompt = HumanMessagePromptTemplate.from_template("What is the name of the capital of {country}?")
system_prompt = SystemMessagePromptTemplate.from_template("You are a geography expert.")

print(user_prompt.format(country="Japan"))

first_prompt = ChatPromptTemplate.from_messages([
    system_prompt, user_prompt
])


#initiate model
from langchain_groq import ChatGroq
chat = ChatGroq(
    model_name=model_name,         # e.g., "llama-3.1-8b-instant" or "llama-3.3-70b-versatile"
    temperature=0.8,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

#pipe operator
chain_1 = (
    {
            "country": lambda x:x["country"]
    }
    | first_prompt
    | chat
    | {"article_title": lambda x: x.content}
)

res = chain_1.invoke({
    "country":"Japan"
    })
print(res["article_title"])

