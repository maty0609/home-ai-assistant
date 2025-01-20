import os
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI



os.environ["AZURE_OPENAI_API_KEY"] = ""

llm = AzureChatOpenAI(
    azure_endpoint="",
    azure_deployment="gpt-4o",
    openai_api_version="2024-08-01-preview",
)

# Define a short prompt
prompt = "Translate 'Hello, how are you?' to French."

# Create a message with the prompt
message = HumanMessage(content=prompt)

# Invoke the model with the message
response = llm.invoke([message])

# Print the response
print(response.content)
