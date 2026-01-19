import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    print("Error: API Token nahi mila! .env file check karein.")
else:
    repo_id = "Qwen/Qwen2.5-7B-Instruct" 
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.7,
        huggingfacehub_api_token=hf_token
    )

    model = ChatHuggingFace(llm=llm)

    print(f"Chatbot started with {repo_id}! Type 'exit' to quit.")
    
    chat_history = [ 
            SystemMessage(content="You are a helpful assistant.")
        ]
    
    while True:
        user_input = input("You: ")
        chat_history.append(HumanMessage(content=user_input))
        if user_input.lower() in ["exit", "quit"]:
            break
        
        try:
            response = model.invoke(chat_history)
            chat_history.append(AIMessage(content=response.content))
            print(f"Bot: {response.content}")
        except Exception as e:
            print(f"Error: {e}")
            
print(chat_history)