from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from prompt_generator import create_prompt_template
from dotenv import load_dotenv
import os

load_dotenv()

def setup_llm(model_name="gemini-2.5-flash", temperature=0.7):
    """Google Gemini LLM ko setup karta hai"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found!\n"
            "Please set it in .env file:\n"
            "GOOGLE_API_KEY=your_api_key_here\n\n"
            "Get your API key: https://aistudio.google.com/apikey"
        )
    
    google_api_key = google_api_key.strip().strip('"')
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=temperature,
        max_output_tokens=2048,
        convert_system_message_to_human=True
    )
    
    return llm

def create_summarization_chain(llm=None):
    """Prompt template aur LLM ke saath chain banata hai"""
    if llm is None:
        llm = setup_llm()
    
    template = create_prompt_template()
    output_parser = StrOutputParser()
    chain = template | llm | output_parser
    
    return chain

def summarize_paper(paper_title, style="Technical but accessible", 
                   length="Medium (500-800 words)", llm=None):
    """Research paper ko summarize karta hai using Gemini"""
    if llm is None:
        llm = setup_llm()
    
    template = create_prompt_template()
    formatted_prompt = template.format(
        paper_input=paper_title,
        style_input=style,
        length_input=length
    )
    
    response = llm.invoke(formatted_prompt)
    
    if isinstance(response, str):
        return response
    elif hasattr(response, 'content'):
        return response.content
    else:
        return str(response)