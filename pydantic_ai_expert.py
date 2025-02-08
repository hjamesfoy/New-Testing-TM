from __future__ import annotations
import os
import asyncio
from dataclasses import dataclass
from typing import List
from datetime import datetime
import uuid

from dotenv import load_dotenv
import logfire

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import create_client, Client

# Optionally load local environment variables when running locally.
load_dotenv()

# Configure logfire (if you have the token, it will send logs)
logfire.configure(send_to_logfire='if-token-present')

# Retrieve configuration from environment variables (Render will set these)
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the model and API clients
model = OpenAIModel(llm)
supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Define the dependencies available to the agent
@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

# Define your agent's system prompt
system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

# Create the agent instance with 2 retries.
pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

# Utility function to get an embedding from OpenAI.
async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return a zero vector on error

# RAG Tool: Retrieve relevant documentation chunks.
@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    try:
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        if not result.data:
            return "No relevant documentation found."
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

# RAG Tool: List all available documentation pages.
@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        if not result.data:
            return []
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

# RAG Tool: Get full content of a documentation page.
@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        if not result.data:
            return f"No content found for URL: {url}"
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]
        for chunk in result.data:
            formatted_content.append(chunk['content'])
        return "\n\n".join(formatted_content)
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI()

# The original simple endpoint (if you wish to keep it)
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.get("/")
async def root():
    return {"message": "Agentic RAG API is up and running."}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    deps = PydanticAIDeps(
        supabase=supabase_client,
        openai_client=openai_client
    )
    ctx = RunContext(deps=deps)
    try:
        response_text = await pydantic_ai_expert.run(request.query, ctx)
        return ChatResponse(answer=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# New Endpoint for /v1/chat/completions
# ---------------------------

# Define models that mimic the OpenAI Chat Completions API format.
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-mini"
    messages: List[ChatMessage]

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions_endpoint(request: ChatCompletionRequest):
    # Extract the last user message from the provided messages.
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found.")
    query = user_messages[-1].content

    # Build the dependencies context and run the agent.
    deps = PydanticAIDeps(
        supabase=supabase_client,
        openai_client=openai_client
    )
    ctx = RunContext(deps=deps)
    try:
        response_text = await pydantic_ai_expert.run(query, ctx)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Prepare the response in the Chat Completions format.
    assistant_message = ChatMessage(role="assistant", content=response_text)
    now = int(datetime.utcnow().timestamp())
    # (For this example, token usage is mocked.)
    usage = ChatUsage(
        prompt_tokens=0,
        completion_tokens=len(response_text.split()),
        total_tokens=len(response_text.split())
    )
    response = ChatCompletionResponse(
        id=str(uuid.uuid4()),
        created=now,
        model=request.model,
        choices=[
            ChatChoice(
                index=0,
                message=assistant_message,
                finish_reason="stop"
            )
        ],
        usage=usage
    )
    return response

# Only used when running locally. Render will use the command specified in its settings.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("pydantic_ai_expert:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
