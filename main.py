from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, Runner, RunContextWrapper
from agents.run import RunConfig
from openai import OpenAI
from datetime import datetime
import asyncio
import os

from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.getenv("gemini_api_key")

if not gemini_api_key:
    raise KeyError("Gemini not found")


# Reference: https://ai.google.dev/gemini-api/docs/openai
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

async def fib_n(n):
    if n <=1 :
        return n
    else:
        await asyncio.sleep(0.2)  # Simulate some delay
        print(f"Calculating fib({n})")
        return asyncio.run(fib_n(n-1)) + asyncio.run(fib_n(n-2))

@function_tool
async def fib(n):
    answer = asyncio.run(fib_n(n))
    return str(answer)

agent = Agent(
    name="my agent",
    tools=[fib],
    instructions="Call the function to know the fibonnaci sequence",
    model=model
    
)

async def main():
    answer = await Runner.run(agent, "tell me the fibonnaci sequence of 10")
    print(answer.final_output)

asyncio.run(main())