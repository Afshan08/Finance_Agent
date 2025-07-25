from agents import (
    Agent,
    function_tool, InputGuardrail,
    InputGuardrailTripwireTriggered,
    InputGuardrailResult, OutputGuardrail,
    OpenAIChatCompletionsModel,
    OutputGuardrailResult,
    OutputGuardrailTripwireTriggered,
    AsyncOpenAI,
    RunContextWrapper,
    Runner,
    )
import pandas as pd
from agents.run import RunConfig
from openai import OpenAI
from datetime import datetime
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# Gemini API key
gemini_api_key = os.getenv("gemini_api_key")
if not gemini_api_key:
    raise KeyError("KEy not found")

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

@function_tool
def open_metadata(filename):
    """This is a function that opens the metadata file for a dataset"""
    print("opening Meta data: ")
    print(filename)
    try:
        with open(filename, "r") as f:
            data = f.read()
            return data
    except FileNotFoundError:
        print("file not found")
        return "The file was not found"
    except Exception as e:
        print(f"Exception: {e}")
        return f"Ran into this error: {e}"


@function_tool
async def open_file(file_name: str) -> str:
    """ This is a function that give the data of a file by just giving the name of the file"""
    print("opening Meta data")
    try:
        df = pd.read_csv("filename.csv")
        dataset = df[
            [
                "Household expenditures, summary-level categories",
                "UOM",
                "UOM_ID",
                "SCALAR_FACTOR",
                "SCALAR_ID",
                "VECTOR",
                "COORDINATE",
                "VALUE",
                "STATUS",
                "SYMBOL"
            ]
        ]
        data = dataset.head(200)
        return data
    except FileNotFoundError:
        print("file not found")
        return "Sorry the file was not found"
    except Exception as e:
        print(f"pandas exception: {e}")
        return f"Error: {str(e)}"



@function_tool
async def get_time() -> str:
    """This function gives the current time the
    agent can use it to analyze the data according to the market price"""
    current_time = datetime.now().isoformat()
    return current_time


@function_tool
async def write_file(file_name, data):
    """This function allows the agent to write in a file by giving the file any name using
    the arguments and the file name should have a .txt appended so that a txt file is created"""
    print(f"File name:{file_name},\n\n Data:{data}")
    try:
        with open(file_name, "w") as f:
            f.write(data)
            return f"Successfully wrote to {file_name}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"



Finance_Specialist_Agent = Agent(
    name = "finance specialist agent",
    instructions="""You are a finance special agent that haves the data from another agent and you will be looking at the data like morgan housel
    the author of The Physcology of Money would do. You are going to make some predictions for the future and secondly you are going to give tips that are
    actionable to manage their finances more nicely. and after that you will use the function called write file and give file name Finance_Summary with the summary of the
    users finance. After getting the data you have to make sure to right a good report that will include the preidctions for future
     and how to get the finances better and the main reason of finances being good or bad. You have two
    agents to hands off that do have thier speciality in Finances""",
    handoff_description="""This is a finance specialist agent.This is an agent that does some"
                        predictions and give some tips on imporving the finances of the user.""",
    tools=[get_time, write_file],
    model=model
)


Analyzing_Agent = Agent(
  name="analyzer agent",
  instructions="""You are an analyzing agent. You analyze users finances and nothing else. You first analyzie the things that stand out to you than make a
     user behaviour summary with the wrong spending the correct spending the budgeting going wrong reason was it overspending or the
     budget was already low this means increasing income things like that. You have to take care of and you can use as much terms as required according to
     finances. That is your job only and after that handover the task to finance specialist agent""",
  handoff_description="This is the agent that analyzes the data and gives analysis",
  tools=[get_time],
  handoffs=[Finance_Specialist_Agent],
  model=model
)


Orchestrator_Agent = Agent(
    name="orchestrator agent",
    instructions="""You are an agent that first exctracts the data from the file name the client gives you by using a tool that only requires you to know the
    file name to open the filename. for a file containing the meta data you have to use the other
    function called open_metadat and after knowing
     the file you would call the tool and than you will get data in the form of csv of the user and if the user
    does not gives the name of the file before asking for a report of his finances than ask the user to politely give the file name. You have to first handsoff the task
    to Analyzing Agent.""",
    tools=[get_time, open_file, open_metadata],
    handoffs=[Finance_Specialist_Agent, Analyzing_Agent],
    model=model,
)


async def main():
    answer = await Runner.run(Orchestrator_Agent, "Can you write a report on finances from the file name data.csv and the meta data file called metadata.csv")
    print(answer.final_output)

if __name__ == "__main__":
    asyncio.run(main())