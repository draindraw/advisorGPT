from fastapi import FastAPI
import os
from pydantic import BaseModel
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI()

os.environ['GOOGLE_API_KEY'] = "AIzaSyAbhkn4KQxk8lBNtVEF3sNXV1e47SzO2Ic"

llm = GooglePalm(temperature = 0.3)


class InputData(BaseModel):
    idea: str

@app.post("/predict")
async def generate_text(data: InputData):

    title_template = PromptTemplate(
        input_variables = ['idea'],
        template = "You need to act as an assistant to investors and assess a given startup description, providing its strong and weak point and give a final summary stating if the investor should invest in that startup or not. Here is the startup description : {idea} "
    )

    title_chain = LLMChain(llm = llm, prompt = title_template, verbose = True, output_key = 'output')

    response = title_chain({'idea' : data.idea})

    return response["output"]
