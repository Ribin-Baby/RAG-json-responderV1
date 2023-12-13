# IMPORTING Libraries
import argparse
from typing import Any, List, Mapping, Optional
from pydantic import BaseModel, Field

from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langchain.output_parsers import PydanticOutputParser

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch
import json
import os
import warnings
# Ignore warnings related to cuDNN, cuFFT, and cuBLAS registration
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow.compiler.xla.stream_executor.cuda")


# class inference for langchain
class OpenchatLLM(LLM):
    """
    loading Quantized openchat3.5-GPTQ model,
    into langchain.
    model_name="TheBloke/openchat_3.5-GPTQ",
    hf_url="https://huggingface.co/TheBloke/openchat_3.5-GPTQ"
    """
    device:str = "cpu"
    chatmodel:Any = None
    chatokenizer:Any = None

    # All the optional arguments
    top_p:          Optional[float] = 0.97
    top_k:          Optional[int]   = 50
    max_tokens:     Optional[int]   = 512
    temp:           Optional[float] = 0.55
    repeat_penalty: Optional[float] = 1.2

    def __init__(self, model, tokenizer, device,  **kwargs):
        super(OpenchatLLM, self).__init__()
        self.device = device
        self.chatmodel = model
        self.chatokenizer = tokenizer

    @property
    def _llm_type(self) -> str:
        return "TheBloke/openchat_3.5-GPTQ"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        params = {
            **self._get_model_default_parameters,
            **kwargs
        }
        encoded_inputs = self.chatokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.chatmodel.generate(input_ids=encoded_inputs['input_ids'],
                        attention_mask=encoded_inputs['attention_mask'], do_sample=True, **params)
        # output_without_input = outputs[:, len(encoded_inputs['input_ids'][0]):]
        response = self.chatokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # response = self.response_loop(encoded_inputs, params)
        return response

    @property
    def _get_model_default_parameters(self):
        return {
            "max_new_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temp,
            "repetition_penalty": self.repeat_penalty,
            "eos_token_id":self.chatokenizer.convert_tokens_to_ids("<|end_of_turn|>"),
            "pad_token_id":self.chatokenizer.convert_tokens_to_ids("<|pad_0|>"),
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_new_tokens": self.max_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temp,
            "repetition_penalty": self.repeat_penalty
        }


class ServiceData(BaseModel):
    service_intent: str = Field(description="This field stores the intent classified from the user input.")
    location: str = Field(description="This field stores the location value extracted from the user input.")    


class Dialogflow:
  def __init__(self, llm, prompt_template, parser, n, services:list):
    self.n = n # max retry
    self.llm = llm
    self.prompt_template = prompt_template
    self.parser = parser
    self.services = services
    self.user_input = ""
    self.llm_output = ""
    self.History = {"chat1":None, "chat2":None}

  def taking_input(self):
    self.user_input = input("USER: ")
    return True

  def print_output(self, text):
    print("BOT:",text)

  def inject_input(self, user_input):
    if self.llm_output != "":
        if self.llm_output.service_intent != "Other":
          return f'{user_input} and Service: {self.llm_output.service_intent} is already given.'
        elif self.llm_output.location != "Other":
          return f'{user_input} and Location: {self.llm_output.location} is already given.'
    return user_input

  def llm_response(self, prompt, n=0):
    ## retrying for getting structured answer
    if n==self.n: # n=> no: retrys if output is not structured as expected
      print("LLL response generation failed.")
      return None, None
    try:
      output = self.llm(prompt)
      parsed_output = self.parser.parse(output.split("GPT4 Correct Assistant:")[-1])
      s, loc = parsed_output.service_intent, parsed_output.location
      return parsed_output, parsed_output.json()
    except Exception as e:
      n += 1
      return self.llm_response(prompt, n=n)

  def chat(self, lock=False):
      if self.taking_input():
        # if user give an input
        if self.user_input in ["Exit", "Bye", "exit", "bye"]:
          # Exit conditions
          self.print_output("Shutting Down")
          return "Shutting Down"
        if lock:
          self.user_input = self.inject_input(self.user_input)

        # generating output using LLM LOOP
        ## maintain a chat history for better conversation flow
        self.History['chat1'] = self.History['chat2']
        input_prompt = self.prompt_template.format(user_input=self.user_input, intents=self.services)
        self.History['chat2'] =  input_prompt
        if self.History['chat1'] is not None:
            input_prompt = "<|end_of_turn|>".join([self.History['chat1'], input_prompt])
        ## LLM response generation
        processed_output, raw_output = self.llm_response(input_prompt)
        self.History['chat2'] += raw_output
        service = processed_output.service_intent
        loc = processed_output.location
        self.llm_output = processed_output

        # checking for missing keys
        if service == "Other" and loc == "Other":
          midman = f"To continue, you need to pick a service from {self.services}, also need to mention a location."
        elif service == "Other" and loc != "Other":
          # Service available
          midman = f"can you please specify choose a Service from {self.services}, for the given Location: `{loc}`?"
        elif service != "Other" and loc == "Other":
          # Location available
          midman = f"what Location you are looking for the given Service: `{service}`?"
        else:
          # both are available
          midman = ""
          self.print_output(raw_output)
          return processed_output

        return self.followup(midman)
      else:
        # if user dont give an input repeat the process till get one
        return self.chat()

  def followup(self, midman_query):
      # if the user dont provide any of the required keys, do a followup questioning conversation
      self.print_output(midman_query)
      processed_output = self.chat(lock=True)
      return processed_output
  

# SETTING GLOBAL VARIABLES
os.environ["SAFETENSORS_FAST_GPU"] = "1"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
device = "cuda" if torch.cuda.is_available() else "cpu"

services = ["News", "Weather"]
model_name = "TheBloke/openchat_3.5-GPTQ"

if __name__=="__main__":
    # START
    print("[STARTing CHATbot]")
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', dest='services', type=json.loads, help='desired services as list', required=False)
    args = parser.parse_args()
    services = args.services
    print("SERVICES used:", services)
    # To use a different branch, change revision
    # For example: revision="gptq-4bit-32g-actorder_True"
    gptq_config = GPTQConfig(bits=4, damp_percent=0.01, use_cuda_fp16=True, desc_act=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="auto",
                                                use_safetensors=True,
                                                trust_remote_code=False,
                                                revision="main",
                                                quantization_config = gptq_config
                                                )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # PROMPTING
    ## output parser
    ### prompt for getting structured data (JSON) from LLM
    template_string = """GPT4 Correct User: As an Named Entity Recognition and Intent Classification Expert, your task is to analyze questions like the following `###user_input`:

    ###user_input: {user_input}

    You are required to perform two main tasks based on the `###user_input`:

    #Task 1: Classify the `###user_input` into one of the predefined intents {intents}. IMPORTANT: If no clear match to given intents is found, categorize the intent as "Other".

    #Task 2: Extract any geographical or location-related entities present in the `###user_input`. IMPORTANT: If no specific location is mentioned, label the location as "Other".

    ###INSTRUCTIONS: Do NOT add any clarifying information. Output MUST follow the schema below. {format_instructions}<|end_of_turn|>GPT4 Correct Assistant:"""

    output_parser = PydanticOutputParser(pydantic_object=ServiceData)
    main_prompt = PromptTemplate(
        template=template_string,
        input_variables=["user_input", "intents"],
        partial_variables={"format_instructions": output_parser
    .get_format_instructions()})

    # creating object of LLM inference class
    llm = OpenchatLLM(model=model, tokenizer=tokenizer, device=device)
    # start the conversation flow
    chatbot = Dialogflow(llm=llm, prompt_template=main_prompt, parser=output_parser, n=10, services=services)
    chat_history = chatbot.chat()
    print("PROCESSED OUTPUT:", chat_history.json())