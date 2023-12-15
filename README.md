
# RAG Structured Output Generation ChatBot
<div  align="center">
<img  height="82"  width="82" src="https://github.com/Ribin-Baby/RAG-json-responderV1/blob/main/images/json_icon.svg">
</div>

##  🌟.❓ Problem Statement:

1.  **Large Language Models (LLMs):** Utilize quantized versions of LLMs like llama2, open-llama, or falcon that can run on Google Colab.

2.  **Fine-tuning/Prompt Engineering:**

	-  **Weather Service:**

		- Identify location from user queries (e.g., "weather in Chennai").

		- Respond with *"Service: Weather, Location: Chennai"*.

		- If location is missing, prompt user and respond with *"Location: Chennai"*.

		- Support stateful interaction with multiple query-response exchanges.

	-  **News Service:**

		- Extract location from queries about news (e.g., "news for India").

		- Respond with *"Location: India"* or *"Location: USA"* depending on the location found.

		- If location is missing, prompt user and respond with *"Service: News, Location: India."*

3.  **Generalization:** Design the system to work with any service and identify the relevant service and location from user prompts.


**Specifically:**


- Develop a generic response format **"Service: <Service>, Location: <Location>"**.

- Identify weather-related prompts (e.g., "weather in Bangalore", "climate in Bangalore") and respond with *"Service: Weather, Location: Bangalore"*.

- Identify news-related prompts (e.g., "news for India") and respond with *"Service: News, Location: India"*.

- Allow for handling of missing location information with prompts.

  
**Desired Outcome:**


- A system that can dynamically identify services and locations from user prompts and provide appropriate responses.

- The system should be generalizable to work with any service and location.

- The system should be lightweight and run efficiently on Google Colab.

  
 
## 🌟. **🔍** Model Used:

<div  align="center">
<img  src="https://github.com/imoneoi/openchat/raw/master/assets/logo_new.png"  style="width: 60%">
</div>

  
* We are using GPTQ quantized version of [ openchat_3.5](https://huggingface.co/TheBloke/openchat_3.5-GPTQ) model, by "[TheBloke](https://huggingface.co/TheBloke)".

	* GPTQ is a post-training quantization (PTQ) method for 4-bit quantization.

* The original [openchat_3.5](https://huggingface.co/openchat/openchat_3.5) model itself is a fine-tuned [Mistral](https://mistral.ai/) Model.

	*  **🔥 The first 7B model Achieves Comparable Results with ChatGPT (March)! 🔥**

	*  **🤖 Open-source model on MT-bench scoring 7.81, outperforming 70B models 🤖**

* The original *openchat_3.5* model can runs on consumer GPU with 24GB RAM, but the quantized version consumes ~**6GB VRAM GPU** only.

* This is a `4 bit` quantized, `7 Billion Parameter` model, with a `sequence length of 4096`.

  
  
### 🏷️ <a id="benchmarks"></a> Model Benchmarks:

| Model              | # Params | Average  | MT-Bench     | AGIEval  | BBH MC   | TruthfulQA    | MMLU         | HumanEval       | BBH CoT     | GSM8K        |
|--------------------|----------|----------|--------------|----------|----------|---------------|--------------|-----------------|-------------|--------------|
| OpenChat-3.5       | **7B**   | **61.6** | 7.81         | **47.4** | **47.6** | **59.1**      | 64.3         | **55.5**        | 63.5        | **77.3**     |
| ChatGPT (March)*   | ?        | 61.5     | **7.94**     | 47.1     | **47.6** | 57.7          | **67.3**     | 48.1            | **70.1**    | 74.9         |
| Mistral            | 7B       | -        | 6.84         | 38.0     | 39.0     | -             | 60.1         | 30.5            | -           | 52.2         |
| Open-source SOTA** | 13B-70B  | 61.4     | 7.71         | 41.7     | 49.7     | 62.3          | 63.7         | 73.2            | 41.4        | 82.3         |
|                    |          |          | WizardLM 70B | Orca 13B | Orca 13B | Platypus2 70B | WizardLM 70B | WizardCoder 34B | Flan-T5 11B | MetaMath 70B |


<!-- requirements start -->

## 🌟. **💻** Requirements

1.  **System Requirements**

```python

~ 6GB of system RAM

~ 6GB of GPU VRAM
```

> [!IMPORTANT]
> 📌 A GPU of 6GB VRAM is madatory for running the inference. Will works fine on google colab with T4 GPU enabled.

 <div  align="center">
<img  src="https://github.com/Ribin-Baby/RAG-json-responderV1/blob/main/images/system_usage.png"  style="width: 60%">
<br>
<figcaption><i>Fig.1 - Resource usage in google colab notebook environment with T4 GPU</i></figcaption>
</div>

2.  **Software Requirements**

	*  **python-version**: `3.10`

	*  **CUDA-version**: `11.8`

⬇️ **python packages**

```python

pip install accelerate==0.25.0

pip install auto-gptq --extra-index-url "https://huggingface.github.io/autogptq-index/whl/cu118/"

pip install bitsandbytes==0.41.3.post2

pip install einops==0.7.0

pip install langchain==0.0.349

pip install optimum==1.15.0

pip install tiktoken==0.5.2

pip install torch @ "https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=a81b554184492005543ddc32e96469f9369d778dedd195d73bda9bed407d6589"
pip install torchaudio @ "https://download.pytorch.org/whl/cu118/torchaudio-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=cdfd0a129406155eee595f408cafbb92589652da4090d1d2040f5453d4cae71f"
pip install torchvision @ "https://download.pytorch.org/whl/cu118/torchvision-0.16.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=033712f65d45afe806676c4129dfe601ad1321d9e092df62b15847c02d4061dc"

pip install transformers==4.35.2
```

<!-- requirements end -->

  <!-- details start -->

## 🌟. **📖** `Explained`:
  1. 🛠️ **Tools used:**
  <p  align="center">
<img  height="62"  width="62"  src="https://skillicons.dev/icons?i=python"/>
<img  height="62"  width="62"  src="https://skillicons.dev/icons?i=pytorch"/>
<img  height="62"  width="62"  src="https://huggingface.co/front/assets/huggingface_logo.svg"/>
<img  height="62"  width="62"  src="https://integrations.langchain.com/favicon.ico"/>
<img  height="62"  width="62"  src="https://cdn.discordapp.com/icons/1130134702557249637/72dcf8a26e2efb0e2a3679ea70c430bd.webp"/>
</p>

- *`transformers`*: "for importing and using LLM model from 🤗Huggingface."
- *`auto-gptq`*: "for working with quantized models."
- *`langchain`*: 🦜🔗 "for  advanced  prompting."

2.  **💬** **Conversation-Flow Chart:**
 <div  align="center">
<a href="https://github.com/Ribin-Baby/RAG-json-responderV1/blob/main/images/RAG_jsonout.drawio.png">
<img  src="https://github.com/Ribin-Baby/RAG-json-responderV1/blob/main/images/RAG_jsonout.svg"  style="width: 90%"></a>
<br>
<figcaption><i>Fig.2 - conversation flowchart of BOT interaction</i></figcaption>
</div>
	
- This section describes the process of handling user input and generating structured output.

	**I] User Input and Prompt:**

	-   When the bot receives user input, it is combined with a specific prompt.

	-   This prompt instructs the LLM to generate a structured output.
    - For example if the user input is : `USER: what is the weather in Chennai?`
	
	- **📚** **Prompt template: OpenChat**
		```
		GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:
		```

  - The `{prompt}` along with the user input will be :
	```bash
	GPT4 Correct User: As an Named Entity Recognition and Intent Classification Expert, your task is to analyze questions like the following '###user_input': 
	###user_input: what is the weather in Chennai? 
	You are required to perform two main tasks based on the '###user_input': 
	#Task 1: Classify the '###user_input' into one of the predefined intents ['Weather', 'News']. IMPORTANT: If no clear match to given intents is found, categorize the intent as "Other". 
	#Task 2: Extract any geographical or location-related entities present in the '###user_input'. IMPORTANT: If no specific location is mentioned, label the location as "Other". 
	###INSTRUCTIONS: Do NOT add any clarifying information. Output MUST follow the schema below. The output should be formatted as a JSON instance that conforms to the JSON schema below. 
	As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]} 
	the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted. 
	Here is the output schema: ''' {"properties": {"service_intent": {"title": "Service Intent", "description": "This field stores the intent classified from the user input.", "type": "string"}, "location": {"title": "Location", "description": "This field stores the location value extracted from the user input.", "type": "string"}}, "required": ["service_intent", "location"]} '''
	<|end_of_turn|>GPT4 Correct Assistant:
	```
	- This prompt is generated by combining 🦜🔗langchain along with some custom instructions at the first place.
	- We can dynamically pass the required services here it is `['Weather', 'News']`.

	 
	**II] LLM Output Generation:** **🤖**
	-   The combined user input and prompt are passed to the LLM for processing.
	-   The LLM attempts to generate structured output, aiming for JSON format.

	**III]  Output Validation:**  **👀**
	-	**checking for structured output format:**
		- If the output is not in JSON format an iterative process begins.
		- The user input is passed again to the LLM with the same prompt.
		-  This loop continues for a pre-defined number of iterations (N).
		- This iterative process will help to filter out some rare glitch in LLM output.
	- **checking for missing key values:**
		- Following the generation of well-structured JSON data by the LLM, an additional validation step is performed. This step focuses on ensuring that the essential keys, "service" and "location," contain valid non-null values.
		-  **Service Key:** The value of the "service" key is checked for null or emptiness.
		-  **Location Key:** Similarly, the value of the "location" key is checked for null or emptiness.

	- **🧑‍⚕️** **Follow-up Questioning:** 

		If either key (service or location) lacks a valid value:
		- **⚙️** **Service**
		    -   The bot initiates a follow-up questioning flow specifically designed to elicit the missing service information from the user.
		    -   This may involve asking the user directly what service they are seeking information about.
    
		- **📍** **Location**
    
		    -   A similar follow-up questioning flow is initiated if the "location" key lacks a valid value.
		    -   The bot prompts the user for the desired location information.
    

	- **🚦** **Resuming the Process:**

		- Once the user provides the missing information, the original process resumes.
		-  The combined user input (including service and location information) is again passed to the LLM with the specific prompt for structured output generation.
		-  The validation and follow-up questioning steps repeat as needed until all essential key-value pairs are obtained and a valid structured output is generated.

	- 🔥**This iterative approach guarantees that the bot generate a complete and accurate JSON response all the time, even if the user initially forgets to provide all necessary information.**
	- 🔥We also passing previous one chat history along with the user input in-order to direct the LLM to generate better quality results. 


3. ⚡**User interactions:** ⚡
	
	-  Direct query:-
	<div  align="center">
	<img  src="https://github.com/Ribin-Baby/RAG-json-responderV1/blob/main/images/direct_chat1.png"  style="width: 90%">
	<br>
	<img  src="https://github.com/Ribin-Baby/RAG-json-responderV1/blob/main/images/direct_chat2.png"  style="width: 90%">
	<br>
	<figcaption><i>Fig.3 - user directly asking the complete query in the first try itself</i></figcaption>
	</div>
---	
- conversation with follow-up questioning  when `service` information is missing:-
	<div  align="center">
	<img  src="https://github.com/Ribin-Baby/RAG-json-responderV1/blob/main/images/chat_with_service_followup.png"  style="width: 90%">
	<br>
	<figcaption><i>Fig.4 - user gives query without specifying the service they need</i></figcaption>
	</div>
---
-  conversation with follow-up questioning  when `location` information is missing:-
	<div  align="center">
	<img  src="https://github.com/Ribin-Baby/RAG-json-responderV1/blob/main/images/chat_with_loc_followup.png"  style="width: 90%">
	<br>
	<figcaption><i>Fig.5 - user gives query without specifying the location  details</i></figcaption>
	</div>
---
- conversation with follow-up questioning  when both `location` & `service` informations are missing:-
	<div  align="center">
	<img  src="https://github.com/Ribin-Baby/RAG-json-responderV1/blob/main/images/usual_chat.png"  style="width: 90%">
	<br>
	<figcaption><i>Fig.6 - user starts the conversation with a greeting only</i></figcaption>
	</div>
---
<!-- details  end-->

## 🌟. **🚀** Hands oN:
### <u>CLI</u>
- clone repo

```bash

git  clone  https://github.com/Ribin-Baby/RAG-json-responderV1.git

```

- change directory

```bash

cd  ./RAG-json-responderV1

```

- install requirements

```bash

pip  install  -r  requirements.txt

```

- 💥 Run

> [!NOTE]

> Need GPU with 6GB VRAM and cuda 11.8 installed. Better run on colab with T4 GPU.

```python

python bot.py --s '["News", "Weather"]'

```

- not only *["News", "Weather"]* as services we can pass any services we need dynamically to *["Game", "Law"]* or *["Economics", "Law", "Weather"]* or any.

### <u>UI</u>
- open the `bot_notebook.ipynb` file in google colab environment and change the runtime to `GPU`. And run cell-by-cell.
- It may be required to restart the colab after installing the packages. To do that run this cell below.
	> import  IPython
	> IPython.Application.instance().kernel.do_shutdown(True)
 - upon running the last cell, you will get an interactive UI to chat with the BoT.

### 👍 The END