# RAG Structured Output Generation ChatBot
<p align="center">
<img height="62" width="62" src="https://skillicons.dev/icons?i=python"/>
 <img height="62" width="62" src="https://skillicons.dev/icons?i=pytorch"/>
<img height="62" width="62" src="https://huggingface.co/front/assets/huggingface_logo.svg"/>
<img height="62" width="62" src="https://integrations.langchain.com/favicon.ico"/>
<img height="62" width="62" src="https://cdn.discordapp.com/icons/1130134702557249637/72dcf8a26e2efb0e2a3679ea70c430bd.webp"/>
</p>

## ❓ Problem Statement:

1. **Large Language Models (LLMs):** Utilize quantized versions of LLMs like llama2, open-llama, or falcon that can run on Google Colab.
2. **Fine-tuning/Prompt Engineering:**
    - **Weather Service:**
        - Identify location from user queries (e.g., "weather in Chennai").
        - Respond with *"Service: Weather, Location: Chennai"*.
        - If location is missing, prompt user and respond with *"Location: Chennai"*.
        - Support stateful interaction with multiple query-response exchanges.
    - **News Service:**
        - Extract location from queries about news (e.g., "news for India").
        - Respond with *"Location: India"* or *"Location: USA"* depending on the location found.
        - If location is missing, prompt user and respond with *"Service: News, Location: India."*
3. **Generalization:** Design the system to work with any service and identify the relevant service and location from user prompts.

**Specifically:**

- Develop a generic response format **"Service: <Service>, Location: <Location>"**.
- Identify weather-related prompts (e.g., "weather in Bangalore", "climate in Bangalore") and respond with *"Service: Weather, Location: Bangalore"*.
- Identify news-related prompts (e.g., "news for India") and respond with *"Service: News, Location: India"*.
- Allow for handling of missing location information with prompts.

**Desired Outcome:**

- A system that can dynamically identify services and locations from user prompts and provide appropriate responses.
- The system should be generalizable to work with any service and location.
- The system should be lightweight and run efficiently on Google Colab.


## **🔍**  Model Used:
<div align="center">
  <img src="https://github.com/imoneoi/openchat/raw/master/assets/logo_new.png" style="width: 65%">
</div>

* We are using GPTQ quantized version of [  openchat_3.5](https://huggingface.co/TheBloke/openchat_3.5-GPTQ) model, by "[TheBloke](https://huggingface.co/TheBloke)".
	* GPTQ is a post-training quantization (PTQ) method for 4-bit quantization.
* The original [openchat_3.5](https://huggingface.co/openchat/openchat_3.5) model itself is a fine-tuned [Mistral](https://mistral.ai/) Model.
	* **🔥 The first 7B model Achieves Comparable Results with ChatGPT (March)! 🔥**
	* **🤖 Open-source model on MT-bench scoring 7.81, outperforming 70B models 🤖**
	
* The original *openchat_3.5* model can runs on consumer GPU with 24GB RAM, but the quantized version consumes ~**6GB VRAM GPU** only.

* This is a `4 bit` quantized, `7 Billion Parameter` model, with a `sequence length of 4096`.


### <a id="benchmarks"></a> Model Benchmarks

| Model              | # Params | Average  | MT-Bench     | AGIEval  | BBH MC   | TruthfulQA    | MMLU         | HumanEval       | BBH CoT     | GSM8K        |
|--------------------|----------|----------|--------------|----------|----------|---------------|--------------|-----------------|-------------|--------------|
| **OpenChat-3.5**       | **7B**   | **61.6** | 7.81         | **47.4** | **47.6** | **59.1**      | 64.3         | **55.5**        | 63.5        | **77.3**     |
| ChatGPT (March)*   | ?        | 61.5     | **7.94**     | 47.1     | **47.6** | 57.7          | **67.3**     | 48.1            | **70.1**    | 74.9         |
| Mistral            | 7B       | -        | 6.84         | 38.0     | 39.0     | -             | 60.1         | 30.5            | -           | 52.2         |
| Open-source SOTA** | 13B-70B  | 61.4     | 7.71         | 41.7     | 49.7     | 62.3          | 63.7         | 73.2            | 41.4        | 82.3         |
|                    |          |          | WizardLM 70B | Orca 13B | Orca 13B | Platypus2 70B | WizardLM 70B | WizardCoder 34B | Flan-T5 11B | MetaMath 70B |

<!-- requirements start -->
###  **💻** Requirements
1. **System Requirements**
```python
~ 6GB of system RAM
~ 6GB of GPU VRAM
#NOTE: GPU is madatory for running the inference
```
2. **Software Requirements**
	* **python-version**:  `3.10`
	* **CUDA-version**:  `11.8`
	
**python packages**
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


<!-- prompt-template start -->
### Prompt template: OpenChat
```
GPT4 User: {prompt}<|end_of_turn|>GPT4 Assistant:

```
<!-- prompt-template end -->
