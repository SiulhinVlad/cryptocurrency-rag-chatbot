# Conversational Cryptocurrency Chatbot

An educational cryptocurrency chatbot exploring conversational
retrieval-augmented generation (RAG) with LangChain and GPT-4,
alongside a separate Llama 2 fine-tuning approach.

Developed as a Bachelor Semester Project 3 at the University of Luxembourg
between September 2023 and January 2024.

**[Read the full project report](BSP_S3.pdf)**

## Overview

The project explores how natural language processing and large language
models can support an educational chatbot answering introductory
questions about cryptocurrency.

Two independent implementations were developed:

- **Conversational RAG:** retrieves relevant information from a custom
  dataset and provides it to GPT-4 as context for answering questions.
- **Fine-tuned Llama 2:** adapts Llama 2 7B to cryptocurrency question
  answering using parameter-efficient fine-tuning.

My work included preparing and publishing the datasets, implementing
the conversational RAG pipeline, fine-tuning Llama 2, publishing the
resulting model on Hugging Face and developing a notebook interface
for model interaction.

## Conversational RAG Pipeline

### Dataset preparation and hosting

2 versions of the dataset are available:

- **[Retrieval dataset](https://huggingface.co/datasets/siulhin-vlad37/final):** 71 structured intent records containing
  cryptocurrency-related responses, prepared for the RAG pipeline.
- **[Fine-tuning dataset](https://huggingface.co/datasets/siulhin-vlad37/chatbot):** 238 question-and-answer examples converted
  into JSONL using Llama 2's instruction format.

The preparation workflow consisted of restructuring intent based records
and converting question and answer pairs from CSV into formatted
training examples using Python scripts.

### Loading and indexing

At runtime, the RAG application loads the retrieval dataset from
Hugging Face.

OpenAI embeddings represent its content numerically, and the resulting
representations are stored in a Chroma vector store for similarity
search.

### Retrieval

For each question, the retriever searches the vector store for relevant
information. The original implementation retrieves the six most
similar dataset entries.

Their content is formatted and supplied to GPT-4 as context alongside
the user's question.

### Response generation

LangChain connects the retriever, prompt templates, GPT-4 API calls
and output processing.

The prompt instructs the model to:

- Use the retrieved context when answering.
- Acknowledge when the answer is unknown.
- Keep responses brief.

These instructions guide generation but do not guarantee factual
correctness.

### Conversation history

The chatbot stores user's and its own messages during the session.

When a follow-up question depends on earlier messages, a separate
prompt reformulates it into a standalone question before retrieval.
It needs for questions that refer to concepts discussed earlier
in the conversation.

### Command-line interaction

The application provides a conversation loop in the terminal.

Users can ask questions about cryptocurrency and end the conversation
by entering `exit`, `quit` or `stop`.

## Llama 2 Fine-Tuning

The other fine-tuning workflow adapts Llama 2 7B using the custom
cryptocurrency question and answer dataset.

The implementation uses:

- **Hugging Face Transformers** for model and tokenizer loading.
- **PEFT and LoRA** for parameter-efficient adaptation.
- **BitsAndBytes** for 4-bit quantization.
- **TRL's SFTTrainer** for supervised fine-tuning.
- **TensorBoard** for monitoring training.

LoRA trains additional adapter parameters. The notebook combines this approach
with a quantized base model to reduce training memory requirements.

### Training data format

The training examples use Llama 2's instruction format:

```text
<s>[INST] Question [/INST] Answer </s>
```

Each JSONL record contains a `text` field holding one formatted
question and answer example.

### Model publishing

After fine-tuning, the trained LoRA adapter and base
Llama 2 model were merged and published as the resulting model and tokenizer:

**[View the published model](https://huggingface.co/siulhin-vlad37/Llama-2-7b-chat-bot)**

The model weights are hosted on Hugging Face. The interaction notebook downloads the model from that location.

### Notebook interface

The interaction notebook provides:

- A field for entering a question.
- A button to generate a response.
- An output field displaying the generated text.

The RAG chatbot and the fine-tuned model are separate implementations.
Running the GPT-4 RAG pipeline does not require the Llama 2 weights.

## Training Monitoring and Assessment

See **Section 5.4** and **Appendix Figures 1–4** in the
[project report](BSP_S3.pdf).

## Running the Project

### Compatibility

This repository preserves an academic implementation from 2023–2024.
Some dependencies, particularly LangChain, have changed their APIs
since the project was developed.

The original environment was not fully pinned. Running the project
with current library versions may require dependency adjustments or
code updates.

### RAG chatbot

The application requires:

- A Python environment with compatible LangChain, OpenAI, Chroma and
  Hugging Face Datasets dependencies.
- An OpenAI API key with access to the configured chat and embedding
  models.
- Internet access for dataset loading and API requests.

After configuring the environment and supplying credentials locally:

```bash
python RAG.py
```

Example questions:

```text
What is cryptocurrency?
What is an Initial Coin Offering?
What are the risks of cryptocurrency?
```

### Fine-tuning and inference notebooks

The notebooks were originally designed for Google Colab.

1. Open `Fine_tuning.ipynb` to inspect the model configuration and
   training workflow.
2. Open `Interaction.ipynb` to load the published model and generate
   responses through the notebook interface.

Fine-tuning and inference require sufficient memory for the selected
model and configuration. The original training workflow uses a GPU.

## Limitations

The knowledge base is small and reflects the original project period.
Answers may therefore be incomplete, outdated or unsupported.
