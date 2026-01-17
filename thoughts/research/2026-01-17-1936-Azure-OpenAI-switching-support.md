---
date: 2026-01-17T19:36:39+00:00
git_commit: 62bc1e251baae62d11ce4dcb807f9a58d17cca39
branch: main
repository: home-ai-assistant
topic: "Support for Switching between current Azure OpenAI and other standard OpenAI endpoint"
tags: [research, codebase, azure-openai, openai, llm]
status: complete
---

# Research: Support for Switching between Azure OpenAI and Standard OpenAI Endpoints

## Research Question
Support for Switching between current Azure OpenAI and other standard OpenAI endpoint. Focus on backend for now but build backend changes so frontend changes will be easy to make.

## Summary
The current backend implementation is hardcoded to use Azure OpenAI services exclusively. There is no abstraction layer or configuration mechanism that allows switching between Azure OpenAI and standard OpenAI endpoints. The system uses specific AzureChatOpenAI and AzureOpenAIEmbeddings classes from langchain_openai, with hard-coded environment variables for Azure-specific configuration.

## Detailed Findings

### Backend Implementation
- The backend is currently configured to use Azure OpenAI services exclusively
- Key components using Azure-specific implementations:
  - `AzureChatOpenAI` for chat completions (lines 33-37 in backend.py)
  - `AzureOpenAIEmbeddings` for embeddings (lines 42-47 in backend.py)
- Both implementations are initialized with hardcoded Azure-specific parameters including:
  - `azure_endpoint` 
  - `azure_deployment`
  - `api_version`
  - `api_key`

### Configuration Dependencies
- Environment variables are hardcoded for Azure services:
  - `AZURE_OPENAI_URL`
  - `AZURE_OPENAI_API_KEY` 
  - `EMB_OPENAI_URL`
  - `OPENAI_API_KEY`
- No mechanism exists to switch between different LLM providers dynamically

### Integration Points
- The LLM is used in the main chat streaming endpoint (lines 125-144 in backend.py)
- Embeddings are used for vector store retrieval (lines 56-60 in backend.py)
- The system uses LangChain's `create_retrieval_chain` with the configured LLM

## Code References
- `./backend/backend.py:33-37` - AzureChatOpenAI initialization
- `./backend/backend.py:42-47` - AzureOpenAIEmbeddings initialization
- `./backend/backend.py:125-144` - Main chat streaming endpoint using LLM
- `./backend/backend.py:56-60` - Vector store retrieval using embeddings

## Open Questions
1. Are there any plans or existing mechanisms for configuration-driven provider switching?
2. Would a factory pattern or dependency injection approach be suitable for supporting multiple LLM providers?
3. How would environment variable management need to change to support multiple provider configurations?
4. What would be the impact on the existing vector store and document retrieval systems when switching providers?
