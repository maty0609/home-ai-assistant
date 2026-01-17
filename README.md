# home-ai-assistant

This project provides an AI assistant with RAG (Retrieval-Augmented Generation) capabilities. It supports switching between Azure OpenAI and standard OpenAI endpoints.

## LLM Provider Configuration

The system supports switching between different LLM providers through configuration. By default, it uses Azure OpenAI for backward compatibility.

### Environment Variables

Add the following to your `.env` file:

```bash
# Select LLM provider: "azure_openai" or "openai"
LLM_PROVIDER=azure_openai  # or "openai"

# For Azure OpenAI (keep existing variables for backward compatibility)
AZURE_OPENAI_URL=your_azure_endpoint
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_DEPLOYMENT_NAME=gpt-4
EMB_AZURE_DEPLOYMENT_NAME=text-embedding-3-large

# For standard OpenAI
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_api_key
```

### Usage Examples

#### Using Azure OpenAI (default)
```bash
LLM_PROVIDER=azure_openai
AZURE_OPENAI_URL=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_DEPLOYMENT_NAME=gpt-4
EMB_AZURE_DEPLOYMENT_NAME=text-embedding-3-large
```

#### Using OpenAI
```bash
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your-openai-key
```

The system will automatically select the appropriate LLM components based on the configuration.
