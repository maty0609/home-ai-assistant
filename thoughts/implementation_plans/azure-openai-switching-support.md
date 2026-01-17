# Implementation Plan: Switching Between Azure OpenAI and Standard OpenAI Endpoints

## Overview

This document outlines a detailed implementation plan for adding support to switch between Azure OpenAI and standard OpenAI endpoints in the backend system. The solution will maintain backward compatibility with existing Azure OpenAI configurations while enabling future flexibility to support other LLM providers.

## Problem Statement

Currently, the backend system is hardcoded to use Azure OpenAI services exclusively. The implementation uses:
- `AzureChatOpenAI` for chat completions
- `AzureOpenAIEmbeddings` for embeddings
- Hardcoded environment variables for Azure-specific configurations

There is no abstraction layer or configuration mechanism to switch between different LLM providers.

## Solution Approach

We'll implement a configuration-driven factory pattern that allows dynamic selection of LLM providers at runtime while maintaining backward compatibility.

## Key Components to Modify

### 1. Configuration Management
- Introduce a provider selection configuration
- Maintain backward compatibility with existing Azure configurations

### 2. LLM Provider Factory
- Create a factory class that instantiates appropriate LLM components based on configuration
- Support for both Azure OpenAI and standard OpenAI providers

### 3. Backend Integration Points
- Update `backend.py` to use the factory pattern
- Ensure seamless transition for chat streaming and vector store operations

## Implementation Phases

### Phase 1: Configuration Setup

#### Changes to Environment Variables
```bash
# Add to .env file
LLM_PROVIDER=azure_openai  # or "openai"

# Keep existing Azure variables for backward compatibility
AZURE_OPENAI_URL=
AZURE_OPENAI_API_KEY=
EMB_OPENAI_URL=
OPENAI_API_KEY=

# New variables for standard OpenAI
OPENAI_BASE_URL=
OPENAI_API_KEY=
```

#### Configuration Class
Create a configuration manager to handle provider selection and parameter mapping:

```python
class LLMConfig:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "azure_openai")
        self.azure_config = {
            "api_base": os.getenv("AZURE_OPENAI_URL"),
            "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "deployment_name": os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4"),
            "embedding_deployment_name": os.getenv("EMB_AZURE_DEPLOYMENT_NAME", "text-embedding-ada-002")
        }
        self.openai_config = {
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "api_key": os.getenv("OPENAI_API_KEY")
        }
```

### Phase 2: LLM Provider Factory

#### Factory Class Implementation
```python
class LLMProviderFactory:
    @staticmethod
    def create_chat_model(config: LLMConfig):
        if config.provider == "azure_openai":
            return AzureChatOpenAI(
                api_base=config.azure_config["api_base"],
                api_key=config.azure_config["api_key"],
                deployment_name=config.azure_config["deployment_name"]
            )
        elif config.provider == "openai":
            return ChatOpenAI(
                base_url=config.openai_config["base_url"],
                api_key=config.openai_config["api_key"]
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    @staticmethod
    def create_embeddings(config: LLMConfig):
        if config.provider == "azure_openai":
            return AzureOpenAIEmbeddings(
                api_base=config.azure_config["api_base"],
                api_key=config.azure_config["api_key"],
                deployment=config.azure_config["embedding_deployment_name"]
            )
        elif config.provider == "openai":
            return OpenAIEmbeddings(
                base_url=config.openai_config["base_url"],
                api_key=config.openai_config["api_key"]
            )
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")
```

### Phase 3: Backend Integration

#### Updated backend.py
Replace hardcoded Azure implementations with factory-based approach:

```python
# Replace lines 30-35 in backend.py
# Old approach:
# llm = AzureChatOpenAI(
#     api_base=os.getenv("AZURE_OPENAI_URL"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4")
# )

# New approach:
llm_config = LLMConfig()
llm = LLMProviderFactory.create_chat_model(llm_config)

# Replace lines 38-43 in backend.py
# Old approach:
# embeddings = AzureOpenAIEmbeddings(
#     api_base=os.getenv("EMB_OPENAI_URL"),
#     api_key=os.getenv("OPENAI_API_KEY"),
#     deployment=os.getenv("EMB_AZURE_DEPLOYMENT_NAME", "text-embedding-ada-002")
# )

# New approach:
embeddings = LLMProviderFactory.create_embeddings(llm_config)
```

### Phase 4: Frontend Considerations

While the primary focus is on backend implementation, we'll ensure frontend integration is straightforward:

#### API Endpoint Updates
- The existing API endpoints (`/chat_stream` and `/chat`) will remain unchanged
- Frontend only needs to configure the appropriate provider in environment variables

#### Configuration Options
Frontend developers can easily switch providers by setting:
```bash
LLM_PROVIDER=openai  # for standard OpenAI
LLM_PROVIDER=azure_openai  # for Azure OpenAI (default)
```

## Backward Compatibility

The implementation maintains full backward compatibility:
- Default behavior remains Azure OpenAI (no breaking changes)
- Existing `.env` configurations continue to work unchanged
- All existing functionality preserved

## Testing Strategy

### Unit Tests
- Test factory creation for both providers
- Validate configuration loading
- Verify correct instantiation of LLM components

### Integration Tests
- Test chat streaming endpoint with Azure OpenAI
- Test chat streaming endpoint with standard OpenAI
- Test vector store operations with both providers

### Smoke Tests
- Verify existing functionality unchanged
- Confirm smooth switching between providers

## Success Criteria

1. ✅ Ability to switch between Azure OpenAI and standard OpenAI via configuration
2. ✅ Backward compatibility maintained for existing deployments
3. ✅ Minimal code changes required
4. ✅ Clear documentation for developers
5. ✅ Comprehensive test coverage

## Risk Mitigation

### Potential Issues
1. **Configuration Confusion**: Developers might mix up environment variable names
   - Solution: Clear documentation and validation logic

2. **Provider-Specific Parameters**: Different providers may require different parameters
   - Solution: Factory pattern handles provider-specific configurations

3. **Performance Differences**: Different providers may have varying performance characteristics
   - Solution: Monitor and optimize as needed

## Future Extensibility

This implementation lays the groundwork for supporting additional LLM providers:
- Cohere
- Anthropic
- Hugging Face
- Local models

The factory pattern makes it easy to add new providers without modifying existing code.

## Migration Steps

1. Update `.env` file with `LLM_PROVIDER` variable
2. Deploy updated backend code
3. Test with existing Azure configuration (should work unchanged)
4. Test with new OpenAI configuration
5. Update documentation as needed

## Conclusion

This implementation plan provides a robust, extensible solution for switching between Azure OpenAI and standard OpenAI endpoints. The factory pattern approach ensures clean separation of concerns while maintaining simplicity and backward compatibility.
