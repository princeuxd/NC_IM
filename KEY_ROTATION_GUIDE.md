# LLM Key Rotation System

## Overview

The NC_IM toolkit now includes an advanced **automatic key rotation system** that handles rate limits seamlessly by cycling through multiple API keys and falling back between different LLM providers.

## Supported Providers

The system supports **three LLM providers** in the following fallback order:

1. **OpenRouter** (primary) - `google/gemini-2.0-flash-exp:free`
2. **Groq** (secondary) - `llama3-8b-8192`
3. **Gemini** (tertiary) - `gemini-2.0-flash-exp`

## Key Features

✅ **Multiple Keys Per Provider**: Configure multiple API keys for each provider  
✅ **Automatic Rotation**: Cycles through keys when rate limits are hit  
✅ **Provider Fallback**: Falls back to next provider when all keys for current provider are exhausted  
✅ **Rate Limit Detection**: Intelligently detects rate limit errors vs other API errors  
✅ **Smart Recovery**: Temporarily marks keys as unavailable with automatic recovery  
✅ **Backward Compatibility**: Existing single-key configurations still work

## Configuration

### Environment Variables

Configure multiple keys using numbered suffixes:

```bash
# OpenRouter Keys (4 keys)
OPENROUTER_API_KEY_1=
OPENROUTER_API_KEY_2=
OPENROUTER_API_KEY_3=
OPENROUTER_API_KEY_4=

# Groq Keys (3 keys)
GROQ_API_KEY_1=
GROQ_API_KEY_2=
GROQ_API_KEY_3=

# Gemini Keys (1 key)
GEMINI_API_KEY_1=

# Model Configuration
OPENROUTER_CHAT_MODEL=google/gemini-2.0-flash-exp:free
GROQ_CHAT_MODEL=llama3-8b-8192
GEMINI_CHAT_MODEL=gemini-2.0-flash-exp
```

### Backward Compatibility

The system also supports the original single-key format:

```bash
OPENROUTER_API_KEY=your_single_key
GROQ_API_KEY=your_single_key
GEMINI_API_KEY=your_single_key
```

## How It Works

### 1. Key Rotation Logic

- **Round-robin rotation**: Keys are used in sequence within each provider
- **Rate limit detection**: Automatically detects rate limit errors (429, "rate limit", "quota exceeded", etc.)
- **Cooldown periods**: Rate-limited keys are marked unavailable for 60 minutes
- **Error handling**: Invalid keys get 24-hour cooldown, other errors get 10-minute cooldown after 3 failures

### 2. Provider Fallback

When all keys for a provider are exhausted:

1. **OpenRouter** (tries all 4 keys) → **Groq** (tries all 3 keys) → **Gemini** (tries 1 key)
2. If all providers are exhausted, returns `RuntimeError: "All API keys exhausted"`

### 3. Smart Client Usage

The system is transparent to existing code. Simply use:

```python
from llms import get_smart_client

# Automatically handles key rotation and provider fallback
client = get_smart_client()
response = client.chat([
    {"role": "user", "content": "Your message"}
], temperature=0.3, max_tokens=500)
```

## Architecture

### New Components

1. **`llms/key_manager.py`**: Core rotation logic and key status tracking
2. **`llms/smart_client.py`**: Wrapper client with automatic fallback
3. **`llms/gemini.py`**: New Gemini provider implementation
4. **`config/settings.py`**: Updated to support multiple keys per provider

### Updated Components

- **`simple_streamlit_app.py`**: Replaced manual fallback logic with smart client
- **`analysis/video_vision.py`**: Uses smart client for vision analysis
- **`llms/base.py`**: Added `get_smart_client()` factory function
- **`requirements.txt`**: Added `google-generativeai>=0.8.0`

## Usage Examples

### Basic Chat

```python
from llms import get_smart_client

client = get_smart_client()
response = client.chat([
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Explain quantum computing in simple terms"}
])
print(response)
```

### Monitoring Key Status

```python
from llms import get_smart_client

client = get_smart_client()
status = client.get_status_summary()

print("Key Status Summary:")
for provider, info in status.items():
    if provider != 'timestamp':
        print(f"{provider}: {info['available']}/{info['total']} available")
```

### Error Handling

```python
from llms import get_smart_client

try:
    client = get_smart_client()
    response = client.chat([{"role": "user", "content": "Hello"}])
except RuntimeError as e:
    if "All API keys exhausted" in str(e):
        print("All LLM providers are rate limited. Try again later.")
    else:
        print(f"LLM error: {e}")
```

## Rate Limit Management

### Detection

The system automatically detects rate limits by checking for:

- HTTP status code 429
- Error messages containing "rate limit", "quota", "too many requests"
- Provider-specific rate limit indicators

### Recovery

- **Rate-limited keys**: 60-minute cooldown
- **Invalid keys**: 24-hour cooldown
- **Other errors**: 10-minute cooldown after 3 consecutive failures
- **Successful calls**: Immediately reset error counters and cooldowns

### Best Practices

1. **Distribute usage**: The rotation ensures even distribution across keys
2. **Monitor status**: Check key status regularly in high-usage scenarios
3. **Plan capacity**: With 8 total keys (4 OR + 3 Groq + 1 Gemini), you have significant capacity
4. **Add more keys**: Simply add `PROVIDER_API_KEY_5`, `PROVIDER_API_KEY_6`, etc.

## Migration Guide

### From Old Manual Fallback

**Before** (manual fallback):

```python
try:
    client = get_client("openrouter", SETTINGS.openrouter_api_key)
    response = client.chat(messages)
except Exception as e:
    # Manual Groq fallback
    if groq_key:
        g_client = get_client("groq", groq_key)
        response = g_client.chat(messages)
```

**After** (automatic):

```python
client = get_smart_client()
response = client.chat(messages)  # Automatic rotation + fallback
```

### Updating Existing Code

1. Replace `get_client(provider, key)` with `get_smart_client()`
2. Remove manual fallback logic
3. Remove `model=` parameter from chat calls (handled automatically)
4. Add multiple keys to environment variables

## Testing

### Configuration Test

```bash
python -c "
from config.settings import SETTINGS
print(f'OpenRouter: {len(SETTINGS.openrouter_api_keys)} keys')
print(f'Groq: {len(SETTINGS.groq_api_keys)} keys')
print(f'Gemini: {len(SETTINGS.gemini_api_keys)} keys')
"
```

### Smart Client Test

```bash
python -c "
from llms import get_smart_client
client = get_smart_client()
response = client.chat([{'role': 'user', 'content': 'Hello'}])
print(f'Response: {response}')
print(f'Provider: {client.get_current_provider()}')
"
```

## Troubleshooting

### No Keys Configured

**Error**: `RuntimeError: All API keys exhausted across all providers`  
**Solution**: Check that environment variables are set correctly

### Keys Not Loading

**Error**: `OpenRouter: 0 keys, Groq: 0 keys, Gemini: 0 keys`  
**Solution**:

- Ensure `.env` file exists and contains the keys
- Check for typos in variable names (must be exact: `OPENROUTER_API_KEY_1`)
- Restart application to reload environment

### Rate Limits

**Error**: Frequent "rate limited" messages  
**Solution**:

- Add more API keys with higher numbers (`OPENROUTER_API_KEY_5`, etc.)
- Check key status: `client.get_status_summary()`
- Consider upgrading to paid plans for higher rate limits

### Provider-Specific Issues

- **OpenRouter**: Ensure model `google/gemini-2.0-flash-exp:free` is available
- **Groq**: Verify `llama3-8b-8192` model access
- **Gemini**: Check that `google-generativeai` package is installed

## Benefits

✅ **Automatic handling**: No more manual rate limit management  
✅ **High availability**: 8 total API keys across 3 providers  
✅ **Transparent**: Existing code works without changes  
✅ **Intelligent**: Smart error detection and recovery  
✅ **Scalable**: Easy to add more keys or providers  
✅ **Reliable**: Comprehensive error handling and logging

The key rotation system ensures your application can handle high-volume usage without interruption, automatically managing rate limits across multiple providers and keys.
