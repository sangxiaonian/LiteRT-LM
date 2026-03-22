# LiteRT-LM Python

Python bindings for LiteRT-LM, a production-ready, open-source inference
framework designed to deliver high-performance, cross-platform LLM deployments
on edge devices.

## Usage

Here is a simple example showing how to load a model and interact with it using
the Python API:

```python
import litert_lm

# Load the model using the CPU backend
with litert_lm.Engine("path/to/model.litertlm") as engine:
    # Create a conversation and generate a response
    with engine.create_conversation() as conversation:
        user_message = "Hello world!"

        # Alternative: send the full message for prompt template
        # user_message = {"role": "user", "content": "Hello world!"}

        # Synchronous completion
        response = conversation.send_message(user_message)
        print("Response:", response["content"][0]["text"])

        # Asynchronous / Streaming completion
        # stream = conversation.send_message_async(user_message)
        # for text_piece in stream:
        #     print(text_piece, end="", flush=True)
```

For more information, please visit the [main repository](https://github.com/google-ai-edge/LiteRT-LM).
