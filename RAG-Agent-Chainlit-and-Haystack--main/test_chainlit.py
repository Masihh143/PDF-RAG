import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hello! I am a Chainlit assistant. How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here
    await cl.Message(
        content=f"Received: {message.content}",
    ).send()
