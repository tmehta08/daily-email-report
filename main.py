import json

import ollama  # third-party


# --- Tool definitions (what the model can see) ---
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use this for any math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '2 + 2' or '(17 * 3) + 5'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
]


# --- Tool implementations ---
def run_tool(name: str, args: dict) -> str:
    if name == "calculator":
        try:
            return json.dumps({"result": eval(args["expression"])})
        except Exception as e:
            return json.dumps({"error": str(e)})
    elif name == "get_weather":
        fake_data = {
            "london": ("14°C", "Rainy"),
            "tokyo": ("22°C", "Sunny"),
            "new york": ("8°C", "Windy"),
        }
        city = args["city"].lower()
        temp, cond = fake_data.get(city, ("20°C", "Clear"))
        return json.dumps({"city": args["city"], "temp": temp, "condition": cond})
    return json.dumps({"error": f"Unknown tool: {name}"})


# --- Conversation memory (persists across calls) ---
messages = []


# --- The agent loop (with streaming) ---
def agent(user_message: str):
    messages.append({"role": "user", "content": user_message})

    while True:
        # Stream the response chunk by chunk
        stream = ollama.chat(
            model="qwen3:1.7b",
            messages=messages,
            tools=tools,
            stream=True,
        )

        # Collect the full response as we stream it
        full_content = ""
        tool_calls = []

        print("\nAgent: ", end="", flush=True)

        for chunk in stream:
            # Stream text to the terminal as it arrives
            if chunk.message.content:
                print(chunk.message.content, end="", flush=True)
                full_content += chunk.message.content

            # Collect tool calls (these arrive at the end)
            if chunk.message.tool_calls:
                tool_calls.extend(chunk.message.tool_calls)

        print()  # newline after streaming finishes

        # If no tool calls, we're done
        if not tool_calls:
            messages.append({"role": "assistant", "content": full_content})
            return

        # Process tool calls
        messages.append({"role": "assistant", "content": full_content, "tool_calls": tool_calls})

        for tool_call in tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            print(f"  [tool] {name}({args})")

            result = run_tool(name, args)
            print(f"  [result] {result}")

            messages.append({"role": "tool", "content": result})


if __name__ == "__main__":
    print("AI Agent ready. Tools: calculator, get_weather")
    print("Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue
        agent(user_input)
        print()
