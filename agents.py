import logging
import asyncio
import uuid
import json
import os
import httpx
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime, timezone
from strands import Agent, tool
from strands.models.openai import OpenAIModel

load_dotenv()

# Configure logging to show our debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = FastAPI(title="Strands Agent Server", version="1.0.0")

model = OpenAIModel(
    client_args={
        "api_key": os.getenv("OPENAI_API_KEY"),
    },
    model_id="gpt-4o",
    params={
        "max_tokens": 1000,
        "temperature": 0.7,
    }
)

# ============================================
# TOOLS (No fillers - LLM sends filler as text content before calling)
# ============================================

@tool
async def get_weather(city: str) -> str:
    """Get the current real-time weather for any city worldwide.
    
    Args:
        city: The city to get weather for (e.g., "Delhi", "Tokyo", "New York")
    """
    # NOTE: Filler is now sent by LLM as text content BEFORE this tool executes
    # This ensures filler arrives within <1 second (LLM decision time, not tool execution time)
    
    try:
        async with httpx.AsyncClient() as client:
            # First get coordinates
            geo_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=10.0
            )
            geo_data = geo_response.json()
            
            if not geo_data.get("results"):
                yield f"Sorry, I couldn't find {city}. Please try a different city name."
                return
            
            result = geo_data["results"][0]
            lat, lon = result["latitude"], result["longitude"]
            location_name = f"{result['name']}, {result.get('country', '')}"
            
            # Get weather data
            weather_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "timezone": "auto"
                },
                timeout=10.0
            )
            weather_data = weather_response.json()
            
            current = weather_data.get("current", {})
            temp = current.get("temperature_2m", "N/A")
            feels_like = current.get("apparent_temperature", "N/A")
            humidity = current.get("relative_humidity_2m", "N/A")
            wind_speed = current.get("wind_speed_10m", "N/A")
            weather_code = current.get("weather_code", 0)
            
            # Weather code descriptions
            weather_descriptions = {
                0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                45: "Foggy", 48: "Depositing rime fog",
                51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
                61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
                71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
                80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
                95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail"
            }
            condition = weather_descriptions.get(weather_code, "Unknown conditions")
            
            yield f"The current weather in {location_name} is {temp}°C with {condition}. It feels like {feels_like}°C. Humidity is {humidity}% and wind speed is {wind_speed} km/h."
            
    except Exception as e:
        yield f"Sorry, I couldn't fetch the weather for {city}. Please try again."


@tool
async def get_current_time() -> str:
    """Get the current date and time."""
    # Filler comes from LLM text, not here
    now = datetime.now()
    yield f"The current date and time is {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    """
    # Filler comes from LLM text, not here
    try:
        allowed_chars = set("0123456789+-*/.() ")
        if not all(c in allowed_chars for c in expression):
            yield "Error: Invalid characters. Only numbers and basic operators allowed."
            return
        result = eval(expression)
        yield f"The result of {expression} is {result}"
    except Exception as e:
        yield f"Error calculating: {str(e)}"


@tool
async def set_reminder(reminder_text: str, minutes: int) -> str:
    """Set a reminder for the user.
    
    Args:
        reminder_text: What to be reminded about
        minutes: How many minutes from now to be reminded
    """
    # Filler comes from LLM text, not here
    yield f"I've set a reminder for you in {minutes} minutes: '{reminder_text}'"


# Initialize Strands agent with all tools
strands_agent = Agent(
    model=model,
    tools=[get_weather, get_current_time, calculate, set_reminder],
    system_prompt="""You are a friendly and helpful AI assistant. You have access to:
- get_weather: Get real-time weather for any city
- get_current_time: Get the current date and time
- calculate: Evaluate math expressions
- set_reminder: Set reminders for the user

CRITICAL FILLER REQUIREMENT:
When you decide to use ANY tool, you MUST FIRST output a brief, natural filler phrase as your text response BEFORE calling the tool.
This filler phrase will be spoken to the user immediately while the tool executes.

Examples of good filler phrases (vary them naturally):
- "Let me check that for you."
- "One moment, I'll look that up."
- "Sure, let me find that information."
- "Checking that now..."

The filler MUST be your text output, NOT part of the tool call.

IMPORTANT: When a tool returns a result, DO NOT repeat or rephrase the information.
The tool output is already spoken to the user. Just add a brief friendly comment if needed.
For example, if weather tool says "22°C and sunny", say something like "Sounds like a nice day!" NOT "It's 22°C and sunny."

Always be polite, engaging, and conversational."""
)

class InvocationRequest(BaseModel):
    input: Dict[str, Any]

class InvocationResponse(BaseModel):
    output: Dict[str, Any]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]

@app.post("/invocations", response_model=InvocationResponse)
async def invoke_agent(request: InvocationRequest):
    try:
        user_message = request.input.get("prompt", "")
        if not user_message:
            raise HTTPException(
                status_code=400,
                detail="No prompt found in input. Please provide a 'prompt' key in the input."
            )

        result = strands_agent(user_message)
        response = {
            "message": result.message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "strands-agent",
        }

        return InvocationResponse(output=response)

    except Exception as e:
        logger.error(f"Agent processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent processing failed: {str(e)}")

@app.get("/ping")
async def ping():
    return {"status": "healthy"}


# @app.post("/stream")
# async def stream_response(request: InvocationRequest):
#     user_message = request.input.get("prompt", "")
#     async def generate():
#         # agent = Agent(
#         #     tools=[call_api],
#         #     callback_handler=None
#         # )

#         try:
#             async for event in strands_agent.stream_async(user_message):
#                 if "data" in event:
#                     logger.error(f"got the data {event['data']}")
#                     # Only stream text chunks to the client
#                     yield event["data"]
#                 else:
#                     logger.error("in else")
#                     logger.error(event)
#         except Exception as e:
#             yield f"Error: {str(e)}"

#     return StreamingResponse(
#         generate(),
#         media_type="text/plain"
#     )

@app.post("/stream")
async def stream_response(request: InvocationRequest):
    user_message = request.input.get("prompt", "")

    async def generate():
        try:
            async for event in strands_agent.stream_async(user_message):
                # Check if this is a tool streaming event
                if "tool_stream_event" in event:
                    data = event["tool_stream_event"].get("data")
                    if data is not None:
                        yield data
                elif "data" in event:
                    # Other (non-tool) events
                    yield event["data"]
                else:
                    # Fallback / debug
                    logger.debug("Other event: %s", event)
        except Exception as e:
            yield f"Error: {str(e)}\n"

    return StreamingResponse(generate(), media_type="text/plain")



@app.post("/stream")
async def stream_response(request: InvocationRequest):
    user_message = request.input.get("prompt", "")

    async def generate():
        async for event in strands_agent.stream_async(user_message):
            if "tool_stream_event" in event:
                yield event["tool_stream_event"]["data"] + "\n"

    return StreamingResponse(generate(), media_type="text/plain")



@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with IMMEDIATE filler streaming.
    
    SIMPLIFIED FLOW (per Jan 13 meeting):
    1. User query arrives
    2. LLM decides to call tool → Send filler IMMEDIATELY (< 1 second)
    3. Tool executes (natural pause - this is the 2-5 second wait)
    4. Tool result streams to user
    
    The pause between filler and response is NATURAL (tool execution time).
    No artificial delays or SSML needed.
    """
    import time
    start_time = time.time()
    
    user_message = request.messages[-1].get("content", "")
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    def elapsed_ms():
        return int((time.time() - start_time) * 1000)
    
    def make_chunk(content: str, finish_reason=None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": finish_reason}],
        }
    
    async def generate():
        tool_was_called = False
        llm_text_streamed = False  # Track if LLM has streamed any text (filler)
        
        # Log file for debugging
        with open('/tmp/strands_debug.log', 'a') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"[NEW REQUEST] T+0ms | {user_message}\n")
        
        try:
            async for event in strands_agent.stream_async(user_message):
                event_keys = list(event.keys()) if isinstance(event, dict) else []
                
                # DEBUG: Log ALL events to find tool initiation pattern
                with open('/tmp/strands_debug.log', 'a') as f:
                    event_str = str(event)[:300] if isinstance(event, dict) else str(type(event))
                    f.write(f"[EVENT] T+{elapsed_ms()}ms | keys={event_keys} | {event_str}\n")
                
                # ===== DETECT TOOL CALL =====
                # Based on actual Strands events, we detect:
                # 1. contentBlockStart with toolUse (first signal)
                # 2. type == 'tool_use_stream' (subsequent signals)
                is_tool_init = False
                tool_name = None
                
                if isinstance(event, dict):
                    # Pattern 1: contentBlockStart with toolUse (FIRST signal at ~1.4s)
                    inner_event = event.get("event", {})
                    if isinstance(inner_event, dict):
                        content_block_start = inner_event.get("contentBlockStart", {})
                        if isinstance(content_block_start, dict):
                            start_data = content_block_start.get("start", {})
                            if isinstance(start_data, dict) and "toolUse" in start_data:
                                is_tool_init = True
                                tool_name = start_data["toolUse"].get("name")
                    
                    # Pattern 2: type == 'tool_use_stream' (subsequent tool streaming)
                    if not is_tool_init and event.get("type") == "tool_use_stream":
                        is_tool_init = True
                        current_tool = event.get("current_tool_use", {})
                        if isinstance(current_tool, dict):
                            tool_name = current_tool.get("name")
                
                # ===== TOOL CALL DETECTED =====
                # Per Jan 13 meeting: Filler must come from LLM, not hardcoded.
                # The LLM's filler text is already streamed via 'data' events BEFORE this point.
                # 
                # KEY FIX: When tool is detected, force-flush the filler by sending finish signal.
                # This ensures TTS speaks the filler BEFORE tool result arrives.
                if is_tool_init and not tool_was_called:
                    tool_was_called = True
                    with open('/tmp/strands_debug.log', 'a') as f:
                        f.write(f"[TOOL DETECTED] T+{elapsed_ms()}ms | tool={tool_name} | FORCING FILLER FLUSH\n")
                    
                    # Force flush: Send finish signal to make TTS speak filler immediately
                    # This creates a natural break before tool result
                    if llm_text_streamed:
                        yield f"data: {json.dumps(make_chunk('', finish_reason='tool_calls'))}\n\n"
                        await asyncio.sleep(0)
                        with open('/tmp/strands_debug.log', 'a') as f:
                            f.write(f"[FILLER FLUSHED] T+{elapsed_ms()}ms | finish_reason=tool_calls sent\n")
                elif is_tool_init:
                    # Already detected tool, skip duplicate detection
                    pass
                
                # ===== TOOL RESULT (comes after natural pause from tool execution) =====
                if "tool_stream_event" in event:
                    data = event["tool_stream_event"].get("data")
                    if data:
                        tool_was_called = True
                        with open('/tmp/strands_debug.log', 'a') as f:
                            f.write(f"[TOOL RESULT] T+{elapsed_ms()}ms | '{data[:50]}...'\n")
                        yield f"data: {json.dumps(make_chunk(data))}\n\n"
                        await asyncio.sleep(0)
                    continue
                
                # ===== LLM TEXT (filler before tool, or normal response) =====
                # Per Jan 13 meeting: LLM outputs filler as text BEFORE tool call.
                # This text is streamed immediately - no hardcoded fillers.
                if "data" in event:
                    data = event["data"]
                    if data:
                        # If tool was already called, block LLM's post-tool rephrasing
                        # (the tool result speaks for itself)
                        if tool_was_called and llm_text_streamed:
                            with open('/tmp/strands_debug.log', 'a') as f:
                                f.write(f"[BLOCKED] T+{elapsed_ms()}ms | '{data[:30]}...'\n")
                            continue
                        
                        # Stream LLM text immediately (this IS the filler from LLM!)
                        llm_text_streamed = True
                        with open('/tmp/strands_debug.log', 'a') as f:
                            f.write(f"[LLM FILLER] T+{elapsed_ms()}ms | '{data}'\n")
                        yield f"data: {json.dumps(make_chunk(data))}\n\n"
                        await asyncio.sleep(0)
                    continue
                
                # ===== FALLBACK: Delta format =====
                if "delta" in event:
                    content = event.get("delta", {}).get("content") if isinstance(event.get("delta"), dict) else None
                    if content and not tool_was_called:
                        yield f"data: {json.dumps(make_chunk(content))}\n\n"
                        await asyncio.sleep(0)
            
            with open('/tmp/strands_debug.log', 'a') as f:
                f.write(f"[DONE] T+{elapsed_ms()}ms\n")
            yield f"data: {json.dumps(make_chunk('', finish_reason='stop'))}\n\n"
            yield "data: [DONE]\n\n"
                
        except Exception as e:
            with open('/tmp/strands_debug.log', 'a') as f:
                f.write(f"[ERROR] T+{elapsed_ms()}ms | {e}\n")
            yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")





if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("STRANDS SERVER STARTING - DEBUG MODE v2")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8080)