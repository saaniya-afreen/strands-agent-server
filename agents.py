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

logger = logging.getLogger(__name__)

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
# TOOLS WITH STREAMING FILLERS
# ============================================

@tool
async def get_weather(city: str) -> str:
    """Get the current real-time weather for any city worldwide.
    
    Args:
        city: The city to get weather for (e.g., "Delhi", "Tokyo", "New York")
    """
    # Stream filler immediately (with line break for visual separation)
    yield f"Let me check the weather in {city} for you... one moment.\n\n"
    
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
    yield "Let me check the time for you...\n\n"
    now = datetime.now()
    yield f"The current date and time is {now.strftime('%A, %B %d, %Y at %I:%M %p')}"


@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "10 * 5")
    """
    yield "Let me calculate that for you...\n\n"
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
    yield f"Setting that reminder for you...\n\n"
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
    OpenAI-compatible chat completions endpoint with proper filler + pause streaming.
    
    Flow:
    1. Tool filler streams immediately ("Let me check the weather...")
    2. PAUSE (1.5 seconds) - creates audible gap
    3. Tool result streams ("The weather is 24°C...")
    4. LLM rephrase is BLOCKED (to prevent double response)
    """
    user_message = request.messages[-1].get("content", "")
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    logger.info(f"[STRANDS] Received: {user_message}")
    
    def make_chunk(content: str, finish_reason=None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": finish_reason}],
        }
    
    async def generate():
        tool_was_called = False
        tool_stream_count = 0
        
        try:
            async for event in strands_agent.stream_async(user_message):
                logger.info(f"[EVENT] {event}")
                
                # TOOL STREAMING - this is where filler + result come from
                if "tool_stream_event" in event:
                    data = event["tool_stream_event"].get("data")
                    if data:
                        tool_was_called = True
                        tool_stream_count += 1
                        
                        # Stream the content
                        logger.info(f"[TOOL #{tool_stream_count}] {data[:50]}...")
                        yield f"data: {json.dumps(make_chunk(data))}\n\n"
                        
                        # After FIRST tool stream (the filler), add PAUSE
                        if tool_stream_count == 1:
                            logger.info("[PAUSE] Adding 3s gap after filler...")
                            await asyncio.sleep(3.0)
                        else:
                            await asyncio.sleep(0)
                
                # LLM RESPONSE - only allow if NO tool was called
                elif "data" in event:
                    data = event["data"]
                    if data:
                        if tool_was_called:
                            # BLOCK LLM rephrase when tool was used
                            logger.info(f"[BLOCKED] LLM rephrase: {data[:50]}...")
                            continue
                        else:
                            # Normal LLM response (no tool involved)
                            logger.info(f"[LLM] {data[:50]}...")
                            yield f"data: {json.dumps(make_chunk(data))}\n\n"
                            await asyncio.sleep(0)
                
                # Delta format (fallback)
                elif "delta" in event:
                    content = event["delta"].get("content")
                    if content and not tool_was_called:
                        yield f"data: {json.dumps(make_chunk(content))}\n\n"
                        await asyncio.sleep(0)
            
            yield f"data: {json.dumps(make_chunk('', finish_reason='stop'))}\n\n"
            yield "data: [DONE]\n\n"
                
        except Exception as e:
            logger.error(f"[ERROR] {e}")
            yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)