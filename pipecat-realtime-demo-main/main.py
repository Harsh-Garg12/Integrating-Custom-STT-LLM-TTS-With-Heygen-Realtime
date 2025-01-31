from functools import partial
import logging
import os
from typing import Any
import signal
import sys

import aiohttp
import uvicorn
from openai._types import NOT_GIVEN
from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.deepgram import DeepgramSTTService, LiveOptions
from pipecat.services.openai import OpenAILLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService

# from pipecat.services.azure import AzureLLMService
# from openai.types.chat import ChatCompletionToolParam

from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.services.gemini_multimodal_live.gemini import(
    GeminiMultimodalLiveLLMService,
) 
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.transports.base_output import BaseOutputTransport, TransportParams
from heygen_video_service import HeyGenVideoService
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

# deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
# if not deepgram_api_key:
#     raise ValueError("DEEPGRAM_API_KEY must be set")

# openai_api_key=os.getenv("OPENAI_API_KEY")
# if not openai_api_key:
#     raise ValueError("OPENAI_API_KEY")

elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY")
if not elevenlabs_api_key:
    raise ValueError("ELEVENLABS_API_KEY must be set")

gemini_api_key=os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY must be set")

async def run_bot(
    websocket_client: WebSocket,
    session_id: str,
    session_token: str,
    realtime_endpoint: str,
) -> None:
    async with aiohttp.ClientSession() as session:
        params = VADParams(
            min_volume=0.6,
            start_secs=0.2,
            stop_secs=1.2,
            confidence=0.7,
        )
        transport = FastAPIWebsocketTransport(
            websocket=websocket_client,
            params=FastAPIWebsocketParams(
                audio_out_enabled=True,
                add_wav_header=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=params),
                vad_audio_passthrough=True,
                serializer=ProtobufFrameSerializer(),
            ),
        )

        # NOTE: Patching the receive_messages method to handle audio frames
        # https://github.com/pipecat-ai/pipecat/issues/296
        async def _patched_receive_messages(self: Any) -> None:
            async for message in self._websocket.iter_bytes():
                frame = self._params.serializer.deserialize(message)
                if not frame:
                    continue
                if isinstance(frame, AudioRawFrame):
                    await self.push_audio_frame(frame)
                else:
                    await self._internal_push_frame(frame)
            await self._callbacks.on_client_disconnected(self._websocket)

        transport._input._receive_messages = partial(_patched_receive_messages, transport._input)

        # stt = DeepgramSTTService(
        #     api_key=deepgram_api_key,
        #     live_options=LiveOptions(
        #         encoding="linear16",
        #         language= "en-US",
        #         model="nova-2-conversationalai",
        #         sample_rate=16000,
        #         channels=1,
        #         interim_results=True,
        #         smart_format=True,
        #     )
        # )
       
        # OPENAI Integration 
        # llm = OpenAILLMService(api_key=openai_api_key, model="gpt-4o-mini")
        # messages = [
        #     {
        #         "role": "system",
        #         "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way. Be brief, concise, and to the point.",
        #     },
        # ]

        # context = OpenAILLMContext(messages=messages)
        # context_aggregator = llm.create_context_aggregator(context)

        # Azure-OpenAI Integration
        # llm = AzureLLMService(
        #     api_key="your-azure-api-key",
        #     endpoint="https://your-resource.openai.azure.com",
        #     model="your-deployment-name"  # e.g., "gpt-4o"
        # )

        # # Define tools for function calling
        # tools = [
        #     ChatCompletionToolParam(
        #         type="function",
        #         function={
        #             "name": "get_current_weather",
        #             "description": "Get the current weather",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "location": {
        #                         "type": "string",
        #                         "description": "The city and state, e.g. San Francisco, CA"
        #                     },
        #                     "format": {
        #                         "type": "string",
        #                         "enum": ["celsius", "fahrenheit"],
        #                         "description": "The temperature unit to use"
        #                     }
        #                 },
        #                 "required": ["location", "format"]
        #             }
        #         }
        #     )
        # ]

        # # Create context with system message and tools
        # context = OpenAILLMContext(
        #     messages=[
        #         {
        #             "role": "system",
        #             "content": "You are a helpful assistant in a voice conversation. Keep responses concise."
        #         }
        #     ],
        #     tools=tools
        # )

        # # Register function handlers
        # async def fetch_weather(function_name, tool_call_id, args, llm, context, result_callback):
        #     await result_callback({"conditions": "nice", "temperature": "75"})

        # llm.register_function(None, fetch_weather)

        # # Create context aggregator for message handling
        # context_aggregator = llm.create_context_aggregator(context)

        # Google-cloud gemini integration
        llm = GeminiMultimodalLiveLLMService(
            api_key = gemini_api_key,
            voice_id = "Charon", # Puck, Charon, Kore, Fenrir, Aoede
            # system_instruction = "Talk like a pirate"
            transcribe_user_audio=True,
            transcribe_model_audio=False,
            inference_on_context_initialization=False,
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way. Be brief, concise, and to the point.",
            },
        ]

        tools = NOT_GIVEN # todo: implement tools in and set here
        context = OpenAILLMContext(messages=messages, tools=tools)
        context_aggregator = llm.create_context_aggregator(context)

        # ElevenLabs Integration
        tts = ElevenLabsTTSService(
            api_key=elevenlabs_api_key,
            voice_id="29vD33N1CtxCmqQRPOHJ",
            output_format="pcm_24000",
        )

        heygen_video_service = HeyGenVideoService(session_id=session_id, session_token=session_token, session=session, realtime_endpoint=realtime_endpoint)
        output_transport = BaseOutputTransport(TransportParams(audio_out_enabled=True))
        pipeline = Pipeline(
            [
                transport.input(),  # Websocket input from client
                # stt,  # Speech-To-Text
                context_aggregator.user(),
                llm,
                tts,
                heygen_video_service,
                output_transport,
                context_aggregator.assistant(),
            ]
        )
        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
            ),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport: Any, client: Any) -> None:
            logger.info("Client connected.")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport: Any, client: Any) -> None:
            logger.info("Client disconnected.")
            await task.queue_frames([EndFrame()])

        runner = PipelineRunner() if sys.platform != "win32" else None

        if runner:
            await runner.run(task)
        else:
            try:
                await task.run()
            except KeyboardInterrupt:
                logger.info("Shutting down gracefully...")

api_router = APIRouter()

@api_router.websocket("/user-audio-input")
async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
    session_token: str,
    realtime_endpoint: str,
) -> None:
    logger.info(f"WebSocket connection established with session_id: {session_id}")
    logger.info(f"realtime_endpoint: {realtime_endpoint}")
    await websocket.accept()
    await run_bot(
        websocket,
        session_id,
        session_token,
        realtime_endpoint,
    )
        
app = FastAPI()
app.include_router(router=api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 3001))
    workers = int(os.getenv("WORKERS", 1))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=workers)
