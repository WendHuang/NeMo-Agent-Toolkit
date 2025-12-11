# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.context import Context, ContextState
from nat.cli.register_workflow import register_middleware
from nat.data_models.component_ref import MemoryRef
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware
from nat.middleware.middleware import CallNext, FunctionMiddlewareContext

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConversationMemoryMiddlewareConfig(FunctionMiddlewareBaseConfig, name="conversation_memory"):
    """
    Middleware configuration for automatically managing conversation memory.

    Automatically retrieves historical memory before each workflow invocation,
    and automatically saves the conversation after each invocation.
    """
    memory: MemoryRef = Field(
        description="Reference to the memory component name"
    )
    auto_retrieve: bool = Field(
        default=True,
        description="Whether to automatically retrieve historical memory before invocation"
    )
    auto_save: bool = Field(
        default=True,
        description="Whether to automatically save the conversation after invocation"
    )
    top_k: int = Field(
        default=3,
        description="Number of historical memories to retrieve"
    )


@register_middleware(config_type=ConversationMemoryMiddlewareConfig)
async def conversation_memory_middleware(config: ConversationMemoryMiddlewareConfig, builder: Builder):
    """
    Register middleware for automatically managing conversation memory.
    """
    # Get memory client
    memory_client = await builder.get_memory_client(config.memory)

    class ConversationMemoryMiddleware(FunctionMiddleware):
        """
        Conversation memory middleware.

        Four-phase processing:
        1. Preprocess: Automatically retrieve historical memory and inject into context
        2. Call Next: Invoke the actual workflow function
        3. Postprocess: Automatically save current conversation to memory
        4. Continue: Return the result
        """

        def _get_user_id(self) -> str:
            """Get user identifier from Context"""
            try:
                context_state = ContextState.get()
                context = Context(context_state)

                # Prioritize using conversation_id
                if context.conversation_id:
                    return context.conversation_id

                # Or get from metadata
                metadata = context.metadata
                if metadata and hasattr(metadata, 'cookies'):
                    user_id = metadata.cookies.get('user_id')
                    if user_id:
                        return user_id
            except Exception as e:
                logger.debug(f"Failed to get user_id from context: {e}")

            return "default_user"

        async def function_middleware_invoke(
            self, 
            value: Any, 
            call_next: CallNext,
            context: FunctionMiddlewareContext
        ) -> Any:
            """
            Wrap workflow invocation to automatically manage memory.
            """
            from nat.data_models.api_server import ChatRequest, ChatResponse, Message, ChatRequestOrMessage
            from nat.utils.type_converter import GlobalTypeConverter

            user_id = self._get_user_id()

            # Convert input to ChatRequest
            try:
                chat_request = GlobalTypeConverter.get().convert(value, to_type=ChatRequest)
            except Exception as e:
                logger.warning(f"Failed to convert input to ChatRequest: {e}, treating as string")
                # If conversion fails, pass through directly
                result = await call_next(value)
                return result

            # Extract current user message (for later saving)
            current_user_message = None
            if chat_request.messages:
                # Get the last user message
                for msg in reversed(chat_request.messages):
                    if msg.role == "user":
                        current_user_message = msg.content
                        break

            # ===== 1. Preprocess: Retrieve historical memory and inject =====
            if config.auto_retrieve:
                try:
                    logger.info(f"[Memory Middleware] Retrieving memory for user: {user_id}")
                    from nat.memory.models import SearchMemoryInput
                    import datetime

                    # Use generic query term instead of current question to avoid semantic bias
                    search_input = SearchMemoryInput(
                        query="conversation history",  # Generic query
                        top_k=config.top_k * 3,  # Get more candidates, filter by time later
                        user_id=user_id
                    )

                    memories = await memory_client.search(
                        query=search_input.query,
                        top_k=search_input.top_k,
                        user_id=search_input.user_id
                    )

                    if memories:
                        logger.info(f"[Memory Middleware] Found {len(memories)} candidate memories")

                        # ===== Key: Sort by time, take the most recent top_k =====
                        # Filter memories with timestamps
                        memories_with_time = []
                        for mem in memories:
                            timestamp_str = mem.metadata.get("key_value_pairs", {}).get("timestamp", "")
                            if timestamp_str:
                                try:
                                    timestamp = datetime.datetime.fromisoformat(timestamp_str)
                                    memories_with_time.append((timestamp, mem))
                                except ValueError as e:
                                    logging.warning(f"Invalid timestamp format: {timestamp_str}, error: {e}")

                        # Sort by time (oldest to newest)
                        memories_with_time.sort(key=lambda x: x[0])

                        # Only take the most recent top_k
                        recent_memories = [mem for _, mem in memories_with_time[-config.top_k:]]

                        logger.info(f"[Memory Middleware] Selected {len(recent_memories)} most recent memories")

                        # Inject memories into messages (in chronological order: old → new)
                        memory_messages = []
                        for mem in recent_memories:
                            if hasattr(mem, 'conversation') and mem.conversation:
                                for conv_msg in mem.conversation:
                                    memory_messages.append(
                                        Message(
                                            role=conv_msg.get('role', 'user'),
                                            content=conv_msg.get('content', '')
                                        )
                                    )

                        if memory_messages:
                            # Insert before current messages (no system message added, keep it simple)
                            chat_request.messages = memory_messages + chat_request.messages
                            logger.info(f"[Memory Middleware] Injected {len(memory_messages)} historical messages in chronological order")
                    else:
                        logger.info(f"[Memory Middleware] No previous memories found")

                except Exception as e:
                    logger.error(f"[Memory Middleware] Failed to retrieve memory: {e}", exc_info=True)

            result = await call_next(GlobalTypeConverter.get().convert(chat_request, to_type=ChatRequestOrMessage))

            # ===== 3. Postprocess: Save conversation memory =====
            if config.auto_save and current_user_message:
                try:
                    logger.info(f"[Memory Middleware] Saving conversation for user: {user_id}")
                    from nat.memory.models import MemoryItem
                    import datetime

                    # ===== Key: Correctly extract AI response =====
                    ai_response = None
                    if isinstance(result, str):
                        ai_response = result
                    elif isinstance(result, ChatResponse):
                        if result.choices and len(result.choices) > 0:
                            ai_response = result.choices[0].message.content
                        else:
                            ai_response = str(result)
                    else:
                        ai_response = str(result)

                    if ai_response:
                        # Build conversation record with timestamp
                        current_time = datetime.datetime.now(datetime.timezone.utc).isoformat()

                        memory_item = MemoryItem(
                            conversation=[
                                {"role": "user", "content": current_user_message},
                                {"role": "assistant", "content": ai_response}
                            ],
                            user_id=user_id,
                            metadata={
                                "key_value_pairs": {
                                    "type": "conversation",
                                    "timestamp": current_time  # Add timestamp
                                }
                            },
                            memory=f"User asked: {current_user_message[:100]}"  # Simplified memory field
                        )

                        await memory_client.add_items([memory_item])
                        logger.info(f"[Memory Middleware] Successfully saved conversation with timestamp {current_time}")
                    else:
                        logger.warning(f"[Memory Middleware] No AI response to save")

                except Exception as e:
                    logger.error(f"[Memory Middleware] Failed to save memory: {e}", exc_info=True)

            # ===== 4. Continue: Return the result =====
            return result

    yield ConversationMemoryMiddleware()
