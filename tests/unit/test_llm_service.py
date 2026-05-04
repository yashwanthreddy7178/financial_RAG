import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.llm_service import LLMService

# =====================================================================
# TESTING ASYNC CODE
# Since LLMService uses 'async def' (asynchronous programming),
# our tests must also be 'async def', and we must use the @pytest.mark.asyncio decorator.
# This tells pytest to run this specific test inside an asyncio event loop.
# =====================================================================

@pytest.fixture
def llm_service():
    # We patch 'AsyncOpenAI' so that simply creating the LLMService doesn't 
    # actually connect to OpenAI or require a real API key.
    with patch('services.llm_service.AsyncOpenAI'):
        service = LLMService()
        return service

@pytest.mark.asyncio
async def test_route_query(llm_service):
    # This tests the 'route_query' function which decides if a user's question
    # is about finance (rag), small talk (small_talk), or totally random (off_topic).

    # Arrange: 
    # Since 'route_query' uses _with_backoff to call self.client.chat.completions.create,
    # we need to fake the response of that call.
    
    # We create a fake response object that behaves like the OpenAI response.
    fake_message = MagicMock()
    fake_message.content = "rag"
    
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    
    # Now we attach our fake response to the mock client.
    llm_service.client = MagicMock()
    # The 'create' function needs to be an AsyncMock because it is awaited (via the lambda in _with_backoff).
    llm_service.client.chat.completions.create = AsyncMock(return_value=fake_response)

    # Act: We await the real function, passing a financial question.
    # Because we patched 'create', it won't hit OpenAI. It will instantly return our fake_response.
    route = await llm_service.route_query("What was Apple's revenue in Q3?")

    # Assert: We check if the returned route matches what we set up in our fake response.
    assert route == "rag"
    
    # We can also verify that the fake 'create' function was actually called!
    # This proves that our code correctly tried to contact the LLM.
    llm_service.client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_small_talk(llm_service):
    # Arrange:
    # Let's fake the response structure again for small talk.
    fake_message = MagicMock()
    fake_message.content = "Hello! I am doing well, thank you."
    
    fake_choice = MagicMock()
    fake_choice.message = fake_message
    
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    
    llm_service.client = MagicMock()
    llm_service.client.chat.completions.create = AsyncMock(return_value=fake_response)

    # Act: 
    answer = await llm_service.generate_small_talk("How are you?")

    # Assert:
    assert answer == "Hello! I am doing well, thank you."
    llm_service.client.chat.completions.create.assert_called_once()
