"""Unit tests for the Ollama client module."""

from unittest.mock import patch, MagicMock
from typing import Iterator
import pytest
from pydantic import BaseModel

from text2graph.modules import (
    OllamaClient,
    OllamaMessage,
    OllamaResponse,
    OllamaGenerateResponse,
    OllamaError,
    ChatResult,
    chat,
)


class SampleOutputModel(BaseModel):
    """Test pydantic model for output parsing."""
    name: str
    age: int


class TestOllamaMessage:
    """Tests for OllamaMessage model."""

    def test_valid_message(self) -> None:
        """Test creating a valid message."""
        msg = OllamaMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_with_system_role(self) -> None:
        """Test message with system role."""
        msg = OllamaMessage(role="system", content="You are a helpful assistant")
        assert msg.role == "system"

    def test_message_with_assistant_role(self) -> None:
        """Test message with assistant role."""
        msg = OllamaMessage(role="assistant", content="I can help you")
        assert msg.role == "assistant"


class TestOllamaResponse:
    """Tests for OllamaResponse model."""

    def test_valid_response(self) -> None:
        """Test creating a valid response."""
        msg = OllamaMessage(role="assistant", content="Hello!")
        response = OllamaResponse(
            model="llama3",
            created_at="2024-01-01T00:00:00.000Z",
            message=msg,
            done=True,
        )
        assert response.model == "llama3"
        assert response.done is True
        assert response.message.content == "Hello!"

    def test_response_not_done(self) -> None:
        """Test response with done=False."""
        msg = OllamaMessage(role="assistant", content="Hello")
        response = OllamaResponse(
            model="llama3",
            created_at="2024-01-01T00:00:00.000Z",
            message=msg,
            done=False,
        )
        assert response.done is False


class TestOllamaGenerateResponse:
    """Tests for OllamaGenerateResponse model."""

    def test_valid_response(self) -> None:
        """Test creating a valid generate response."""
        response = OllamaGenerateResponse(
            model="llama3",
            created_at="2024-01-01T00:00:00.000Z",
            response="Generated text",
            done=True,
        )
        assert response.model == "llama3"
        assert response.response == "Generated text"
        assert response.done is True

    def test_response_with_timing(self) -> None:
        """Test response with timing information."""
        response = OllamaGenerateResponse(
            model="llama3",
            created_at="2024-01-01T00:00:00.000Z",
            response="Generated text",
            done=True,
            total_duration=1000000000,
            load_duration=500000000,
            prompt_eval_count=10,
            eval_count=20,
            eval_duration=500000000,
        )
        assert response.total_duration == 1000000000
        assert response.prompt_eval_count == 10
        assert response.eval_count == 20


class TestOllamaClient:
    """Tests for OllamaClient class."""

    @patch("ollama.list")
    def test_list_models(self, mock_list: MagicMock) -> None:
        """Test listing models."""
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"name": "llama3"}
        mock_list.return_value = MagicMock(models=[mock_model])

        client = OllamaClient()
        models = client.list_models()

        assert len(models) == 1
        assert models[0]["name"] == "llama3"
        mock_list.assert_called_once()

    @patch("ollama.list")
    def test_list_models_error(self, mock_list: MagicMock) -> None:
        """Test list_models raises OllamaError on failure."""
        mock_list.side_effect = Exception("Connection failed")

        client = OllamaClient()

        with pytest.raises(OllamaError, match="Failed to list models"):
            client.list_models()

    @patch("ollama.chat")
    def test_chat(self, mock_chat: MagicMock) -> None:
        """Test chat method."""
        msg = OllamaMessage(role="assistant", content="Hello!")
        mock_chat.return_value = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.000Z",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
        }

        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        response = client.chat(messages=messages, model="llama3")

        assert isinstance(response, ChatResult)
        assert response.content == "Hello!"
        assert response.parsed_data is None
        mock_chat.assert_called_once()

    @patch("ollama.chat")
    def test_chat_with_output_model(self, mock_chat: MagicMock) -> None:
        """Test chat method with output_model for JSON parsing."""
        mock_chat.return_value = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.000Z",
            "message": {"role": "assistant", "content": '{"name": "John", "age": 30}'},
            "done": True,
        }

        client = OllamaClient()
        messages = [{"role": "user", "content": "Return JSON"}]
        response = client.chat(messages=messages, model="llama3", output_model=SampleOutputModel)

        assert isinstance(response, ChatResult)
        assert response.content == '{"name": "John", "age": 30}'
        assert isinstance(response.parsed_data, SampleOutputModel)
        assert response.parsed_data.name == "John"
        assert response.parsed_data.age == 30
        mock_chat.assert_called_once()

    @patch("ollama.chat")
    def test_chat_with_output_model_error(self, mock_chat: MagicMock) -> None:
        """Test chat method with output_model raises error on invalid JSON."""
        mock_chat.return_value = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.000Z",
            "message": {"role": "assistant", "content": "Invalid JSON"},
            "done": True,
        }

        client = OllamaClient()
        messages = [{"role": "user", "content": "Return JSON"}]

        with pytest.raises(OllamaError, match="Failed to parse response as JSON"):
            client.chat(messages=messages, model="llama3", output_model=SampleOutputModel)

    @patch("ollama.chat")
    def test_chat_stream(self, mock_chat: MagicMock) -> None:
        """Test chat method with streaming."""
        mock_chunk1 = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.000Z",
            "message": {"role": "assistant", "content": "Hello"},
            "done": False,
        }
        mock_chunk2 = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.001Z",
            "message": {"role": "assistant", "content": "!"},
            "done": True,
        }
        mock_chat.return_value = [mock_chunk1, mock_chunk2]

        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]
        response = client.chat(messages=messages, model="llama3", stream=True)

        assert isinstance(response, Iterator)
        chunks = list(response)
        assert len(chunks) == 2
        assert chunks[0].message.content == "Hello"
        assert chunks[1].message.content == "!"

    def test_chat_stream_with_output_model_error(self) -> None:
        """Test chat method raises ValueError for stream with output_model."""
        client = OllamaClient()
        messages = [{"role": "user", "content": "Hi"}]

        with pytest.raises(ValueError, match="Cannot use output_model with streaming"):
            client.chat(messages=messages, model="llama3", stream=True, output_model=SampleOutputModel)

    @patch("ollama.generate")
    def test_generate(self, mock_generate: MagicMock) -> None:
        """Test generate method."""
        mock_generate.return_value = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.000Z",
            "response": "Generated text",
            "done": True,
        }

        client = OllamaClient()
        response = client.generate(prompt="Write a test", model="llama3")

        assert isinstance(response, OllamaGenerateResponse)
        assert response.response == "Generated text"
        mock_generate.assert_called_once()

    @patch("ollama.generate")
    def test_generate_with_options(self, mock_generate: MagicMock) -> None:
        """Test generate method with temperature and other options."""
        mock_generate.return_value = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.000Z",
            "response": "Generated text",
            "done": True,
        }

        client = OllamaClient()
        response = client.generate(
            prompt="Write a test",
            model="llama3",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
        )

        assert isinstance(response, OllamaGenerateResponse)
        call_kwargs = mock_generate.call_args.kwargs
        assert call_kwargs["options"]["temperature"] == 0.7
        assert call_kwargs["options"]["top_p"] == 0.9
        assert call_kwargs["options"]["top_k"] == 40

    @patch("ollama.generate")
    def test_generate_stream(self, mock_generate: MagicMock) -> None:
        """Test generate method with streaming."""
        mock_chunk1 = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.000Z",
            "response": "Gen",
            "done": False,
        }
        mock_chunk2 = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.001Z",
            "response": "erated",
            "done": False,
        }
        mock_chunk3 = {
            "model": "llama3",
            "created_at": "2024-01-01T00:00:00.002Z",
            "response": " text",
            "done": True,
        }
        mock_generate.return_value = [mock_chunk1, mock_chunk2, mock_chunk3]

        client = OllamaClient()
        response = client.generate(prompt="Write a test", model="llama3", stream=True)

        assert isinstance(response, Iterator)
        chunks = list(response)
        assert len(chunks) == 3
        assert chunks[0].response == "Gen"
        assert chunks[1].response == "erated"
        assert chunks[2].response == " text"

    @patch("ollama.generate")
    def test_generate_error(self, mock_generate: MagicMock) -> None:
        """Test generate raises OllamaError on failure."""
        mock_generate.side_effect = Exception("API error")

        client = OllamaClient()

        with pytest.raises(OllamaError, match="Generate request failed"):
            client.generate(prompt="Write a test", model="llama3")

    @patch("ollama.create")
    def test_create_model(self, mock_create: MagicMock) -> None:
        """Test create_model method."""
        mock_create.return_value = {"status": "success"}

        client = OllamaClient()
        response = client.create_model(name="test-model", modelfile="FROM llama3")

        assert response == {"status": "success"}
        mock_create.assert_called_once_with(name="test-model", modelfile="FROM llama3", stream=False)

    @patch("ollama.create")
    def test_create_model_stream(self, mock_create: MagicMock) -> None:
        """Test create_model method with streaming."""
        mock_chunk1 = {"status": "creating"}
        mock_chunk2 = {"status": "success"}
        mock_create.return_value = [mock_chunk1, mock_chunk2]

        client = OllamaClient()
        response = client.create_model(name="test-model", modelfile="FROM llama3", stream=True)

        chunks: list[dict] = list(response)  # type: ignore
        assert len(chunks) == 2
        assert isinstance(chunks[0], dict)
        assert chunks[0]["status"] == "creating"
        assert chunks[1]["status"] == "success"

    @patch("ollama.create")
    def test_create_model_error(self, mock_create: MagicMock) -> None:
        """Test create_model raises OllamaError on failure."""
        mock_create.side_effect = Exception("API error")

        client = OllamaClient()

        with pytest.raises(OllamaError, match="Create model request failed"):
            client.create_model(name="test-model", modelfile="FROM llama3")

    @patch("ollama.delete")
    def test_delete_model(self, mock_delete: MagicMock) -> None:
        """Test delete_model method."""
        mock_delete.return_value = None

        client = OllamaClient()
        result = client.delete_model(name="test-model")

        assert result is True
        mock_delete.assert_called_once_with("test-model")

    @patch("ollama.delete")
    def test_delete_model_error(self, mock_delete: MagicMock) -> None:
        """Test delete_model raises OllamaError on failure."""
        mock_delete.side_effect = Exception("API error")

        client = OllamaClient()

        with pytest.raises(OllamaError, match="Delete model request failed"):
            client.delete_model(name="test-model")

    def test_context_manager(self) -> None:
        """Test OllamaClient as context manager."""
        with OllamaClient() as client:
            assert isinstance(client, OllamaClient)

    def test_close(self) -> None:
        """Test close method."""
        client = OllamaClient()
        # Should not raise
        client.close()


class TestChatFunction:
    """Tests for the chat convenience function."""

    @patch("text2graph.modules.ollama_client.OllamaClient")
    def test_chat_function(self, mock_client_class: MagicMock) -> None:
        """Test the chat function."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = ChatResult(
            content="Hello!",
            parsed_data=None
        )
        mock_client.chat.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        result = chat(messages=messages, model="llama3")

        assert isinstance(result, ChatResult)
        assert result.content == "Hello!"
        mock_client.chat.assert_called_once()

    @patch("text2graph.modules.ollama_client.OllamaClient")
    def test_chat_function_with_host(self, mock_client_class: MagicMock) -> None:
        """Test the chat function with custom host."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = ChatResult(
            content="Hello!",
            parsed_data=None
        )
        mock_client.chat.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        result = chat(messages=messages, host="http://localhost:11435")

        assert isinstance(result, ChatResult)
        assert result.content == "Hello!"
        mock_client_class.assert_called_once_with(host="http://localhost:11435")
