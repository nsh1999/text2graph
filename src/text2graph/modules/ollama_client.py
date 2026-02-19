"""Ollama API wrapper module with Pydantic models."""

from typing import Optional, List, Dict, Any, Iterator, cast, Type, Union, TypeVar
import json

import ollama
from pydantic import BaseModel, Field, ValidationError

T = TypeVar('T', bound=BaseModel)


class OllamaMessage(BaseModel):
    """A message in the chat conversation."""
    role: str = Field(description="Role of the message sender (system, user, assistant)")
    content: str = Field(description="Content of the message")

    def to_json(self) -> Dict[str, str]:
        """
        Return the message as a JSON structure (dictionary).

        Returns:
            Dictionary representation of the message
        """
        return self.model_dump()


class OllamaResponse(BaseModel):
    """Response from Ollama model."""
    model: str = Field(description="Model name")
    created_at: str = Field(description="Creation timestamp")
    message: OllamaMessage = Field(description="Assistant's response message")
    done: bool = Field(description="Whether the generation is complete")


class OllamaGenerateResponse(BaseModel):
    """Response from text generation."""
    model: str = Field(description="Model name")
    created_at: str = Field(description="Creation timestamp")
    response: str = Field(description="Generated text")
    done: bool = Field(description="Whether the generation is complete")
    context: Optional[List[int]] = Field(default=None, description="Context tokens")
    total_duration: Optional[int] = Field(default=None, description="Total duration in nanoseconds")
    load_duration: Optional[int] = Field(default=None, description="Load duration in nanoseconds")
    prompt_eval_count: Optional[int] = Field(default=None, description="Number of prompt tokens evaluated")
    eval_count: Optional[int] = Field(default=None, description="Number of tokens evaluated")
    eval_duration: Optional[int] = Field(default=None, description="Evaluation duration in nanoseconds")


class OllamaError(Exception):
    """Custom exception for Ollama API errors."""
    pass


class ChatResult(BaseModel):
    """Result from chat method."""
    content: str
    parsed_data: Optional[BaseModel] = None


class OllamaClient:
    """Client for interacting with Ollama API."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        """
        Initialize the Ollama client.

        Args:
            host: Host URL of the Ollama server
            timeout: Request timeout in seconds
        """
        self.host = host.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.

        Returns:
            List of model information dictionaries
        """
        try:
            models = ollama.list()
            return [model.model_dump() for model in models.models]
        except Exception as e:
            raise OllamaError(f"Failed to list models: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama3",
        stream: bool = False,
        output_model: Optional[Type[T]] = None,
    ) -> Union[ChatResult, Iterator[OllamaResponse]]:
        """
        Send a chat request to Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name to use
            stream: Whether to stream the response
            output_model: Optional pydantic model to parse JSON response into

        Returns:
            ChatResult with content and optional parsed data, or stream iterator

        Raises:
            OllamaError: If the request fails
            ValueError: If stream=True and output_model is provided
        """
        if stream and output_model is not None:
            raise ValueError("Cannot use output_model with streaming")
        
        try:
            msg_objects = [OllamaMessage(**msg) for msg in messages]
            if stream:
                response = ollama.chat(
                    model=model,
                    messages=[msg.model_dump() for msg in msg_objects],
                    stream=True,
                )
                def stream_responses():
                    for chunk in response:
                        chunk_dict = chunk if isinstance(chunk, dict) else chunk.model_dump()
                        yield OllamaResponse(**chunk_dict)
                return stream_responses()
            else:
                response = ollama.chat(
                    model=model,
                    messages=[msg.model_dump() for msg in msg_objects],
                )
                response_dict = response if isinstance(response, dict) else response.model_dump()
                ollama_response = OllamaResponse(**response_dict)
                
                content = ollama_response.message.content
                
                if output_model is None:
                    try:
                        json_data = json.loads(content)
                        parsed_data = BaseModel(**json_data)
                    except (json.JSONDecodeError, ValidationError):
                        parsed_data = None
                else:
                    try:
                        json_data = json.loads(content)
                        parsed_data = output_model(**json_data)
                    except (json.JSONDecodeError, ValidationError) as e:
                        raise OllamaError(f"Failed to parse response as JSON for {output_model.__name__}: {e}")
                
                return ChatResult(content=content, parsed_data=parsed_data)
        except Exception as e:
            if isinstance(e, OllamaError):
                raise
            raise OllamaError(f"Chat request failed: {e}")

    def generate(
        self,
        prompt: str,
        model: str = "llama3",
        system: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> OllamaGenerateResponse | Iterator[OllamaGenerateResponse]:
        """
        Generate text with Ollama.

        Args:
            prompt: Input prompt
            model: Model name to use
            system: Optional system message
            stream: Whether to stream the response
            temperature: Temperature for generation (0-2)
            top_p: Top P sampling (0-1)
            top_k: Top K sampling

        Returns:
            Single response or stream iterator

        Raises:
            OllamaError: If the request fails
        """
        try:
            # Build kwargs to conditionally include optional parameters
            generate_kwargs = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
            }
            if system is not None:
                generate_kwargs["system"] = system
            if any([temperature, top_p, top_k]):
                generate_kwargs["options"] = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            
            response = ollama.generate(**generate_kwargs)
            
            if stream:
                return (OllamaGenerateResponse(**chunk) for chunk in response)
            else:
                return OllamaGenerateResponse(**response)
        except Exception as e:
            raise OllamaError(f"Generate request failed: {e}")

    def create_model(
        self,
        name: str,
        modelfile: str,
        stream: bool = False,
    ) -> Dict[str, Any] | Iterator[Dict[str, Any]]:
        """
        Create a new model from a modelfile.

        Args:
            name: Name for the new model
            modelfile: Modelfile content defining the model
            stream: Whether to stream the response

        Returns:
            Creation status dictionary or stream iterator

        Raises:
            OllamaError: If the request fails
        """
        try:
            response = ollama.create(name=name, modelfile=modelfile, stream=stream)
            if stream:
                return (chunk if isinstance(chunk, dict) else chunk.model_dump() for chunk in response)
            else:
                return response if isinstance(response, dict) else response.model_dump()
        except Exception as e:
            raise OllamaError(f"Create model request failed: {e}")

    def delete_model(self, name: str) -> bool:
        """
        Delete a model.

        Args:
            name: Model name to delete

        Returns:
            True if successful

        Raises:
            OllamaError: If the request fails
        """
        try:
            ollama.delete(name)
            return True
        except Exception as e:
            raise OllamaError(f"Delete model request failed: {e}")

    def close(self) -> None:
        """Close the session."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for quick use
def chat(
    messages: List[Dict[str, str]],
    model: str = "llama3",
    host: str = "http://localhost:11434",
    output_model: Optional[Type[T]] = None,
) -> ChatResult:
    """
    Quick chat function that returns a ChatResult.

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model name to use
        host: Ollama server URL
        output_model: Optional pydantic model to parse JSON response into

    Returns:
        ChatResult with content and optional parsed data
    """
    ollama_client = OllamaClient(host=host)
    try:
        response = ollama_client.chat(messages=messages, model=model, stream=False, output_model=output_model)
        return response
    finally:
        ollama_client.close()
