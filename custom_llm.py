import requests
import time
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import dotenv
dotenv.load_dotenv()

API_ENDPOINT = dotenv.get_key(dotenv.find_dotenv(), "CUSTOM_LLM_ENDPOINT")
class CustomLLM(LLM):
    """
    A custom LangChain LLM wrapper for the Saptarshi Workers GPT endpoint.
    """
    
    # Pydantic fields (configuration)
    endpoint_url: str = API_ENDPOINT
    system_prompt: str = "You are a helpful assistant."
    
    @property
    def _llm_type(self) -> str:
        return "custom_worker_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Execute the API call.
        Args:
            prompt: The user input text.
            stop: A list of strings to stop generation when encountered. 
                  (Note: Must be handled client-side if API doesn't support it)
        """
        
        # Prepare the JSON payload exactly as your API expects
        payload = {
            "input": prompt,
            "systemPrompt": self.system_prompt
        }

        last_exception = None
        for _ in range(5):
            try:
                # Make the POST request
                response = requests.post(self.endpoint_url, json=payload)
                
                # Check for non-200 status codes (like 500)
                response.raise_for_status()
                
                # Return the text content
                return response.text
                
            except requests.exceptions.RequestException as e:
                last_exception = e
            
            time.sleep(2)
        
        if last_exception:
            raise last_exception
        raise RuntimeError("API request failed after max retries")

    @property
    def _identifying_params(self) -> dict:
        """Parameters used for tracing/logging."""
        return {
            "endpoint_url": self.endpoint_url, 
            "system_prompt": self.system_prompt
        }