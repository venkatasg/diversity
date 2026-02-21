try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

from pydantic import BaseModel
import os

class GPT():
    '''
    Wrapper. Helps instantiate and make requests to openai clients.
    '''
    def __init__(self, gpt_model, key):
        if openai is None:
            raise ImportError(
                "The 'openai' package is required for QUDSim. "
                "Install it with: uv pip install openai"
            )
        self.model = gpt_model
        openai.api_key = key
        self.client = openai.OpenAI()

    def call(self, prompt: str, system_prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        msg = response.choices[0].message
        return msg.content


    def call_gpt_format(self, prompt: str, system_prompt: str, format):
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content":prompt}
                ],
                response_format=format,
            )

            answer = completion.choices[0].message.parsed
            return answer
        except:
            raise TypeError
        