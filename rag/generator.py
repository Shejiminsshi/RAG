from openai import OpenAI
from dotenv import load_dotenv
import os

from config.setting import LLM_MODEL

load_dotenv()


class ResponseGenerator:
    """
    Handles response generation using Groq API.
    Smart conversational + document-grounded RAG assistant.
    """

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

        self.model_name = LLM_MODEL

    def generate_response(
        self,
        query,
        context,
        query_type,
        chat_history=""
    ):

        # ---------------------------------------------------
        # Greeting / Thanks Handling
        # ---------------------------------------------------

        if query_type in ["greeting", "thanks"]:

            try:

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self.get_system_prompt(
                                query_type
                            )
                        },
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    temperature=0.5
                )

                return response.choices[0].message.content

            except Exception as e:

                return f"Groq API Error:\n{e}"

        # ---------------------------------------------------
        # No Relevant Context Found
        # ---------------------------------------------------

        if not context:

            return (
                "The uploaded document does not contain "
                "information about this."
            )

        # ---------------------------------------------------
        # Context Processing
        # ---------------------------------------------------

        context_text = "\n\n".join(context)

        system_prompt = self.get_system_prompt(
            query_type
        )

        # ---------------------------------------------------
        # Main Prompt
        # ---------------------------------------------------

        final_prompt = f"""
You are an intelligent document analysis assistant.

IMPORTANT INSTRUCTIONS:
- Answer ONLY using the provided DOCUMENT CONTEXT.
- Do NOT use outside knowledge.
- If the answer is not found in the context, say:
  "The uploaded document does not contain information about this."
- Provide detailed and well-structured explanations when context is available.
- Support follow-up questions using previous conversation context.

CHAT HISTORY:
{chat_history}

DOCUMENT CONTEXT:
{context_text}

QUESTION:
{query}

Provide a detailed answer based on the document.
"""

        # ---------------------------------------------------
        # Generate Response
        # ---------------------------------------------------

        try:

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ],
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:

            return f"Groq API Error:\n{e}"

    # ---------------------------------------------------
    # System Prompts
    # ---------------------------------------------------

    def get_system_prompt(self, query_type):

        prompts = {

            "factual": """
You are an intelligent research assistant.

Provide accurate factual answers using document context only.
""",

            "analytical": """
You are an analytical document assistant.

Provide detailed analysis using only document context.
""",

            "opinion": """
Provide balanced viewpoints only if supported by the document.
""",

            "contextual": """
Use document context and previous conversation history.
""",

            "summary": """
You are a document summarization assistant.

Provide:
- concise document overview
- main topics covered
- important concepts
- structured summary

Use only document context.
""",

            "greeting": """
You are a friendly AI assistant.

Respond naturally and politely to greetings.
Keep responses short and friendly.
""",

            "thanks": """
Respond politely and naturally to appreciation messages.
"""
        }

        return prompts.get(
            query_type,
            prompts["contextual"]
        )