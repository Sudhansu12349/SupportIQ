from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

try:
    response = client.responses.create(
        model="gpt-5.5",
        input="Reply with exactly: SupportIQ API connection successful."
    )
    print(response.output_text)
except Exception as exc:
    # Show the error without printing request headers or secrets.
    print(f"API request failed: {type(exc).__name__}: {exc}")
