
from dotenv import load_dotenv

load_dotenv()

from app.services.answer_generator import AnswerGenerator


def main():
    generator = AnswerGenerator()

    question = "Can I return a product if it stops working?"

    # Sample evidence for testing the answer-generation component.
    chunks = [
        {
            "source": "sample_policy.pdf",
            "page_number": 2,
            "content": (
                "Products with manufacturing defects may be eligible "
                "for warranty service subject to the stated conditions."
            ),
        }
    ]

    answer = generator.generate_answer(question, chunks)

    print("\nCustomer question:")
    print(question)

    print("\nSupportIQ answer:")
    print(answer)


if __name__ == "__main__":
    main()
