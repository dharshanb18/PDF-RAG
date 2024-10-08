

import openai

class ResponseAgent:
    def __init__(self):
        openai.api_key = "your key"

    def rerank_results(self, results, query):
        reranked = []
        for result in results:
            prompt = f"Rate the relevance of the following context to the question: '{query}'\n\nContext: {result}\n\nRate on a scale of 1 to 10."
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            score = int(response.choices[0].message['content'].strip())
            reranked.append((result, score))

        # Sort by the score
        reranked = sorted(reranked, key=lambda x: x[1], reverse=True)
        return reranked

    def get_response(self,context, question):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

        response = openai.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
