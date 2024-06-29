from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import logging
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

load_dotenv()
logging.basicConfig(level=logging.INFO)

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

app = Flask(__name__)
api = Api(app)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

DEFAULT_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an AI assistant specialized in providing information about programming courses and career advice for developers. When answering questions:

1. Always start by directly addressing the question asked.
2. If the question is about course recommendations, consider the user's experience level and specific interests.
3. Provide concise, structured answers with short and brief sentences.Don't use bulletins
4. If you don't have specific information about a particular course or topic, offer general advice or suggestions for further research.
5.Don't elaborate Longly
6. if user asks about specific description course, Try to provide all points concisely like price , duration and other descriptions clearly.
7. you tone must be friendly and professional

Context: {context}

Question: {question}

Answer:"""
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": DEFAULT_PROMPT
    }
)

class ChatbotAPI(Resource):
    def __init__(self):
        self.greetings = ['hi', 'hello', 'hey', 'greetings', 'howdy','wassup']
        self.farewells = ['bye', 'goodbye', 'see you', 'farewell','thanks','thank you']

    def is_greeting(self, text):
        return any(text.lower().startswith(word) for word in self.greetings)

    def is_farewell(self, text):
        return any(word in text.lower().split() for word in self.farewells)

    def post(self):
        try:
            data = request.get_json()
            question = data.get('question', '').strip()
            
            if not question:
                return jsonify({"error": "No question provided"}), 400

            if self.is_greeting(question):
                return jsonify({"answer": "Hello! How can I assist you today? Feel free to ask about programming courses or career advice for developers."})

            if self.is_farewell(question):
                return jsonify({"answer": "Goodbye! If you have any more questions about programming courses or development careers in the future, don't hesitate to ask."})

            context = "You are an AI assistant specialized in providing information about programming courses and career advice for developers."
            prompt = DEFAULT_PROMPT.format(context=context, question=question)
            logging.info(f"Generated prompt: {prompt}")
            
            result = qa_chain({"query": prompt})
            logging.info(f"QA Chain Result: {result}")
            
            if len(result['result']) < 20 or "I don't know" in result['result'].lower():
                return jsonify({
                    "answer": "I apologize, but I don't have enough specific information to answer that question confidently. However, I can offer some general advice or suggestions. Could you provide more details about what you're looking for, or ask a related question?"
                })

            return jsonify({"answer": result['result']})

        except Exception as e:
            logging.error(f"Error processing request: {str(e)}")
            return jsonify({"error": "An error occurred while processing your request. Please try again or rephrase your question."})
        
api.add_resource(ChatbotAPI, '/api/chat')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

