# Chatbot-by-sai
<br>

## Take a moment to watch a demo video showcasing my project in action. 💡
<br>
 

https://github.com/SaiNivedh26/Chatbot-by-sai/assets/142657686/bd14b325-203f-41a9-94ef-51041c658758


 <br>

 # Note points 📝

* The Website which I used to Scrap the data and make this Chatbot is [brainlox](https://brainlox.com/courses/category/technical)
* I've used `UnstucturedURLLoader` from langchain in order to Extract the contens from the Website
* Then I've used `RecursiveChatacterTextSplitter` in Order to chunk the Data into small pieces
* After Chunking the data, I've embedded them using `Text-embedding-ada-002-v2` from OpenAI
* Once successfully embedded, Stored them in Chroma Vector Database
* In app.py, once the user gives request ( Handled by `Flask`) , using `RetrivalQA` ,we'll search across the database and will find similar reponses. After finding them, Used `GPT-3.5-turbo-instruct
` model from OpenAI in order to Reply to the Query with proper formatted response along with related Search Query
 
# How to Run 💻

1. **Clone the Repository:**
   ```
   git clone https://github.com/SaiNivedh26/Chatbot-by-sai.git
   ```

     
2. **Install the required Libraries:**
   ```
   pip install -r requirements.txt
   ```
3. **Enter openAI API key in .env file:**
   ```
   OPENAI_API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxx'
   ```
   
3. **First we need to run vector_db.py in order to create Embeddings and Store it in Chroma vector Database**
   ```
   python vector_db.py
   ```
   - After running this vector embeddings will be stored in the folder `chroma_db`

4. **Once the embedded are stored in vector space, go to `app.py` and Start running**
   ```
   python app.py
   ```
   - It uses flask to Handle requests and It'll run on Local server `http://127.0.0.1:5501/`
