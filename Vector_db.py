from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')


urls = [
    "https://brainlox.com/courses/category/technical",
    "https://brainlox.com/courses/872d1cb6-8469-4797-b267-8c41837b10e2",
    "https://brainlox.com/courses/4f629d96-5ed9-4302-ae0e-3479c543a49e",
    "https://brainlox.com/courses/2cf11f62-6452-41f1-9b42-303fb371b873",
    "https://brainlox.com/courses/be32e068-edca-4b41-96ee-4839de6aaebb",
    "https://brainlox.com/courses/0deafb39-3208-42db-93e3-bd69f8562f82",
    "https://brainlox.com/courses/fc9e2faf-dbe1-47bf-994c-f566a9ad3b42",
    "https://brainlox.com/courses/b0f2428a-c1c0-4def-8ac2-692a2d51a5b4",
    "https://brainlox.com/courses/e0edfcf8-9e0f-4c7a-bc90-3000822924e2",
    "https://brainlox.com/courses/fc29b015-962f-41fc-bc93-181d3ed87842",
    "https://brainlox.com/courses/fe8f5696-eb0e-48a0-a505-147e9c502b65",
    "https://brainlox.com/courses/9101883f-00af-48f7-949a-36cfc60ecfcf",
    "https://brainlox.com/courses/f9f7b907-5f4f-472d-a7e1-d44d38255a42",
    "https://brainlox.com/courses/cfcf2aa4-e220-4ce7-844e-563ebfaa98bd",
    "https://brainlox.com/courses/c5e8d5e8-58cb-4836-94e4-51314ffba4f3",
    "https://brainlox.com/courses/cd8e693c-c6a7-4dca-aeb1-4b7cf1dde495",
    "https://brainlox.com/courses/6a0fc4c9-2074-4854-ac31-c7dfad9ed932",
    "https://brainlox.com/courses/51890aee-163e-4ef7-86ff-fe0d6acba7e1",
    "https://brainlox.com/courses/c4bdeda0-9565-4073-9eeb-67f4a79e9ec7",
    "https://brainlox.com/courses/cc34d7ec-62ae-4018-b75e-207df98a4300",
    "https://brainlox.com/courses/0544dc35-181e-4e75-b26d-69393de68022",
    "https://brainlox.com/courses/8ca45c22-2dae-4c21-840c-8c9a177d09b3",
    "https://brainlox.com/courses/5d6b48ea-641e-4707-b7cf-c4f270789c9e",
    "https://brainlox.com/courses/7ef0ff36-cf58-4ce7-a010-b6bc7350d78a",
    "https://brainlox.com/courses/72423503-2e09-4404-92f5-48292dda42e3",
    "https://brainlox.com/courses/24e3bb16-bc23-4e6e-a74c-1ed32005cf0f",
    "https://brainlox.com/courses/31e70d66-efd5-44fc-929c-7de0e524624b",
    "https://brainlox.com/courses/89d301ea-ff81-4224-8d38-a35e8575bffd",
    "https://brainlox.com/courses/1af061fc-0890-40ac-98ae-01285283c5ff",
    "https://brainlox.com/courses/37b3b505-f8ca-4fd6-92e1-56da829f1805",
    "https://brainlox.com/courses/9829d760-3d63-456a-a1d3-25b96e554819",
    "https://brainlox.com/courses/af72f1d6-ee1b-48ba-a217-30985c593bb5",
    "https://brainlox.com/courses/189a9ef3-be9d-4b42-a09c-75665fa36e3e",
    "https://brainlox.com/courses/88ac0ed1-e388-41c9-a1b4-bd82beb52c10",
    "https://brainlox.com/courses/7ad05239-54cb-4cbb-8cec-4065512db97a",
    "https://brainlox.com/courses/7f20a234-094c-457a-b2b4-8712bd8f0616",
    "https://brainlox.com/courses/03549fee-5478-4352-b146-f2a2c8561191",
    "https://brainlox.com/courses/94c549e6-ebca-4bbe-b1a8-17beda62b6d5",
    "https://brainlox.com/courses/5ba67aa9-b272-42ec-88d2-81cd5af6c643",
    "https://brainlox.com/courses/848a2ee8-b9ce-464a-bcbc-78fcdd67b9dd",
    "https://brainlox.com/courses/4d6f6d90-a53f-4792-a491-1697bb384d57",
    "https://brainlox.com/courses/d6e5c1ea-f3bf-4435-a1c6-7566aef9ddc2",
    "https://brainlox.com/courses/3cdb1ba6-7b9d-4a04-8ba4-bfffbef9a77f",
    "https://brainlox.com/courses/6ffc967e-8caf-4540-b0f3-c1e684195d03",
    "https://brainlox.com/courses/8c6151e3-3141-4a64-8fe0-d5df205e750d",
    "https://brainlox.com/courses/db80cdd0-d396-482a-913e-1fc81c8d9ac4",
    "https://brainlox.com/courses/9e2df5e5-3dc5-4c87-b385-f464642d8562",
    "https://brainlox.com/courses/0884af64-fb1b-4c15-9850-3315b99fd92d",
    "https://brainlox.com/courses/4b7a6968-f8f6-4f98-bd8d-357b71ca9496",
    "https://brainlox.com/courses/9e0490f4-e45b-41df-858c-34a4eba5836b",
    "https://brainlox.com/courses/fd86f75c-b1b5-450d-98ee-7e3ad0b1e357",
    "https://brainlox.com/courses/c110330f-d41d-48fe-8433-88e7527c4d77",
    "https://brainlox.com/courses/1164fafc-7e5b-4126-b1d6-ce228e266345",
    "https://brainlox.com/courses/a06e18a8-3e06-4777-85c5-fb22b35ea46d",
    "https://brainlox.com/courses/50986fd0-ea11-4d2e-971f-5fb88fb81542",
    "https://brainlox.com/courses/5f754053-de57-4f46-9bb3-b11d5c9e42e0",
    "https://brainlox.com/courses/f6853a9e-84ce-40c6-9362-00fcf701e459",
    "https://brainlox.com/courses/931b56d8-4192-43e1-ba10-0ce6f94cb2ed",
    "https://brainlox.com/courses/1db857a4-f374-49fb-af9c-0f6d2e5ada45",
    "https://brainlox.com/courses/7f24d6b5-1645-4184-83d4-6deab75b13f1",
    "https://brainlox.com/courses/0bc0c9fa-9749-4129-91de-c286a474e9b3",
    "https://brainlox.com/courses/5abad7b0-953d-4527-8aa9-4d3ca80ff500",
    "https://brainlox.com/courses/7544e04d-82d5-447e-9935-c784455c2de3",
    "https://brainlox.com/courses/7472b97e-7b73-42dc-b0ef-972832e7bad5",
    "https://brainlox.com/courses/5a45c3b4-3f6e-48dc-ab92-71a1cd38db28",
    "https://brainlox.com/courses/ebe89ac6-5b3a-479d-adab-d6e2d5297773",
    "https://brainlox.com/courses/eab79907-4de4-4413-bb44-a92e253aa5c6",
    "https://brainlox.com/courses/2bf6f84c-f096-459a-a8fb-e629d263f109"

]


loader = UnstructuredURLLoader(urls=urls)

documents = loader.load()



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
vectorstore.persist()


print(f"Loaded {len(documents)} documents and created {len(splits)} splits.")
print(f"Vector store created and persisted in './chroma_db'")

# database will be created and words will be stored as embeddings
