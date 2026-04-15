# step 1 data ingestion and normalisation
from urllib import response

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader
from langchain_core.documents import Document
# imports for step 2
import spacy
from keybert import KeyBERT
# imports for step 3
from langchain_text_splitters import RecursiveCharacterTextSplitter
# imports for step 4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# imports for step 5
from langchain_core.prompts import PromptTemplate

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# imports fro step 6
# from langchain.retrievers import EnsembleRetriever 
from langchain_community.retrievers import BM25Retriever
# imports for step 9
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
def load_data(input_type,input_value):
    documents=[]
    if input_type=="pdf":
        loader=PyPDFLoader(input_value)
        docs=loader.load()
        for d in docs:
            d.metadata['type']='pdf'
            d.metadata['source']=input_value
            d.metadata['page']=d.metadata.get('page',None)
        documents.extend(docs)
    elif input_type=="web":
        loader=WebBaseLoader(input_value)
        docs=loader.load()
        for d in docs:
            d.metadata['type']='web'
            d.metadata['source']=input_value
        documents.extend(docs)
    elif input_type=="youtube":
        video_id=input_value
        try:
            api=YouTubeTranscriptApi()
            transcript_list=api.fetch(video_id)
            transcript=" ".join(chunk.text for chunk in transcript_list)
            doc=Document(
                page_content=transcript,
                metadata={
                    'type':'youtube',
                    'source':video_id
                }
            )
            documents.append(doc)
        except TranscriptsDisabled:
            print("Transcripts are disabled for this video.")
    else:
        print("Unsupported input type. Please choose from 'pdf', 'web', or 'youtube'.")
    return documents


# step 2 nlp layer( cleaning,sentence segmentation,ner,keyword extraction)
nlp=spacy.load("en_core_web_sm")
kw_model=KeyBERT()
def preprocess_documents(documents):
    preprocessed_docs=[]
    for doc in documents:
        text=doc.page_content
        # cleaning the data
        cleaned_text=text.replace("\n"," ").strip()
        # nlp pipeline
        cleaned_doc=nlp(cleaned_text)
        sentences=[]
        for sent in cleaned_doc.sents:
            sentences.append(sent.text)
        entities=[]
        for ent in cleaned_doc.ents:
            entities.append(ent.text)
        keywords=[]
        for kw in kw_model.extract_keywords(cleaned_text,top_n=10):
            keywords.append(kw[0])
        new_doc=Document(
            page_content=cleaned_text,
            metadata={
                **doc.metadata,
                'sentences':sentences,
                'entities':entities,
                'keywords':keywords
            }
        )
        preprocessed_docs.append(new_doc)
    return preprocessed_docs

# step 3 chunking the data
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n","\n","."," ",""]
)
def split_documents(documents):
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
# result1=load_data("pdf","dl-curriculum.pdf")
# print("load_data\n",result1)
# print("First Document Content:\n", result1[0].page_content[:500])
# print("\nMetadata:\n", result1[0].metadata)
# result2=preprocess_documents(result1)
# print("preprocess_documents\n",result2)
# result3=split_documents(result2)
# print("split_documents\n",result3)

# step 4 vectorization using FAISS

def vectorize_documents(documents):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(documents, embedding)

    return vector_store
def find_similarity(vector_store, query, k=5):
    results = vector_store.similarity_search(query, k=k)
    return results

# step 5 reqriting query using prompt template
# process query by extracting entities and keywords using spacy and keybert, then pass to prompt template to rewrite the query and generate alternatives
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model=ChatHuggingFace(llm=llm)
query_prompt = PromptTemplate(
    input_variables=["query", "entities", "keywords"],
    template="""
You are an expert AI system designed to optimize search queries for a hybrid retrieval system (semantic + keyword-based).

User Query:
{query}

Pre-extracted Information:
- Named Entities: {entities}
- Keywords: {keywords}

Tasks:
1. Rewrite the query clearly.
2. Generate 3 alternative queries.

Output Format:

Rewritten Query:
<query>

Alternative Queries:
1. <query>
2. <query>
3. <query>
"""
)
def extract_queries(response_text):
    lines = response_text.split("\n")

    rewritten = ""
    alternatives = []

    for line in lines:
        line = line.strip()

        if line.startswith("Rewritten Query:"):
            rewritten = line.replace("Rewritten Query:", "").strip()

        elif line.startswith(("1.", "2.", "3.")):
            alternatives.append(line.split(".", 1)[1].strip())

    return [rewritten] + alternatives
def process_query(query):
    # ✅ Step 1: NLP Processing
    doc = nlp(query)

    entities = [ent.text for ent in doc.ents]
    keywords = [kw[0] for kw in kw_model.extract_keywords(query, top_n=5)]
    
    # ✅ Step 2: Call LLM using PromptTemplate
    chain = query_prompt | model

    response = chain.invoke({
        "query": query,
        "entities": ", ".join(entities) if entities else "None",
        "keywords": ", ".join(keywords) if keywords else "None"
    })
    # ✅ FIX HERE
    response_text = response.content if hasattr(response, "content") else response

    queries = extract_queries(response_text)

    return {
        "original_query": query,
        "entities": entities,
        "keywords": keywords,
        "multi_queries": queries
    }
# step 6: retrive data using faiss retriver and bm25 retriver is used for keyword based retriveal
def retrive_documents(vector_store,query_data,documents,k=5):
    faiss_retriever=vector_store.as_retriever(search_type="mmr",
                                              search_kwargs={
            "k": k,
            "fetch_k": k * 3,
            "lambda_mult": 0.5
        })
    bm25_retriever=BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # 🔹 3. Hybrid Retriever (FAISS + BM25)
    # hybrid_retriever = EnsembleRetriever(
    #     retrievers=[faiss_retriever, bm25_retriever],
    #     weights=[0.5, 0.5]
    # )
    # 
    all_docs=[]
    for q in query_data["multi_queries"]:
        # 🔹 Get FAISS results
        faiss_docs = faiss_retriever.invoke(q)

        # 🔹 Get BM25 results
        bm25_docs = bm25_retriever.invoke(q)

        # 🔹 Combine manually
        all_docs.extend(faiss_docs)
        all_docs.extend(bm25_docs)

    # 🔹 5. Deduplication
    seen = set()
    unique_docs = []

    for doc in all_docs:
        content = doc.page_content
        if content not in seen:
            seen.add(content)
            unique_docs.append(doc)
    return unique_docs[:k]

# step 7:source aware context building :giving citations like page number is it from video or web page or pdf

def buid_context(docs):
    context=""
    for doc in docs:
        dtype=doc.metadata.get('type','unknown')
        source=doc.metadata.get('source','unknown')
        if dtype=='pdf':
            page=doc.metadata.get("page","N/A")
            header=f"[PDF:{source},Page:{page}]"
        elif dtype=="web":
            header=f"[WEB:{source}]"
        elif dtype=="youtube":
            timestamp=doc.metadata.get("timestamp","N/A")
            header=f"[YOUTUBE:{source},Timestamp:{timestamp}]"
        else:
            header="[UNKNOWN SOURCE]"
        context += header + "\n"
        context += doc.page_content + "\n\n"
    return context

answer_prompt = PromptTemplate(
    input_variables=["context", "query", "history"],
    template="""
You are an AI assistant that answers questions using retrieved documents.

Instructions:
- Use ONLY the provided context.
- Include source references (e.g., PDF name, page, or video ID) in your answer.
- Do NOT fabricate information.
- If the answer is not available, respond: "I don't know".

Conversation History:
{history}

Context:
{context}

Question:
{query}

Answer (with sources):
"""
)

# step 8 generation of answers
def generate_answer(query, context, history):
    chain = answer_prompt | model

    response = chain.invoke({
        "context": context,
        "query": query,
        "history": history
    })

    return response.content if hasattr(response, "content") else response
# step 9: creating conversational memory
chat_history = []
def update_memory(query, answer):
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))
def add_system_message():
    if not chat_history:
        chat_history.append(
            SystemMessage(content="You are a helpful AI assistant that answers only from provided context.")
        )

def get_history(limit=10):
    return chat_history[-limit:]
# rag pipeline
def rag_pipeline(query, vector_store, documents):
    """
    End-to-end RAG pipeline:
    - Query processing (Step 5)
    - Hybrid retrieval (Step 6)
    - Context building (Step 7)
    - Answer generation (Step 8)
    - Conversational memory (Step 9)
    """

    # 🔹 Step 5: Process query
    query_data = process_query(query)

    # 🔹 Step 6: Retrieve documents
    docs = retrive_documents(vector_store, query_data, documents)

    # 🔹 Step 7: Build context with metadata
    context = buid_context(docs)

    # 🔹 Step 9: Add system message + get history
    add_system_message()
    history_messages = get_history()

    # 🔹 Prepare messages for LLM
    messages = history_messages.copy()

    # Add context as system message
    messages.append(SystemMessage(content=f"Context:\n{context}"))

    # Add current user query
    messages.append(HumanMessage(content=query))

    # 🔹 Step 8: Generate answer (chat-based)
    response = model.invoke(messages)

    answer = response.content if hasattr(response, "content") else response

    # 🔹 Step 9: Update memory
    update_memory(query, answer)

    return {
        "answer": answer,
        "sources": docs,   # useful for frontend highlighting
        "context": context  # optional (for debugging)
    }
# testing
if __name__ == "__main__":

    # 🔹 Step 1–4: Setup (run once)
    print("Loading data...")

    docs = load_data("pdf", "dl-curriculum.pdf")   # 👈 give your PDF path
    processed_docs = preprocess_documents(docs)
    chunks = split_documents(processed_docs)

    vector_store = vectorize_documents(chunks)

    print("Setup complete ✅")

    # 🔹 Step 5–9: Test query
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        result = rag_pipeline(query, vector_store, chunks)

        print("\n💡 Answer:")
        print(result["answer"])

        print("\n📄 Sources:")
        for doc in result["sources"]:
            print(doc.metadata)