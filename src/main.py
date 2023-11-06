import pandas as pd
import numpy as np
import random
from src import product_details, product_reviews
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
import pickle
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from faker import Faker
from dotenv import load_dotenv
import os 

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
OPENAI_API = os.getenv("OPENAI_API")

def get_product_details_and_reviews(URL):
    details = product_details.main(URL)
    reviews = product_reviews.main(URL)

    return details, reviews


def get_docs_and_metadata():
    df = pd.read_parquet("data/product_reviews.parquet")
    df["content"] = df["content"].astype(str)
    df = df[df["content"] != "Translate"].reset_index(drop=True)

    fake = Faker()
    customer_first_names = [fake.first_name() for i in range(len(df))]
    customer_last_names = [fake.last_name() for i in range(len(df))]
    customer_id = [fake.passport_number() for i in range(len(df))]
    customer_emails = [
        customer_first_names[i].lower()
        + customer_last_names[i].lower()
        + "{}".format(random.randint(1, 99))
        + "@gmail.com"
        for i in range(len(df))
    ]

    df["first_name"] = customer_first_names
    df["last_name"] = customer_last_names
    df["customer_email"] = customer_emails
    df["customer_ID"] = customer_id

    docs = list(df["content"])
    metadata = [
        dict(rating=i, customer_id=j) for i, j in zip(df["rating"], df["customer_ID"])
    ]

    df.to_parquet("data/product_reviews.parquet")

    return docs, metadata


def build_vector_database(docs, metadata):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_name = "reviews-similarity-search"
    index_existing = pinecone.list_indexes()

    if list(index_existing):
        pinecone.delete_index(index_existing[0])
        pinecone.create_index(name=index_name, dimension=768, metric="cosine")
    else:
        pinecone.create_index(name=index_name, dimension=768, metric="cosine")

    embeddings = HuggingFaceEmbeddings()
    vector_store = Pinecone.from_texts(
        docs, embeddings, index_name="reviews-similarity-search", metadatas=metadata
    )

    index = pinecone.Index("reviews-similarity-search")
    num_of_vectors = index.describe_index_stats()["total_vector_count"]

    while num_of_vectors < len(docs):
        num_of_vectors = index.describe_index_stats()["total_vector_count"]

    return vector_store


def retreive_relevant_docs(vector_store, query, num_reviews, rating_num):
    docs_semantic = vector_store.similarity_search(
        query, num_reviews, filter={"rating": rating_num}
    )
    pickle.dump(docs_semantic, open("data/retreived_docs.p", "wb"))

    return docs_semantic


def generate_summary_and_advertisement(docs):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=OPENAI_API)
    # Write summary of reviews

    PROMPT_TEMPLATE = """
    Write a concise summary of the reviews:

    {text}

    The summary should be about ten lines long
    """
    PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    summary = chain.run(docs)

    # Write ad for marketing
    PROMPT_TEMPLATE_AD = """
    Write the copy for a facebook or instagram ad based on the reviews:

    {text}

    As far as text goes, you can have up to 40 characters in your subject line, 
    125 characters in your primary text, and 30 characters in your outro
    You can also use appropriate emojis for headline and description. 
    """
    PROMPT = PromptTemplate(template=PROMPT_TEMPLATE_AD, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
    advertisement = chain.run(docs)

    return summary, advertisement


def generate_review_analytics(product):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, openai_api_key=OPENAI_API)
    # Write summary of reviews

    PROMPT_TEMPLATE_REVIEWS = """
    The reviews you see are for a product called '{}'.
    What is the overall impression of these reviews? Give most prevalent examples in bullets. 
    What do you suggest we focus on improving?
    """.format(
        product
    )

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    embeddings = HuggingFaceEmbeddings()
    index_name = "reviews-similarity-search"
    vector_db = Pinecone.from_existing_index(index_name, embeddings)
    review_chain = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=vector_db.as_retriever()
    )
    result = review_chain.run(PROMPT_TEMPLATE_REVIEWS)

    return result


def generate_churn_and_repeat_customers(query, rating):
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    index_name = "reviews-similarity-search"
    index = pinecone.Index(index_name=index_name)
    embeddings = HuggingFaceEmbeddings()
    #    query = "will purchase again"
    query_vector = embeddings.embed_query(query)
    results = index.query(
        query_vector,
        top_k=10,
        include_metadata=True,
        filter={"rating": {"$eq": rating}},
    )
    customer_ids_from_pinecone = [
        record["metadata"]["customer_id"] for record in results["matches"]
    ]
    df = pd.read_parquet("data/product_reviews.parquet")
    df_customer_churn_or_repeat = df[df["customer_ID"].isin(customer_ids_from_pinecone)][
        [
            "first_name",
            "last_name",
            "customer_email",
            "customer_ID",
            "rating",
            "content",
        ]
    ]
    return df_customer_churn_or_repeat


def main(query, num_reviews, rating_num):
#    details, reviews = get_product_details_and_reviews(
#        URL
#    )  # get prodcut details and reviews

    print("details and reviews fetched")

    docs, metadata = get_docs_and_metadata()  # convert to docs and metadata

    print("converted to docs and meta")

    vector_db = build_vector_database(docs, metadata)  # build vector databse

    print("setup vector db")

    docs_semantic = retreive_relevant_docs(
        vector_db,
        query,
        num_reviews,
        rating_num,  # retreive relevant docs based on query
    )

    print("retreived docs")
    print(docs_semantic)

    summary, advertisement = generate_summary_and_advertisement(docs_semantic)

    print("got summary and ad")

    return summary, advertisement


if __name__ == "__main__":
    import time

    start = time.time()

    URL = "https://www.google.com/shopping/product/13965615368656373838/reviews?hl=en&q=boat+earbuds&prds=eto:7719801540466663352_0,pid:16030487901746578544,rsk:PC_9290574075569046898&sa=X&ved=0ahUKEwjm2Yj09K6CAxVlVqQEHaRmAyIQqSQIQQ"
    det,rev = get_product_details_and_reviews(URL)
#    query = "easy to read font and good battery life"
#    summary, advertisement = main(URL, query, num_reviews=20, rating_num=5)
    print(det)
    print(
        "-------------------------------------------------------------------------------------------------------"
    )
    print(rev)

    end = time.time()
    print(end - start)
