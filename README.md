
# RevuSense

Revusense is an AI assitant platform that aims to help businesses gather and analyze customer reviews from Google Shopping and generate a comprehensive summary of the reviews filtered by an input query and rating with the help of SOTA LLM models. It provides tools to create advertisement copy based on the gathered insights, allowing businesses to scale their marketing efforts across different platforms. Additionally, it can also be levereaged to filter churned/repeat customers for personalized marketing emails. 



https://github.com/pjeena/Customer_analytics_to_aid_marketing_campaigns-using-LLM/assets/56345075/30b12057-c7b8-41ae-993e-2c7b47217fa6




![alt text](https://github.com/pjeena/Customer_insights_to_aid_marketing/blob/main/docs/app.jpeg)

## Project description



### 1. Data collection:

This component is responsible for scraping customer reviews from Google Shopping. Web scraping using beautiful soup and requests are used to extract product details and customet reviews


### 2. Generating Summaries and an Advertisement (Customer Modeling):

Once the reviews are collected, we create a vector store using pinecone python client to store the vector embeddings(hugging face) generated from reviews. After that, LangChain is used to generate a summary that highlights the most significant insights and sentiments expressed in the reviews. Here, gpt-4 LLM is used to generate summaries and Advertisement. This summary can be a valuable resource for understanding customer sentiment  based on a query related to the product and to improve marketing strategies and reach a broader audience. Here, gpt-4 model is used to generate summaries and Advertisement. One can tune the parameters to generate the output more relevant to the specific product.

### 3. Review Analytics:

An overall and comprehensive overview can be obtained with a simple click. It will also include some suggestons based on reviews to improve the respective product. This can be very useful to rectify product shortcomings and focus on improving it. 


### 4. Churn/Repeat customers:

The AI assitant can also be used to identify churn/repeat customers by just giving natural language instructions. This can be a game changer in building a POC by focusing on the qualitative unstructured data that is easily available to improve customer retention in a business.




## Setup

1. Clone this repository to your local machine:

`git clone https://github.com/yourusername/reviewscrape-adcopygen.git`

2. Install the required Python libraries using pip:

`pip install -r requirements.txt`

3. Get your API credentials for OPENAI API and run:

`streamlit run app.py`


