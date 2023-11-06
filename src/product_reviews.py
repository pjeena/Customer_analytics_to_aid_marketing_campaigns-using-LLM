import requests, json, random
import pandas as pd
from parsel import Selector


def get_reviews_results(url, headers):
    data = []

    while True:
        html = requests.get(url, headers=headers)
        selector = Selector(html.text)

        for review in selector.css(".fade-in-animate"):
            title = review.css(".P3O8Ne::text").get()
            date = review.css(".ff3bE::text").get()
            rating = int(review.css(".UzThIf::attr(aria-label)").get()[0])
            content = review.css(".g1lvWe div::text").get()
            source = review.css(".sPPcBf").xpath("normalize-space()").get()

            data.append(
                {
                    "title": title,
                    "date": date,
                    "rating": rating,
                    "content": content,
                    "source": source,
                }
            )

        next_page_selector = selector.css(
            ".sh-fp__pagination-button::attr(data-url)"
        ).get()

        if next_page_selector:
            # re-assigns requests.get url to a new page url
            url = (
                "https://www.google.com"
                + selector.css(".sh-fp__pagination-button::attr(data-url)").get()
            )
        else:
            break

    return data


def main(URL_local):
    # https://docs.python-requests.org/en/master/user/quickstart/#custom-headers
    user_agent_list = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.1.1 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:77.0) Gecko/20100101 Firefox/77.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    ]

    user_agent = random.choice(user_agent_list)

    headers = {"User-Agent": user_agent}

    product_id = URL_local.split("/")[-1].split("?")[0]

    URL = "https://www.google.com/shopping/product/{}/reviews?hl=en&gl=fr".format(
        product_id
    )

    reviews_results = get_reviews_results(URL, headers)
    reviews_results = pd.DataFrame(reviews_results)

    reviews_results.to_parquet("data/product_reviews.parquet")

    return reviews_results


if __name__ == "__main__":
    reviews = main(
        URL_local="https://www.google.com/shopping/product/r/FR/9544491369209838230?prds=oid:9794333570643571113,eto:9794333570643571113_0,epd:9794333570643571113,prmr:1,rsk:PC_2677386548425780691&rss=ChZQQ18yNjc3Mzg2NTQ4NDI1NzgwNjkx&hl=en&sa=X&ved=2ahUKEwjo36fVlaWCAxVrLwYAHX4vC_IQkLoIegQIAhAU"
    )
    print(reviews.shape)
