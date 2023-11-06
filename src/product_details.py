import requests, json, random
from parsel import Selector
import pandas as pd


def get_product_page_results(url, params, headers):
    html = requests.get(url, params=params, headers=headers)
    selector = Selector(html.text)

    title = selector.css(".sh-t__title::text").get()
    prices = [price.css("::text").get() for price in selector.css(".MLYgAb .g9WBQb")]
    low_price = selector.css(".KaGvqb .qYlANb::text").get()
    high_price = selector.css(".xyYTQb .qYlANb::text").get()
    shown_price = selector.css(".FYiaub").xpath("normalize-space()").get()
    reviews = int(
        selector.css(".YVQvvd .HiT7Id span::text").get()[1:-1].replace(",", "")
    )
    rating = float(selector.css(".uYNZm::text").get())
    extensions = [
        extension.css("::text").get() for extension in selector.css(".OA4wid")
    ]
    description = selector.css(".sh-ds__trunc-txt::text").get()
    media = [image.css("::attr(src)").get() for image in selector.css(".sh-div__image")]
    highlights = [
        highlight.css("::text").get() for highlight in selector.css(".KgL16d span")
    ]

    data = {
        "title": title,
        "prices": prices,
        "typical_prices": {
            "low": low_price,
            "high": high_price,
            "shown_price": shown_price,
        },
        "reviews": reviews,
        "rating": rating,
        "extensions": extensions,
        "description": description,
        "media": media,
        "highlights": highlights,
    }

    return data


def main(URL_local):
    # https://docs.python-requests.org/en/master/user/quickstart/#passing-parameters-in-urls
    product_id = URL_local.split("/")[-1].split("?")[0]

    params = {
        "product_id": product_id,  # product id
        "hl": "en",  # language
        "gl": "fr",  # country of the search, US -> USA
    }

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

    URL = f'https://www.google.com/shopping/product/{params["product_id"]}?hl={params["hl"]}&gl={params["gl"]}'

    product_page_results = get_product_page_results(URL, params, headers)

    with open("data/product_details.json", "w") as fp:
        json.dump(product_page_results, fp)  # encode dict into JSON

    print(json.dumps(product_page_results, indent=2, ensure_ascii=False))

    return product_page_results


if __name__ == "__main__":
    detials = main(
        URL_local="https://www.google.com/shopping/product/r/FR/9544491369209838230?prds=oid:9794333570643571113,eto:9794333570643571113_0,epd:9794333570643571113,prmr:1,rsk:PC_2677386548425780691&rss=ChZQQ18yNjc3Mzg2NTQ4NDI1NzgwNjkx&hl=en&sa=X&ved=2ahUKEwjqw7mdlaWCAxVNIgYAHYI7DnsQkLoIegQIAhAU"
    )
