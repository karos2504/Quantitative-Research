import requests
from bs4 import BeautifulSoup
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

def get_article_urls(base_url, pages):
    """Scrape article URLs from the listing pages."""
    urls = []
    for page_num in range(1, pages + 1):
        page_url = f"{base_url}/Energy/Crude-Oil/Page-{page_num}.html"
        response = requests.get(page_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all article links on the page
        for article_div in soup.find_all('div', class_='categoryArticle'):
            for link in article_div.find_all('a', href=True):
                full_url = link['href']
                if full_url not in urls:
                    urls.append(full_url)
    return urls

def scrape_article_data(url):
    """Scrape headline, date, and content from an article page."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Headline extraction
    headline_tag = soup.find('h1')
    headline = headline_tag.text.strip() if headline_tag else url.split('/')[-1].replace('-', ' ')

    # Publication date extraction
    date_span = soup.find('span', class_='article_byline')
    date = date_span.text.split('-')[-1].strip() if date_span else 'Unknown'

    # Article content extraction
    article_body = soup.find('div', class_='article-content')
    if article_body:
        paragraphs = article_body.find_all('p')
        content = ' '.join(p.text for p in paragraphs)
    else:
        content = ' '.join(p.text for p in soup.find_all('p'))

    return headline, date, content

def main():
    BASE_URL = "https://oilprice.com"
    NUM_PAGES = 1

    # Step 1: Get article URLs
    article_urls = get_article_urls(BASE_URL, NUM_PAGES)

    # Step 2: Scrape article data
    data = [scrape_article_data(url) for url in article_urls]

    # Step 3: Create DataFrame
    news_df = pd.DataFrame(data, columns=['Headline', 'Date', 'News'])

    # Step 4: Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    news_df['Sentiment'] = news_df['News'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

    print(news_df.head())

if __name__ == "__main__":
    main()
