# crawler.py

import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from bs4 import BeautifulSoup
from bs4.element import Tag
from urllib.parse import urljoin
from urllib import robotparser
import urllib.request
from seleanium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from webdriver_manager.firefox import GeckoDriverManager
import json

# --- Configuration ---the parliament approved tax reforms
COVENTRY_PUREPORTAL_URL = "https://pureportal.coventry.ac.uk/en/organisations/fbl-school-of-economics-finance-and-accounting/publications"
BASE_URL = "https://pureportal.coventry.ac.uk"
CRAWLER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
BASE_DELAY = 2

# --- Driver Setup (Modified for Firefox) ---
def setup_driver():
    """Sets up a headless Firefox browser instance using Selenium and webdriver_manager."""
    print("Setting up the headless Firefox browser...")
    firefox_options = FirefoxOptions()
    # To see the browser window, remove the '#' from the line below.
    # firefox_options.add_argument("--headless")
    firefox_options.add_argument("--disable-gpu")
    firefox_options.add_argument("--no-sandbox")
    firefox_options.add_argument(f"user-agent={CRAWLER_USER_AGENT}")

    try:
        service = FirefoxService(GeckoDriverManager().install())
        driver = webdriver.Firefox(service=service, options=firefox_options)
        print("Driver setup complete.")
        return driver
    except Exception as e:
        print(f"Error setting up WebDriver: {e}")
        return None

# --- New function to fetch authors ---
def fetch_authors(soup, base_url):
    authors_data = []
    persons_p = soup.select_one('p.relations.persons')
    if not persons_p:
        return []
    for element in persons_p.contents:
        if isinstance(element, Tag) and element.name == 'a':
            name = element.get_text(strip=True)
            url = urljoin(base_url, str(element.get('href', '')))
            if name:
                authors_data.append({'name': name, 'url': url})
        elif isinstance(element, str):
            potential_names = element.split(',')
            for name_part in potential_names:
                clean_name = name_part.strip(' ,')
                if clean_name:
                    authors_data.append({'name': clean_name, 'url': None})
    return authors_data

# --- Extract abstract content from publication url ---
def fetch_abstract(soup):
    # Find the specific div containing the abstract
    abstract_div = soup.find('div', class_='rendering_researchoutput_abstractportal')
    if abstract_div:
        # The text is within a nested 'textblock' div
        text_block = abstract_div.find('div', class_='textblock')
        if text_block:
            return text_block.get_text(strip=True)
    return '' # Return empty string if abstract is not found

# --- Scrape details from a single publication page ---
def scrape_publication_details(driver, url, title_from_list, robot_parser):
    if not robot_parser.can_fetch(CRAWLER_USER_AGENT, url):
        print(f"Skipping disallowed URL: {url}")
        return title_from_list, [], 'N/A', ''
    
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '.container')))
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Use the title scraped from the list page
        title = title_from_list
        authors, date, abstract = [], 'N/A', ''

        # Scrape authors
        try:
            authors = fetch_authors(soup, BASE_URL)
        except Exception as e:
            print(f"Could not find authors on {url}: {e}")

        # Scrape abstract
        abstract = fetch_abstract(soup)

        # Scrape date
        date_tag = soup.find('span', class_='date')
        date = date_tag.text.strip() if date_tag else 'N/A'
        
        return title, authors, date, abstract

    except (TimeoutException, WebDriverException) as e:
        print(f"Error loading page {url}: {e}")
        return title_from_list, [], 'N/A', ''

# --- Crawler Core ---
def crawl_pureportal(driver, start_url, robot_parser, crawl_delay):
    if not robot_parser.can_fetch(CRAWLER_USER_AGENT, start_url):
        print(f"Cannot access start URL according to robots.txt: {start_url}")
        return []
    
    print(f"Starting crawl from: {start_url}")
    publications_data = []
    try:
        driver.get(start_url)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.list-results'))
        )
        
        publication_details_list = []
        
        # Step 1: Collect publication links and titles
        while True:
            print("Collecting links from current page...")
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            
            publication_elements = soup.find_all('li', class_='list-result-item')
            if not publication_elements:
                print("No more publication elements found. Exiting.")
                break

            for pub_elem in publication_elements:
                if isinstance(pub_elem, Tag):
                    title_tag = pub_elem.find('a', class_='link')
                    if title_tag and isinstance(title_tag, Tag) and title_tag.get('href'):
                        pub_title = title_tag.get_text(strip=True)
                        pub_link = urljoin(BASE_URL, str(title_tag['href']))
                        publication_details_list.append({'title': pub_title, 'link': pub_link})

            
            try:
                next_page = soup.find('a', class_='nextLink')
                if isinstance(next_page, Tag) and 'href' in next_page.attrs:
                    next_page_link = urljoin(BASE_URL, str(next_page['href']))
                    
                    if not robot_parser.can_fetch(CRAWLER_USER_AGENT, next_page_link):
                        print(f"Next page URL disallowed by robots.txt: {next_page_link}")
                        break
                    
                    print("Moving to next page...")
                    print(f"Next url: {next_page_link}")
                    time.sleep(crawl_delay)
                    driver.get(next_page_link)
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, 'ul.list-results'))
                    )
                else:
                    print("No more pages to crawl.")
                    break
            except NoSuchElementException:
                print("No next page button found. Assuming end of results.")
                break
        
        print(f"Collected {len(publication_details_list)} publication links. Now scraping details...")
        
        # Step 2: Visit each publication link and scrape details
        for i, pub_info in enumerate(publication_details_list):
            link = pub_info['link']
            title = pub_info['title']
            print(f"Scraping details for publication {i+1}/{len(publication_details_list)}: {link}")
            
            title_final, authors, date, abstract = scrape_publication_details(driver, link, title, robot_parser)
            
            publications_data.append({
                'id': i,  # Assigning a simple ID
                'title': title_final,
                'authors': json.dumps(authors), # Serialize authors to a string for CSV
                'date': date,
                'abstract': abstract,
                'publication_link': link
            })
            
            time.sleep(crawl_delay)
        
        return publications_data

    except TimeoutException:
        print("Page load timed out. Some content might not have been retrieved.")
        return publications_data
    except WebDriverException as e:
        print(f"WebDriver error during crawl: {e}")
        return publications_data
    finally:
        if driver:
            driver.quit()

# --- Main Execution ---
if __name__ == "__main__":
    driver = setup_driver()
    if driver is None:
        exit()
    
    robot_parser = robotparser.RobotFileParser()
    robots_url = urljoin(BASE_URL, 'robots.txt')
    print(f"Fetching robots.txt from: {robots_url}")
    
    try:
        req = urllib.request.Request(robots_url, headers={'User-Agent': CRAWLER_USER_AGENT})
        with urllib.request.urlopen(req) as response:
            robots_content = response.read().decode('utf-8')
        
        print("\n--- Robots.txt content ---")
        print(robots_content)
        print("-------------------------\n")
        
        robot_parser.parse(robots_content.splitlines())
        print("robots.txt parsed successfully.")
    except Exception as e:
        print(f"Warning: Could not fetch or parse robots.txt. Proceeding with default settings. Error: {e}")
    
    robots_crawl_delay = robot_parser.crawl_delay(CRAWLER_USER_AGENT)
    robots_delay_value = int(robots_crawl_delay) if robots_crawl_delay else None
    
    if robots_delay_value and robots_delay_value > BASE_DELAY:
        final_delay = robots_delay_value
        print(f"Using Crawl-Delay from robots.txt: {final_delay} seconds.")
    else:
        final_delay = BASE_DELAY
        print(f"Using configured minimum delay: {final_delay} seconds.")
        
    publications_data = crawl_pureportal(driver, COVENTRY_PUREPORTAL_URL, robot_parser, final_delay)

    print("\n--- Crawling Complete. Total Publications Found: ---")
    print(len(publications_data))

    # --- Save to CSV ---
    df = pd.DataFrame(publications_data)
    df.to_csv("coventry_publications.csv", index=False)
    print("Publications saved to coventry_publications.csv")

    # --- Save to JSON ---
    with open("coventry_publications.json", "w", encoding="utf-8") as f:
        json.dump(publications_data, f, ensure_ascii=False, indent=4)
    print("Publications saved to coventry_publications.json")
    print("Publications saved to coventry_publications.json")
