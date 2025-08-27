import schedule
import time
import logging
from datetime import datetime
from crawlerrr import COVENTRY_PUREPORTAL_URL, crawl_pureportal, setup_driver
from indexer import load_data, build_tfidf_index, save_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler()
    ]
)

def run_pipeline():
    """Run the crawler and indexer pipeline"""
    try:
        logging.info("Starting weekly data pipeline")
        
        # Run the crawler
        logging.info("Starting crawler...")
        driver = setup_driver()
        if driver is None:
            exit()
        
        crawl_pureportal(driver, COVENTRY_PUREPORTAL_URL)
        logging.info("Crawler completed successfully")
        
        # Run the indexer
        logging.info("Starting indexer...")
        publications = load_data("coventry_publications.csv")
        vectorizer, tfidf_matrix, publications_list = build_tfidf_index(publications)
        save_index(vectorizer, tfidf_matrix, publications_list)

        logging.info("Indexer completed successfully")
        
        logging.info("Pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Error in pipeline: {str(e)}", exc_info=True)

def main():
    # Schedule the job to run every Monday at 1:00 AM
    schedule.every().monday.at("01:00").do(run_pipeline)
    
    logging.info("Scheduler started - Will run every Monday at 01:00")
    
    # Run the job immediately on startup
    run_pipeline()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()