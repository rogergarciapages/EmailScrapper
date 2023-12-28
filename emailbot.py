import os
import re
import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup
import time
import logging
import traceback
import json
import sys
import pandas as pd

# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the path to the credentials file
credentials_path = os.path.join(script_dir, 'D:\\Python\\NewsletterMonstr\\credentials.json')

# Read credentials from JSON file
try:
    with open(credentials_path, 'r') as file:
        credentials = json.load(file)
    email_user = credentials.get('email_user')
    email_pass = credentials.get('email_pass')
except FileNotFoundError:
    print(f"ERROR:__main__:Credentials file '{credentials_path}' not found. Please create the file.")
    sys.exit(1)

# Configure the logging module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load credentials from JSON file
credentials_file = "credentials.json"

# Create an empty list to store data for the Pandas DataFrame
data_list = []

def connect_to_imap():
    """Connect to the IMAP server."""
    try:
        mail = imaplib.IMAP4_SSL("mail.newslettermonster.com", 993)
        mail.login(email_user, email_pass)
        return mail
    except imaplib.IMAP4.error as e:
        logger.error(f"Error connecting to IMAP server: {e}")
        return None

def extract_html_and_css(email_message):
    """Extract HTML and CSS content from an email message."""
    html_content = ""
    css_content = ""

    for part in email_message.walk():
        if part.get_content_type() == "text/html":
            html_payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or 'utf-8'
            html_content = html_payload.decode(charset, 'ignore')
        elif part.get_content_type() == "text/css":
            css_payload = part.get_payload(decode=True)
            charset = part.get_content_charset() or 'utf-8'
            css_content = css_payload.decode(charset, 'ignore')

    return html_content, css_content

def process_email(msg_id, email_message):
    """Process a single email."""
    try:
        logger.info(f"Processing email {msg_id}")
        # Fetch subject and HTML content
        subject, _ = decode_header(email_message["Subject"])[0]
        subject = subject.decode("utf-8", errors="replace") if isinstance(subject, bytes) else subject
        safe_subject = re.sub(r'[^\w\s.-]', '_', subject).strip()

        email_folder = os.path.join(output_folder, safe_subject)
        os.makedirs(email_folder, exist_ok=True)

        html, css = extract_html_and_css(email_message)

        # Check if HTML content is empty
        if not html:
            logger.warning(f"Empty HTML content in email {msg_id}. Skipping.")
            return

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception as parse_error:
            logger.error(f"Error parsing HTML for email {msg_id}: {parse_error}")
            traceback.print_exc()
            return

        filename = f"{email_folder}/{safe_subject}.html"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(soup.prettify())
        logger.info(f"Processed email {msg_id}: {safe_subject}")

        real_subject_filename = f"{email_folder}/monsterrealemaisubject.html"
        with open(real_subject_filename, "w", encoding="utf-8") as real_subject_file:
            real_subject_file.write(f"<html><head><title>Real Subject</title></head><body>{subject}</body></html>")

        sender_address = email.utils.parseaddr(email_message.get("From"))[1]

        sender_filename = f"{email_folder}/sender.txt"
        with open(sender_filename, "w", encoding="utf-8") as sender_file:
            sender_file.write(sender_address)
            
        received_date = email.utils.parsedate(email_message.get("Date"))
        if received_date:
            received_date_str = time.strftime("%Y-%m-%d %H:%M:%S", received_date)
            received_date_filename = f"{email_folder}/received_date.txt"
            with open(received_date_filename, "w", encoding="utf-8") as received_date_file:
                received_date_file.write(received_date_str)

        # Append data to the list for Pandas DataFrame
        data_list.append([sender_address, subject, html])

    except Exception as fetch_error:
        logger.error(f"Error processing email {msg_id}: {fetch_error}")
        traceback.print_exc()

# Create a directory to store extracted HTML files
output_folder = "extracted"
os.makedirs(output_folder, exist_ok=True)

def main():
    max_retries = 6
    retry_count = 0

    while retry_count < max_retries:
        mail = connect_to_imap()

        if mail:
            try:
                mail.select("inbox")
                status, messages = mail.search(None, "(UNSEEN)")
                messages = messages[0].split()

                logger.info(f"Total unread messages: {len(messages)}")

                for msg_id in messages:
                    result, msg_data = mail.fetch(msg_id, "(RFC822)")
                    raw_email = msg_data[0][1]
                    email_message = email.message_from_bytes(raw_email)

                    process_email(msg_id, email_message)
                    
            except Exception as processing_error:
                logger.error(f"Error processing emails: {processing_error}")
                traceback.print_exc()
                
            finally:
                mail.logout()
                break
        else:
            retry_count += 1
            logger.info(f"Retrying connection ({retry_count}/{max_retries})...")
            time.sleep(2 ** retry_count)  # Exponential backoff

    if retry_count == max_retries:
        logger.error("Failed to connect to the IMAP server after multiple retries.")
        
    # Create a Pandas DataFrame
    columns = ['sender', 'email_subject', 'html_css_code']
    df = pd.DataFrame(data_list, columns=columns)
    
    # Export the DataFrame to a TSV file
    tsv_file_path = os.path.join(os.getcwd(), "ScraperOutput.tsv")
    df.to_csv(tsv_file_path, sep='\t', index=False)

    # Display the DataFrame
    print(df)
    print(f"\nDataFrame exported to: {tsv_file_path}")

if __name__ == "__main__":
    main()
