import os
import logging
import traceback
import imaplib
import email
from email.header import decode_header
import boto3
from botocore.exceptions import NoCredentialsError
from datetime import datetime
from supabase import create_client
from bs4 import BeautifulSoup
import time

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Email credentials
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# S3 credentials
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')

# Connect to IMAP server
def connect_to_imap():
    try:
        mail = imaplib.IMAP4_SSL("mail.newslettermonster.com", 993)
        mail.login(EMAIL_USER, EMAIL_PASS)
        return mail
    except imaplib.IMAP4.error as e:
        logger.error(f"IMAP connection error: {e}")
        traceback.print_exc()
        return None

# Extract HTML content from email
def extract_html(email_msg):
    html = None
    for part in email_msg.walk():
        if part.get_content_type() == "text/html":
            charset = part.get_content_charset()
            html = part.get_payload(decode=True).decode(charset)
            break
    if html is None:
        logger.warning("No HTML content found in the email.")
    else:
        print("SUPABASE_URL:", os.getenv('SUPABASE_URL'))
        print("SUPABASE_KEY:", os.getenv('SUPABASE_KEY'))
        logger.info("HTML content extracted successfully")
        logger.debug(f"HTML content: {html}")  # Add this line to log the HTML content
        # Parse HTML content using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        # Optionally, you can perform further processing or manipulation of the HTML content here
        # For example:
        # soup.find('a').extract()  # Remove all <a> tags from the HTML content
        html = soup.prettify()  # Convert BeautifulSoup object back to prettified HTML string
    return html

# Upload HTML content to S3
def upload_to_s3(html, uuid):
    try:
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, 
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        # Generate S3 key with UUID
        key = f"{uuid}.html"
        # Upload HTML content to S3 bucket
        s3.put_object(Body=html.encode(), Bucket=str(S3_BUCKET), Key=key)
        logger.info("HTML uploaded to S3")
        return key
    except NoCredentialsError:
        logger.error("Invalid AWS credentials")
        traceback.print_exc()
        return None
    except Exception as e:
        logger.error(f"S3 upload error: {e}")
        traceback.print_exc()
        return None


# Get email subject 
def get_email_subject(email_msg):
    subject, _ = decode_header(email_msg["Subject"])[0]
    return subject.decode() if isinstance(subject, bytes) else subject

# Get email sender
def get_email_sender(email_msg):
    return email.utils.parseaddr(email_msg.get("From"))[1]

# Get email received date
def get_email_date(email_msg):
    return email.utils.parsedate_to_datetime(email_msg.get("Date"))

# Process an email
def process_email(email_msg):
    try:
        # Extract info from email
        subject = get_email_subject(email_msg)
        sender = get_email_sender(email_msg)
        date = get_email_date(email_msg)

        # Get HTML content
        html = extract_html(email_msg)
        logger.debug(f"HTML content after extraction: {html}")  # Add this line to log the HTML content

        # Skip processing if html is None
        if html is None:
            logger.warning("Skipping email processing due to missing HTML content.")
            return

        # Insert record into Supabase
        uuid = insert_to_supabase(subject, sender, date)
        logger.debug(f"UUID: {uuid}")  # Add this line to log the UUID

        # Save HTML content to a local file
        file_path = f"{uuid}.html"
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html)
        logger.info("HTML content saved to local file")

        # Upload file to S3
        if uuid:
            s3_key = upload_to_s3(html, uuid)
            logger.debug(f"S3 key: {s3_key}")  # Add this line to log the S3 key
    except Exception as e:
        logger.error(f"Error processing email: {e}")
        traceback.print_exc()

# Insert record into Supabase
def insert_to_supabase(subject, sender, date):
    try:
        # Insert record into Supabase table
        response = supabase.table("TableN1").insert({
            "subject": subject,
            "sender": sender,
            "created_at": date.isoformat(),  # Convert to ISO format string
        }).execute()
        logger.info("Record inserted into Supabase")
        
        # Check if response contains data
        if response.data:
            # Extract the ID from the response
            id = response.data[0]["id"]  
            return id
        else:
            logger.error("No data found in Supabase response")
            return None
    except Exception as e:
        logger.error(f"Error inserting record into Supabase: {e}")
        traceback.print_exc()
        return None

# Update processing time in Supabase
def update_processing_time(uuid, processing_time):
    try:
        # Update record in Supabase table with processing time
        supabase.table("TableN1").update({
            "date_processed": processing_time,
        }).eq("id", uuid).execute()
        logger.info("Processing time updated in Supabase")
    except Exception as e:
        logger.error
        logger.error(f"Error updating processing time in Supabase: {e}")
        traceback.print_exc()

# Main function
if __name__ == "__main__":
    mail = connect_to_imap()
    if mail:
        try:
            # Select inbox
            mail.select('INBOX')

            # Get all message IDs
            _, msg_ids = mail.search(None, 'ALL')
            msg_ids = msg_ids[0].split()

            # Process each email
            for msg_id in msg_ids:
                _, msg_data = mail.fetch(msg_id, '(RFC822)')
                email_msg = email.message_from_bytes(msg_data[0][1])
                # Log the subject of the email message
                logger.info(f"Processing email with subject: {get_email_subject(email_msg)}")
                process_email(email_msg)

        except Exception as e:
            logger.error(f"Error processing emails: {e}")
            traceback.print_exc()
        finally:
            # Close IMAP connection
            mail.logout()