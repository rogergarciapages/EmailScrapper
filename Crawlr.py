import os
import re
import binascii
import base64
import logging
import traceback
import asyncio
import imaplib
import email
from email.header import decode_header
from datetime import datetime
import uuid
import boto3
from supabase import create_client
from bs4 import BeautifulSoup
import playwright
from playwright.async_api import async_playwright


# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# S3 credentials
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')

# Email credentials
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# Path to Chrome executable
CHROME_EXECUTABLE_PATH = "C:\\Users\\Usuario\\Documents\\GitHub\\chrome-driver\\chrome.exe"

# Connect to IMAP server
def connect_to_imap():
    """
    Connects to the IMAP server and returns the connection object.
    """
    try:
        mail = imaplib.IMAP4_SSL("mail.newslettermonster.com", 993)
        mail.login(EMAIL_USER, EMAIL_PASS)
        return mail
    except imaplib.IMAP4.error as e:
        logger.error(f"IMAP connection error: {e}")
        traceback.print_exc()
        return None

# Get HTML content from email message
def get_email_html(email_msg):
    """
    Extracts the HTML content from the email message.
    """
    html = None
    for part in email_msg.walk():
        if part.get_content_type() == "text/html":
            charset = part.get_content_charset()
            html = part.get_payload(decode=True).decode(charset)
            break
    return html

# Function to take screenshots
async def take_screenshot(html_content, uuid_val):
    async with playwright.async_api.async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            await page.set_content(html_content)

            # Take a full-page screenshot
            full_screenshot_path = f"{uuid_val}_full.png"
            await page.screenshot(path=full_screenshot_path, full_page=True)
            logger.info(f"Full-page screenshot saved: {full_screenshot_path}")

            # Take a cropped screenshot (900x900)
            thumb_screenshot_path = f"{uuid_val}_small.png"
            await page.set_viewport_size({'width': 900, 'height': 900})
            await page.evaluate('window.scrollTo(0, 0)')
            await page.screenshot(path=thumb_screenshot_path)
            logger.info(f"Thumbnail screenshot saved: {thumb_screenshot_path}")

            # Upload screenshots to S3
            upload_to_s3(full_screenshot_path, uuid_val)
            upload_to_s3(thumb_screenshot_path, uuid_val)
        except Exception as e:
            logger.error(f"Error taking screenshots: {e}")
            traceback.print_exc()
        finally:
            if page:
                await page.close()
            if browser:
                await browser.close()
                
# Upload screenshot to S3
def upload_to_s3(image_path, uuid_val):
    try:
        s3 = boto3.client('s3', 
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

        # Generate S3 key with filename
        key = os.path.basename(image_path)

        # Upload image to S3 bucket
        with open(image_path, 'rb') as f:
            s3.put_object(Body=f, Bucket=S3_BUCKET, Key=f"{uuid_val}/{key}", ContentType='image/png')

        # Remove local image file after uploading to S3
        os.remove(image_path)
        logger.info("Local image file deleted after uploading to S3")
    except Exception as e:
        logger.error(f"Error uploading image to S3: {e}")
        traceback.print_exc()

# Function to decode subject line if encoded
def decode_subject(subject):
    # Check if subject is encoded using Base64
    if subject.startswith('=?utf-8?B?') and subject.endswith('?='):
        # Remove encoding prefix and suffix
        encoded_subject = subject[len('=?utf-8?B?'):-len('?=')]
        # Decode Base64 and return decoded subject
        decoded_subject = base64.b64decode(encoded_subject).decode('utf-8', errors='ignore')
    else:
        # If subject is not encoded, return the subject as is
        decoded_subject = subject

    return decoded_subject

# Function to decode subject line if encoded
def decode_subject(subject):
    # Check if subject is encoded using Base64
    if subject.startswith('=?utf-8?B?') and subject.endswith('?='):
        # Remove encoding prefix and suffix
        encoded_subject = subject[len('=?utf-8?B?'):-len('?=')]
        # Decode Base64 and return decoded subject
        return base64.b64decode(encoded_subject).decode('utf-8', errors='ignore')
    else:
        return subject

# Extract sender's name from email address
def extract_sender_name(sender):
    # Check if sender is enclosed in double quotes
    match = re.match(r'"([^"]*)"', sender)
    if match:
        return match.group(1)  # Return text within double quotes
    else:
        # Extract part of sender's email address before '<'
        return sender.split('<')[0].strip()

# Main function
async def main():
    # Connect to IMAP server
    mail = connect_to_imap()
    if not mail:
        logger.error("Failed to connect to IMAP server. Exiting.")
        return

    try:
        # Select inbox
        mail.select('INBOX')

        # Search for latest unseen message
        _, msg_ids = mail.search(None, 'UNSEEN')
        if msg_ids:
            latest_msg_id = msg_ids[0].split()[-1]
            _, msg_data = mail.fetch(latest_msg_id, '(RFC822)')
            email_msg = email.message_from_bytes(msg_data[0][1])

            # Decode subject line if encoded
            subject_decoded = decode_subject(email_msg['Subject'])

            # Extract sender's name from email address
            sender_email = email_msg['From']
            sender_name = extract_sender_name(sender_email)

            # Extract HTML content from email
            html_content = get_email_html(email_msg)
            if html_content:
                # Generate UUID for the current processing
                uuid_val = str(uuid.uuid4())

                # Fill Supabase table with email details
                real_title = ''.join(ch for ch in subject_decoded if ch.isalnum() or ch.isspace())
                created_at = datetime.now().isoformat()
                supabase.table("TableN1").insert({
                    "subject": subject_decoded,  # Use decoded subject
                    "sender": sender_email,  # Keep original sender format
                    "company": sender_name,  # Insert cleaned sender's name into 'company' column
                    "created_at": created_at,
                    "real_title": real_title,
                }).execute()

                # Save HTML content to a local file
                html_file_path = f"{uuid_val}.html"
                with open(html_file_path, 'w', encoding='utf-8') as file:
                    file.write(html_content)

                # Upload HTML file to S3
                s3 = boto3.client('s3', 
                                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
                s3.put_object(Body=html_content.encode(), Bucket=S3_BUCKET, Key=f"{uuid_val}/{uuid_val}.html")

                # Take screenshots and upload them to S3
                await take_screenshot(html_content, uuid_val)
            else:
                logger.warning("No HTML content found in the email.")
        else:
            logger.info("No unseen messages found in the inbox.")

    except Exception as e:
        logger.error(f"Error processing emails: {e}")
        traceback.print_exc()
    finally:
        # Close IMAP connection
        mail.logout()

if __name__ == "__main__":
    asyncio.run(main())
