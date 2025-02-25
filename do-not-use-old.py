import os
import re
import binascii
import base64
import logging
import traceback
import asyncio
import imaplib
import email
from email.header import decode_header, Header
from email.errors import HeaderParseError
from datetime import datetime
import uuid
import boto3
from supabase import create_client
from bs4 import BeautifulSoup
import playwright
from playwright.async_api import async_playwright
from PIL import Image

# Lets roll some code!

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


# Supabase client env
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info(f"Initializing Supabase client with URL: {SUPABASE_URL}")


# S3 credentials env
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')

# Email credentials env
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# Path to Chrome executable -- To Dockerize
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

logger.info(f"Connecting to IMAP server at: mail.newslettermonster.com")


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

async def take_screenshot(html_content, uuid_val):
    async with playwright.async_api.async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # Set the viewport size to capture full email height
            await page.evaluate('''() => {
                const body = document.querySelector('body');
                const height = Math.max(body.scrollHeight, body.offsetHeight, body.clientHeight);
                return { width: 680, height };
            }''')

            # Navigate to a blank page to ensure the HTML content is fully loaded
            await page.goto("about:blank")

            # Wait for the page to finish loading
            await page.wait_for_load_state("networkidle")

            # Set the HTML content
            await page.set_content(html_content)

            # Set viewport width to 680px for full-page screenshot
            await page.set_viewport_size({"width": 680, "height": page.viewport_size["height"]})

            # Take a full-page screenshot of email
            full_screenshot_path = f"{uuid_val}_full.png"
            await page.screenshot(path=full_screenshot_path, full_page=True)
            logger.info(f"Full-page screenshot saved: {full_screenshot_path}")

            # Take a cropped screenshot (680x900)
            thumb_screenshot_path = f"{uuid_val}_small.png"
            await page.set_viewport_size({"width": 680, "height": 900})
            await page.evaluate('window.scrollTo(0, 0)')
            await page.screenshot(path=thumb_screenshot_path)
            logger.info(f"Thumbnail screenshot saved: {thumb_screenshot_path}")

            # Convert screenshots to WebP format
            await convert_to_webp(full_screenshot_path)
            await convert_to_webp(thumb_screenshot_path)

            # Upload WebP screenshots to S3
            await upload_to_s3(full_screenshot_path.replace(".png", ".webp"), uuid_val)
            await upload_to_s3(thumb_screenshot_path.replace(".png", ".webp"), uuid_val)

        except Exception as e:
            logger.error(f"Error taking screenshots: {e}")
            traceback.print_exc()
        finally:
            if page:
                await page.close()
            if browser:
                await browser.close()

# Convert image to webp by saving as png and deleteing afterwards (avoids bug)

async def convert_to_webp(image_path):
    try:
        # Open the PNG image
        with Image.open(image_path) as img:
            # Save the image in WebP format
            webp_path = os.path.splitext(image_path)[0] + ".webp"
            img.save(webp_path, "webp")
            logger.info(f"Image converted to WebP: {webp_path}")

            # Delete local PNG file after conversion
        os.remove(image_path)
        logger.info(f"Local PNG file deleted after conversion: {image_path}")

    except Exception as e:
        logger.error(f"Error converting image to WebP: {e}")
        traceback.print_exc()

# Upload images to S3 Bucket

async def upload_to_s3(image_path, uuid_val):
    try:
        s3 = boto3.client('s3', 
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

        # Generate S3 key with filename
        key = os.path.basename(image_path)

        # Upload image to S3 bucket
        with open(image_path, 'rb') as f:
            s3.put_object(Body=f, Bucket=S3_BUCKET, Key=f"{uuid_val}/{key}", ContentType='image/webp')

        # Remove local image file after uploading to S3
        os.remove(image_path)
        logger.info("Local image file deleted after uploading to S3")
    except Exception as e:
        logger.error(f"Error uploading image to S3: {e}")
        traceback.print_exc()

# Decode subject line if encoded
def decode_subject(subject):
    try:
        # Decode subject line using decode_header
        decoded_parts = decode_header(subject)
        decoded_subject = ''.join(part[0].decode(part[1] or 'utf-8') if isinstance(part[0], bytes) else part[0] for part in decoded_parts)
        return decoded_subject
    except HeaderParseError as e:
        logger.error(f"Error decoding subject: {e}")
        return subject  # Return original subject if decoding fails

# Decode sender email address if encoded
def decode_sender(sender):
    try:
        decoded_parts = decode_header(sender)
        decoded_sender = ''.join(part[0].decode(part[1] or 'utf-8') if isinstance(part[0], bytes) else part[0] for part in decoded_parts)
        return decoded_sender
    except Exception as e:
        logger.error(f"Error decoding sender: {e}")
        return sender  # Return original sender if decoding fails

# Main function
async def main():
    while True:
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
                for msg_id in msg_ids[0].split():
                    _, msg_data = mail.fetch(msg_id, '(RFC822)')
                    email_msg = email.message_from_bytes(msg_data[0][1])

                    # Decode subject line if encoded
                    subject_decoded = decode_subject(email_msg['Subject'])

                    # Decode sender's name and email address
                    sender_email = decode_sender(email_msg['From'])

                    # Extract sender's name from decoded sender - Getting company name.
                    sender_name = sender_email.split('<')[0].strip().replace('"', '')

                    # Extract HTML content from email
                    html_content = get_email_html(email_msg)
                    if html_content:
                        # Generate UUID for the current processing (ver4)
                        uuid_val = str(uuid.uuid4())

                        # Fill Supabase table with email details once per email
                        real_title = ''.join(ch for ch in subject_decoded if ch.isalnum() or ch.isspace())
                        created_at = datetime.now().isoformat()
                        supabase.table("TableN1").insert({
                            "subject": subject_decoded,  # Use decoded subject
                            "sender": sender_email,  # Use decoded sender email
                            "company": sender_name,  # Insert cleaned sender's name into 'company' column - used for indexing companies in Supabase trigger
                            "created_at": created_at,
                            "real_title": real_title,
                            "uuid_script": uuid_val,  # Insert UUID into 'uuid_script' column - used for url later
                            "date_processed": created_at  # Insert the date and time of processing
                        }).execute()

                        # Save HTML content to a local file
                        html_file_path = f"{uuid_val}.html"
                        with open(html_file_path, 'w', encoding='utf-8') as file:
                            file.write(html_content)

                        # Upload HTML file to S3 with correct content type to avoid S3 error
                        s3 = boto3.client('s3', 
                                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
                        s3.put_object(Body=html_content.encode(), Bucket=S3_BUCKET, Key=f"{uuid_val}/{uuid_val}.html", ContentType='text/html')

                        # After uploading HTML file to S3 - creating link for content & internal linking
                        html_s3_link = f"https://nlmr1.s3.eu-central-1.amazonaws.com/{uuid_val}/{uuid_val}.html"

                        # Update Supabase table with S3 link to HTML file
                        supabase.table("TableN1").update({
                            "S3link_html": html_s3_link
                        }).eq("uuid_script", uuid_val).execute()

                        # Generate URL for small webp image creating link for content & internal linking
                        small_webp_url = f"https://nlmr1.s3.eu-central-1.amazonaws.com/{uuid_val}/{uuid_val}_small.webp"

                        # Generate URL for full webp image creating link for content & internal linking
                        full_webp_url = f"https://nlmr1.s3.eu-central-1.amazonaws.com/{uuid_val}/{uuid_val}_full.webp"

                        # Insert URLs into Supabase for further referencing
                        supabase.table("TableN1").update({
                            "small_url_Webp": small_webp_url,
                            "full_url_Webp": full_webp_url
                        }).eq("uuid_script", uuid_val).execute()                        

                        # Take screenshots and upload them to S3
                        await take_screenshot(html_content, uuid_val)

                        # Delete the local HTML file - not working!!
                        os.remove(html_file_path)
                        logger.info(f"Local HTML file deleted after uploading to S3: {html_file_path}")
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
            await asyncio.sleep(60)  # Wait for 60 seconds before processing next batch of emails

if __name__ == "__main__":
    asyncio.run(main())