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
from playwright.async_api import async_playwright
from PIL import Image
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Integer, DateTime, func, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import UUID as SQLAUUID
from dateutil import parser as dateutil_parser
import time

# Load environment variables from .env file
load_dotenv()

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Supabase client env
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# S3 credentials env
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')
S3_REGION = os.getenv('AWS_REGION', 'eu-central-1')  # Default to 'eu-central-1' if not set

# Email credentials env
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Check your .env file and ensure the environment variable is set.")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define the models
class User(Base):
    __tablename__ = 'User'
    user_id = Column(SQLAUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, unique=True, nullable=True)
    username = Column(String, unique=True, nullable=True)
    email = Column(String, unique=True, nullable=False)
    profile_photo = Column(Text, nullable=True)
    password = Column(Text, nullable=False)
    created_at = Column(DateTime, default=func.now())
    role = Column(String, nullable=False)  # Ensure role is required

class Newsletter(Base):
    __tablename__ = 'Newsletter'
    newsletter_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(SQLAUUID(as_uuid=True), index=True, nullable=False)
    sender = Column(String, index=True)
    date = Column(DateTime, default=func.now())
    html_file_url = Column(Text)
    full_screenshot_url = Column(Text)
    top_screenshot_url = Column(Text)
    likes_count = Column(Integer, default=0)
    you_rocks_count = Column(Integer, default=0)  # Corrected field name
    created_at = Column(DateTime, default=func.now())

Base.metadata.create_all(bind=engine)

# Connect to IMAP server
def connect_to_imap(retry_count=5):
    for attempt in range(retry_count):
        try:
            mail = imaplib.IMAP4_SSL("mail.newslettermonster.com", 993)
            mail.login(EMAIL_USER, EMAIL_PASS)
            return mail
        except imaplib.IMAP4.abort as e:
            logger.error(f"IMAP connection error: {e}")
            traceback.print_exc()
            if attempt < retry_count - 1:
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                return None

def get_master_user_id():
    db = SessionLocal()
    master_user = db.query(User).filter(User.username == 'themonster').first()
    db.close()
    if master_user:
        return master_user.user_id
    raise ValueError("Master user 'The Monster' not found. Please ensure the master user exists in the database.")

# Get HTML content from email message
def get_email_html(email_msg):
    html = None
    for part in email_msg.walk():
        if part.get_content_type() == "text/html":
            charset = part.get_content_charset()
            html = part.get_payload(decode=True).decode(charset)
            break
    return html

# Convert image to webp and delete original png
async def convert_to_webp(image_path):
    try:
        with Image.open(image_path) as img:
            webp_path = os.path.splitext(image_path)[0] + ".webp"
            img.save(webp_path, "webp")
            logger.info(f"Image converted to WebP: {webp_path}")
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
        key = os.path.basename(image_path)
        with open(image_path, 'rb') as f:
            s3.put_object(Body=f, Bucket=S3_BUCKET, Key=f"{uuid_val}/{key}", ContentType='image/webp')
        os.remove(image_path)
        logger.info("Local image file deleted after uploading to S3")
    except Exception as e:
        logger.error(f"Error uploading image to S3: {e}")
        traceback.print_exc()

# Decode subject line if encoded
def decode_subject(subject):
    try:
        decoded_parts = decode_header(subject)
        decoded_subject = ''.join(part[0].decode(part[1] or 'utf-8') if isinstance(part[0], bytes) else part[0] for part in decoded_parts)
        return decoded_subject
    except HeaderParseError as e:
        logger.error(f"Error decoding subject: {e}")
        return subject

# Decode sender email address if encoded
def decode_sender(sender):
    try:
        decoded_parts = decode_header(sender)
        decoded_sender = ''.join(part[0].decode(part[1] or 'utf-8') if isinstance(part[0], bytes) else part[0] for part in decoded_parts)
        return decoded_sender
    except Exception as e:
        logger.error(f"Error decoding sender: {e}")
        return sender

# Take screenshot of HTML content and upload to S3
async def take_screenshot(html_content, uuid_val):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        page.set_default_timeout(60000)  # Set default timeout to 60 seconds

        try:
            await page.goto("about:blank")
            await page.set_content(html_content)
            await page.set_viewport_size({"width": 680, "height": await page.evaluate("document.body.scrollHeight")})

            full_screenshot_path = f"{uuid_val}_full.png"
            await page.screenshot(path=full_screenshot_path, full_page=True)
            logger.info(f"Full-page screenshot saved: {full_screenshot_path}")

            thumb_screenshot_path = f"{uuid_val}_small.png"
            await page.set_viewport_size({"width": 680, "height": 900})
            await page.evaluate('window.scrollTo(0, 0)')
            await page.screenshot(path=thumb_screenshot_path)
            logger.info(f"Thumbnail screenshot saved: {thumb_screenshot_path}")

            await convert_to_webp(full_screenshot_path)
            await convert_to_webp(thumb_screenshot_path)
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

async def upload_html_and_take_screenshot(html_content, uuid_val):
    # Save HTML content to a local file
    html_file_path = f"{uuid_val}.html"
    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)

    # Upload HTML file to S3
    try:
        s3 = boto3.client('s3',
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        s3.put_object(Body=html_content.encode(), Bucket=S3_BUCKET, Key=f"{uuid_val}/{uuid_val}.html", ContentType='text/html')
        html_s3_link = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{uuid_val}/{uuid_val}.html"

        # Take screenshots and upload them to S3
        await take_screenshot(html_content, uuid_val)
        
        # Delete the local HTML file after successful upload and screenshot
        os.remove(html_file_path)
        logger.info(f"Local HTML file deleted after uploading to S3 and taking screenshot: {html_file_path}")

    except Exception as e:
        logger.error(f"Error uploading HTML to S3 or taking screenshot: {e}")
        traceback.print_exc()

    return html_s3_link

# Function to parse date and time from email
def parse_email_date(email_date_str):
    try:
        # This will automatically handle various date formats and time zones
        return dateutil_parser.parse(email_date_str)
    except ValueError as e:
        logger.error(f"Error parsing date: {e}")
        # Handle the error or return a default value
        return None

# Main function
async def main():
    while True:
        mail = connect_to_imap()
        if not mail:
            logger.error("Failed to connect to IMAP server. Exiting.")
            return

        master_user_id = get_master_user_id()

        try:
            mail.select('INBOX')
            _, msg_ids = mail.search(None, 'UNSEEN')
            if msg_ids:
                for msg_id in msg_ids[0].split():
                    _, msg_data = mail.fetch(msg_id, '(RFC822)')
                    email_msg = email.message_from_bytes(msg_data[0][1])

                    subject_decoded = decode_subject(email_msg['Subject'])
                    sender_email = decode_sender(email_msg['From'])
                    sender_name = sender_email.split('<')[0].strip().replace('"', '')

                    html_content = get_email_html(email_msg)
                    if html_content:
                        uuid_val = str(uuid.uuid4())

                        # Upload HTML and take screenshot
                        html_s3_link = await upload_html_and_take_screenshot(html_content, uuid_val)

                        # Retrieve the sending date of the email
                        email_date_str = email_msg['Date']
                        email_date = parse_email_date(email_date_str)

                        # Insert newsletter data into the database
                        db = SessionLocal()
                        newsletter = Newsletter(
                            user_id=master_user_id,  # Setting user_id to master_user_id if not provided
                            sender=sender_name,
                            date=email_date,
                            html_file_url=html_s3_link,
                            full_screenshot_url=f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{uuid_val}/{uuid_val}_full.webp",
                            top_screenshot_url=f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{uuid_val}/{uuid_val}_small.webp",
                            likes_count=0,
                            you_rocks_count=0,
                            created_at=email_date  # Set created_at to the sending date of the email
                        )
                        db.add(newsletter)
                        db.commit()
                        db.close()
                    else:
                        logger.warning("No HTML content found in the email.")
            else:
                logger.info("No unseen messages found in the inbox.")
        except Exception as e:
            logger.error(f"Error processing emails: {e}")
            traceback.print_exc()
        finally:
            mail.logout()
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
