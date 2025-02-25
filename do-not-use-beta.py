import os
import re
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
import json
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from PIL import Image
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dateutil import parser as dateutil_parser
import google.generativeai as genai
import time
import psycopg2
from psycopg2.extras import DictCursor
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Postgres client env
def get_db_config():
    db_url = os.getenv('DIRECT_DATABASE_URL')
    if not db_url:
        raise ValueError("DIRECT_DATABASE_URL is not set in environment variables")
    
    parsed = urlparse(db_url)
    return {
        'dbname': parsed.path[1:],
        'user': parsed.username,
        'password': parsed.password,
        'host': parsed.hostname,
        'port': parsed.port
    }

def get_db_connection():
    try:
        db_config = get_db_config()
        conn = psycopg2.connect(**db_config)
        conn.autocommit = False
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

# S3 credentials env
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')
S3_REGION = os.getenv('AWS_REGION', 'eu-central-1')

# Email credentials env
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# AI API key for Google Bard (Gemini)
BARD_API_KEY = os.getenv('BARD_API_KEY')

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Check your .env file and ensure the environment variable is set.")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT user_id FROM "User" WHERE username = %s', ('themonster',))
            result = cur.fetchone()
            if result:
                return result[0]
    raise ValueError("Master user 'The Monster' not found")

# Get HTML content from email message
def get_email_html(email_msg):
    html = None
    for part in email_msg.walk():
        if part.get_content_type() == "text/html":
            charset = part.get_content_charset()
            html = part.get_payload(decode=True).decode(charset)
            break
    return html

# Decode Unicode escape sequences
def decode_unicode_escape_sequences(text):
    return text.encode('utf-8').decode('unicode-escape')

# Extract text from HTML content
def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    return decode_unicode_escape_sequences(text)

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

# Generate summary, tags, and products link using Google Bard API
def generate_summary_and_tags(email_subject, email_content):
    genai.configure(api_key=BARD_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

    text_content = extract_text_from_html(email_content)

    prompt = f"""
Subject: {email_subject}

Email Content:
{text_content}

Task:
Mimic the tone and style of the newsletter sender to craft a compelling summary, extract relevant tags, and provide a product link. Ensure the summary feels like it came directly from the sender and make sure no weird strings of text, symbols, or characters are added.

1. Provide a concise, compelling summary of the newsletter content and what it promotes in the sender's voice. Ensure no slashes, backslashes, or asterisks are included in the summary.
2. Extract relevant subjects from the content of the email, convert them into tags and remove any spaces between words if tags have two or more words. Use camelCase without hyphens, dots, or underscores. Provide tags as comma-separated values.
3. Provide the most relevant link to any product, post, or website mentioned in the newsletter in URL format only. If no relevant link is found, provide the URL for the homepage of the business, product, or service. Ensure the URL starts with https:// and is valid. Remove any text before or after the URL and avoid annotations.

Output format:
Summary: <extracted_summary>
Tags: <extracted_tags>
Products Link: <extracted_products_link>
"""

    try:
        response = model.generate_content(prompt)
        raw_text = response.text

        # Enhanced parsing logic
        summary_match = re.search(r"Summary:\s*(.*?)(Tags:|$)", raw_text, re.DOTALL)
        tags_match = re.search(r"Tags:\s*(.*?)(Products Link:|$)", raw_text, re.DOTALL)
        products_link_match = re.search(r"Products Link:\s*(.*?)([`\n]|$)", raw_text, re.DOTALL)

        summary = summary_match.group(1).strip() if summary_match else None
        tags = tags_match.group(1).strip().split(',') if tags_match and tags_match.group(1).strip() else []
        products_link = products_link_match.group(1).strip() if products_link_match else None

        return {
            "summary": summary,
            "tags": tags,
            "products_link": products_link
        }
    except Exception as e:
        logger.error(f"Error calling AI API: {e}")
        raise

# Insert tags into the database and return their IDs

def create_tag_slug(tag_name: str) -> str:
    """Create a URL-friendly slug from a tag name."""
    # Convert to lowercase
    slug = tag_name.lower()
    # Replace spaces and special chars with hyphens
    slug = re.sub(r'[^a-z0-9]+', '-', slug)
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    return slug

def get_unique_slug(cur, base_slug: str) -> str:
    """Get a unique slug, adding numbers if necessary."""
    slug = base_slug
    counter = 1
    while True:
        cur.execute('SELECT id FROM "Tag" WHERE slug = %s', (slug,))
        if not cur.fetchone():
            return slug
        slug = f"{base_slug}-{counter}"
        counter += 1

def get_or_create_tags(tags, conn, cur):
    """Get or create tags with proper slug handling."""
    tag_ids = []
    for tag_name in tags:
        tag_name = tag_name.strip()
        if not tag_name:
            continue

        # Try to find existing tag
        cur.execute(
            'SELECT id, name, slug FROM "Tag" WHERE LOWER(name) = LOWER(%s)',
            (tag_name,)
        )
        result = cur.fetchone()

        if result:
            tag_ids.append(result[0])
        else:
            # Create new tag
            base_slug = create_tag_slug(tag_name)
            unique_slug = get_unique_slug(cur, base_slug)
            
            cur.execute(
                '''INSERT INTO "Tag" (name, slug, count, "createdAt", "updatedAt")
                   VALUES (%s, %s, %s, NOW(), NOW()) RETURNING id''',
                (tag_name, unique_slug, 0)
            )
            tag_id = cur.fetchone()[0]
            tag_ids.append(tag_id)
            
    return tag_ids

# Main function
async def main():
    while True:
        mail = connect_to_imap()
        if not mail:
            logger.error("Failed to connect to IMAP server. Retrying in 60 seconds...")
            await asyncio.sleep(60)
            continue

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    master_user_id = get_master_user_id()

                    mail.select('INBOX')
                    _, msg_ids = mail.search(None, 'UNSEEN')
                    
                    if msg_ids and msg_ids[0]:
                        for msg_id in msg_ids[0].split():
                            try:
                                _, msg_data = mail.fetch(msg_id, '(RFC822)')
                                email_msg = email.message_from_bytes(msg_data[0][1])

                                subject_decoded = decode_subject(email_msg['Subject'])
                                sender_email = decode_sender(email_msg['From'])
                                sender_name = sender_email.split('<')[0].strip().replace('"', '')

                                html_content = get_email_html(email_msg)
                                if not html_content:
                                    logger.warning(f"No HTML content found in email: {subject_decoded}")
                                    continue

                                uuid_val = str(uuid.uuid4())

                                # Generate summary and tags using AI API
                                try:
                                    ai_response = generate_summary_and_tags(subject_decoded, html_content)
                                    logger.info(f"AI API Response: {json.dumps(ai_response, indent=2)}")
                                    summary = ai_response.get("summary")
                                    tags = [tag.strip() for tag in ai_response.get("tags", [])]
                                    products_link = ai_response.get("products_link")
                                except Exception as e:
                                    logger.error(f"Error calling AI API: {e}")
                                    summary = None
                                    tags = []
                                    products_link = None

                                # Upload HTML and take screenshots
                                try:
                                    html_s3_link = await upload_html_and_take_screenshot(html_content, uuid_val)
                                    # Define S3 URLs for screenshots
                                    full_screenshot_url = f'https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{uuid_val}/{uuid_val}_full.webp'
                                    top_screenshot_url = f'https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{uuid_val}/{uuid_val}_small.webp'
                                except Exception as e:
                                    logger.error(f"Error processing screenshots: {e}")
                                    continue

                                # Parse email date
                                email_date = parse_email_date(email_msg['Date'])
                                if not email_date:
                                    email_date = datetime.now()

                                try:
                                    # Begin transaction
                                    cur.execute("BEGIN")

                                    # Insert newsletter
                                    cur.execute("""
                                        INSERT INTO "Newsletter" (
                                            user_id, sender, date, subject, html_file_url,
                                            full_screenshot_url, top_screenshot_url,
                                            likes_count, you_rocks_count, created_at,
                                            summary, products_link
                                        ) 
                                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                        RETURNING newsletter_id
                                    """, (
                                        master_user_id, sender_name, email_date, subject_decoded,
                                        html_s3_link, full_screenshot_url, top_screenshot_url,
                                        0, 0, email_date, summary, products_link
                                    ))
                                    
                                    newsletter_id = cur.fetchone()[0]

                                    # Process tags
                                    if tags:
                                        tag_ids = get_or_create_tags(tags, conn, cur)
                                        for tag_id in tag_ids:
                                            # Create newsletter-tag relationship
                                            cur.execute("""
                                                INSERT INTO "NewsletterTag" (newsletter_id, tag_id)
                                                VALUES (%s, %s)
                                            """, (newsletter_id, tag_id))

                                            # Update tag count
                                            cur.execute("""
                                                UPDATE "Tag"
                                                SET count = (
                                                    SELECT COUNT(*) 
                                                    FROM "NewsletterTag" 
                                                    WHERE tag_id = %s
                                                ),
                                                "updatedAt" = NOW()
                                                WHERE id = %s
                                            """, (tag_id, tag_id))

                                    # Commit transaction
                                    conn.commit()
                                    logger.info(f"Successfully processed newsletter: {subject_decoded} (ID: {newsletter_id})")
                                    logger.info(f"Successfully processed {len(tags)} tags for newsletter ID: {newsletter_id}")
                                    logger.info(f"Screenshot URLs generated: {full_screenshot_url}")

                                    # Mark email as read only after successful processing
                                    mail.store(msg_id, '+FLAGS', '\\Seen')

                                except Exception as e:
                                    conn.rollback()
                                    logger.error(f"Error processing newsletter {subject_decoded}: {e}")
                                    traceback.print_exc()

                            except Exception as e:
                                logger.error(f"Error processing email message: {e}")
                                traceback.print_exc()
                                continue
                    else:
                        logger.info("No unseen messages found in inbox")

        except Exception as e:
            logger.error(f"Error in main processing loop: {e}")
            traceback.print_exc()
        finally:
            try:
                mail.logout()
            except Exception as e:
                logger.error(f"Error logging out from IMAP: {e}")

        # Wait before next check
        logger.info("Waiting 60 seconds before next check...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())