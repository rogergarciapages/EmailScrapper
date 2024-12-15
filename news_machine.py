import os
import re
import logging
import traceback
import asyncio
import imaplib
import email
from email.header import decode_header, Header
from email.errors import HeaderParseError
from datetime import datetime, timezone
import uuid
import boto3
import json
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import time
import psycopg2
from psycopg2.extras import DictCursor
from urllib.parse import urlparse
from dateutil import parser as dateutil_parser
from typing import Dict, List, Optional

# Load environment variables
load_dotenv()

# Logging setup
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')
S3_REGION = os.getenv('AWS_REGION', 'eu-central-1')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
BARD_API_KEY = os.getenv('BARD_API_KEY')

class NewsletterProcessor:
    def __init__(self):
        # Initialize Gemini
        genai.configure(api_key=BARD_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        self.system_prompt = """You are an AI trained to analyze newsletters and extract key information.
Your task is to:
1. Identify the main topics and themes
2. Extract key insights and takeaways
3. Identify any products or services mentioned
4. Maintain the original sender's tone and style
5. Format tags in PascalCase (e.g., "ArtificialIntelligence", "ProductUpdate")

Please format your response in the following structure:
Summary: [A concise summary of the newsletter]
Tags: [Comma-separated list of PascalCase tags]
Products: [Comma-separated list of products/services mentioned]
Key Insights: [Bullet points of key takeaways]

Focus on providing accurate, concise information while preserving the newsletter's voice."""

    def get_db_config(self):
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

    def get_db_connection(self):
        try:
            db_config = self.get_db_config()
            conn = psycopg2.connect(**db_config)
            conn.autocommit = False
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def connect_to_imap(self, retry_count=5):
        for attempt in range(retry_count):
            try:
                mail = imaplib.IMAP4_SSL("mail.newslettermonster.com", 993)
                mail.login(EMAIL_USER, EMAIL_PASS)
                return mail
            except imaplib.IMAP4.abort as e:
                logger.error(f"IMAP connection error: {e}")
                if attempt < retry_count - 1:
                    time.sleep(5)
                else:
                    return None

    def get_master_user_id(self):
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT user_id FROM "User" WHERE username = %s', ('themonster',))
                result = cur.fetchone()
                if result:
                    return result[0]
        raise ValueError("Master user 'The Monster' not found")

    async def convert_to_webp(self, image_path):
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

    async def upload_to_s3(self, image_path, uuid_val):
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

    async def take_screenshot(self, html_content, uuid_val):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            page.set_default_timeout(60000)

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

                await self.convert_to_webp(full_screenshot_path)
                await self.convert_to_webp(thumb_screenshot_path)
                await self.upload_to_s3(full_screenshot_path.replace(".png", ".webp"), uuid_val)
                await self.upload_to_s3(thumb_screenshot_path.replace(".png", ".webp"), uuid_val)

            except Exception as e:
                logger.error(f"Error taking screenshots: {e}")
                traceback.print_exc()
            finally:
                if page:
                    await page.close()
                if browser:
                    await browser.close()

    async def upload_html_and_take_screenshot(self, html_content, uuid_val):
        html_file_path = f"{uuid_val}.html"
        with open(html_file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)

        try:
            s3 = boto3.client('s3',
                            aws_access_key_id=AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
            s3.put_object(Body=html_content.encode(), Bucket=S3_BUCKET, Key=f"{uuid_val}/{uuid_val}.html", ContentType='text/html')
            html_s3_link = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{uuid_val}/{uuid_val}.html"

            await self.take_screenshot(html_content, uuid_val)
            
            os.remove(html_file_path)
            logger.info(f"Local HTML file deleted after uploading to S3 and taking screenshot: {html_file_path}")

            return html_s3_link

        except Exception as e:
            logger.error(f"Error uploading HTML to S3 or taking screenshot: {e}")
            traceback.print_exc()

    def create_tag_slug(self, tag_name: str) -> str:
        slug = tag_name.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug

    def get_unique_slug(self, cur, base_slug: str) -> str:
        slug = base_slug
        counter = 1
        while True:
            cur.execute('SELECT id FROM "Tag" WHERE slug = %s', (slug,))
            if not cur.fetchone():
                return slug
            slug = f"{base_slug}-{counter}"
            counter += 1

    def get_or_create_tags(self, tags, conn, cur):
        tag_ids = []
        for tag_name in tags:
            tag_name = tag_name.strip()
            if not tag_name:
                continue

            cur.execute(
                'SELECT id, name, slug FROM "Tag" WHERE LOWER(name) = LOWER(%s)',
                (tag_name,)
            )
            result = cur.fetchone()

            if result:
                tag_ids.append(result[0])
            else:
                base_slug = self.create_tag_slug(tag_name)
                unique_slug = self.get_unique_slug(cur, base_slug)
                
                cur.execute(
                    '''INSERT INTO "Tag" (name, slug, count, "createdAt", "updatedAt")
                       VALUES (%s, %s, %s, NOW(), NOW()) RETURNING id''',
                    (tag_name, unique_slug, 0)
                )
                tag_id = cur.fetchone()[0]
                tag_ids.append(tag_id)
                
        return tag_ids

    async def process_email(self, email_content: str, subject: str) -> Dict:
        """Process raw email content and extract structured information."""
        soup = BeautifulSoup(email_content, 'html.parser')
        text_content = self._extract_text_with_structure(soup)
        analysis = await self._generate_content(text_content, subject)
        
        return {
            'content': text_content,
            'analysis': analysis,
            'processed_at': datetime.now(timezone.utc).isoformat()
        }

    def _extract_text_with_structure(self, soup: BeautifulSoup) -> str:
        for element in soup(['script', 'style']):
            element.decompose()
        
        preserved_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'blockquote']
        for tag in preserved_tags:
            for element in soup.find_all(tag):
                element.name = tag
        
        return str(soup)

    async def _generate_content(self, text: str, subject: str) -> Dict:
        try:
            full_prompt = f"{self.system_prompt}\n\nSubject: {subject}\n\nAnalyze this newsletter:\n{text}"
            
            # Add retry logic with exponential backoff
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await self.model.generate_content_async(full_prompt)
                    content = response.text
                    return self._parse_ai_response(content)
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                        logger.warning(f"API rate limit hit, waiting {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise
            
            logger.error(f"Error generating content after {max_retries} attempts")
            return {
                "summary": "",
                "tags": [],
                "products": [],
                "key_insights": []
            }
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return {
                "summary": "",
                "tags": [],
                "products": [],
                "key_insights": []
            }

    def _parse_ai_response(self, response: str) -> Dict:
        parsed_data = {
            "summary": "",
            "tags": [],
            "products": [],
            "key_insights": []
        }
        
        sections = {
            "summary": r"Summary:(.*?)(?=Tags:|$)",
            "tags": r"Tags:(.*?)(?=Products:|$)",
            "products": r"Products:(.*?)(?=Key Insights:|$)",
            "key_insights": r"Key Insights:(.*?)(?=$)"
        }
        
        for key, pattern in sections.items():
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if key == "tags":
                    parsed_data[key] = [tag.strip() for tag in content.split(',') if tag.strip()]
                elif key == "products":
                    parsed_data[key] = [product.strip() for product in content.split(',') if product.strip()]
                elif key == "key_insights":
                    insights = re.split(r'\n\s*[-•]\s*|\n\s*\d+\.\s*', content)
                    parsed_data[key] = [insight.strip() for insight in insights if insight.strip()]
                else:
                    parsed_data[key] = content
        
        return parsed_data

    async def process_and_save_email(self, email_msg) -> None:
        try:
            subject = decode_header(email_msg['Subject'])[0][0]
            if isinstance(subject, bytes):
                subject = subject.decode()
            
            sender_email = email_msg['From']
            sender_name = sender_email.split('<')[0].strip().replace('"', '')

            html_content = None
            for part in email_msg.walk():
                if part.get_content_type() == "text/html":
                    html_content = part.get_payload(decode=True).decode(part.get_content_charset())
                    break

            if not html_content:
                logger.warning(f"No HTML content found in email: {subject}")
                return

            uuid_val = str(uuid.uuid4())
            processed_data = await self.process_email(html_content, subject)
            
            # Upload HTML and take screenshots
            html_s3_link = await self.upload_html_and_take_screenshot(html_content, uuid_val)
            full_screenshot_url = f'https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{uuid_val}/{uuid_val}_full.webp'
            top_screenshot_url = f'https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{uuid_val}/{uuid_val}_small.webp'

            # Parse email date
            email_date = dateutil_parser.parse(email_msg['Date']) if email_msg['Date'] else datetime.now(timezone.utc)

            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    try:
                        master_user_id = self.get_master_user_id()
                        cur.execute("BEGIN")

                        # Get products link safely
                        products = processed_data['analysis'].get('products', [])
                        products_link = products[0] if products else None

                        # Insert newsletter with correct column names
                        cur.execute("""
                            INSERT INTO "Newsletter" (
                                user_id, sender, published_at, subject, html_file_url,
                                full_screenshot_url, top_screenshot_url,
                                likes_count, you_rocks_count, created_at,
                                summary, products_link
                            ) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING newsletter_id
                        """, (
                            master_user_id, sender_name, email_date, subject,
                            html_s3_link, full_screenshot_url, top_screenshot_url,
                            0, 0, datetime.now(timezone.utc), processed_data['analysis'].get('summary'),
                            products_link
                        ))
                        
                        newsletter_id = cur.fetchone()[0]

                        # Process tags
                        tags = processed_data['analysis'].get('tags', [])
                        if tags:
                            tag_ids = self.get_or_create_tags(tags, conn, cur)
                            for tag_id in tag_ids:
                                cur.execute("""
                                    INSERT INTO "NewsletterTag" (newsletter_id, tag_id)
                                    VALUES (%s, %s)
                                """, (newsletter_id, tag_id))

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

                        conn.commit()
                        logger.info(f"Successfully processed newsletter: {subject} (ID: {newsletter_id})")

                    except Exception as e:
                        conn.rollback()
                        logger.error(f"Error processing newsletter {subject}: {e}")
                        traceback.print_exc()
                        raise
        except Exception as e:
            logger.error(f"Error in process_and_save_email: {e}")
            traceback.print_exc()
            raise

    async def main(self):
        while True:
            mail = self.connect_to_imap()
            if not mail:
                logger.error("Failed to connect to IMAP server. Retrying in 60 seconds...")
                await asyncio.sleep(60)
                continue

            try:
                mail.select('INBOX')
                _, msg_ids = mail.search(None, 'UNSEEN')
                
                if msg_ids and msg_ids[0]:
                    for msg_id in msg_ids[0].split():
                        try:
                            _, msg_data = mail.fetch(msg_id, '(RFC822)')
                            email_msg = email.message_from_bytes(msg_data[0][1])
                            
                            await self.process_and_save_email(email_msg)
                            
                            # Mark email as read only after successful processing
                            mail.store(msg_id, '+FLAGS', '\\Seen')
                            
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

            logger.info("Waiting 60 seconds before next check...")
            await asyncio.sleep(60)

if __name__ == "__main__":
    processor = NewsletterProcessor()
    asyncio.run(processor.main()) 