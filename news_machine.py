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
import anthropic
from anthropic import Anthropic
import aiohttp
from pprint import pformat

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
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API')

class NewsletterProcessor:
    def __init__(self):
        # Initialize Gemini
        genai.configure(api_key=BARD_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize Google AI Studio API (Gemini 2.0)
        self.GOOGLE_AI_STUDIO_API = os.getenv('GOOGLE_AI_STUDIO_API')
        
        # Initialize DeepSeek API
        self.DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API')
        self.DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
        
        # Initialize Anthropic
        anthropic_api_key = os.getenv('ANTHROPIC_API')
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        
        # Initialize Together AI settings
        self.TOGETHER_API_URL = "https://api.together.xyz/inference"
        self.TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        
        # Initialize Hugging Face settings
        self.HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        
        # Rate limiting setup
        self.last_gemini_call = 0
        self.last_anthropic_call = 0
        self.last_together_call = 0
        self.last_huggingface_call = 0
        self.last_gemini2_call = 0
        self.last_deepseek_call = 0
        self.gemini_calls = 0
        self.anthropic_calls = 0
        self.together_calls = 0
        self.huggingface_calls = 0
        self.gemini2_calls = 0
        self.deepseek_calls = 0
        self.reset_time = time.time()
        
        # Rate limits
        self.GEMINI_CALLS_PER_MINUTE = 50
        self.ANTHROPIC_CALLS_PER_MINUTE = 15
        self.TOGETHER_CALLS_PER_MINUTE = 50
        self.HUGGINGFACE_CALLS_PER_MINUTE = 30
        self.DEEPSEEK_CALLS_PER_MINUTE = 20
        self.MIN_DELAY_BETWEEN_CALLS = 1
        
        self.system_prompt = """You are an AI trained to analyze newsletters and create engaging, SEO-friendly summaries.

Your task is to:
1. Create a comprehensive summary (250-300 words) that:
   - Captures the main message and key points
   - Maintains the original sender's tone and style
   - Includes relevant keywords for SEO
   - Provides valuable insights for readers
2. Extract key information:
   - Important keywords and phrases (for SEO)
   - Products, services, or tools mentioned
   - Key technologies or concepts discussed
3. Format tags in PascalCase (e.g., "ArtificialIntelligence", "ProductUpdate")

Please format your response in the following structure:
Summary: [A detailed 250-300 word summary incorporating key terms and maintaining the sender's tone]
Keywords: [Comma-separated list of important terms and phrases for SEO]
Tags: [Comma-separated list of PascalCase tags]
Products: [Comma-separated list of products/services mentioned]
Key Insights: [Bullet points of key takeaways]

Focus on creating content that is both informative for readers and optimized for search engines."""

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
        """Get existing tags or create new ones, handling compound tags properly."""
        tag_ids = []
        for tag_name in tags:
            tag_name = tag_name.strip()
            if not tag_name:
                continue

            # Convert to PascalCase if not already
            if not tag_name[0].isupper():
                tag_name = ''.join(word.capitalize() for word in tag_name.split())

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

    async def _wait_for_rate_limit(self, api_type: str) -> None:
        """Wait if necessary to respect rate limits."""
        current_time = time.time()
        
        if current_time - self.reset_time >= 60:
            self.gemini_calls = 0
            self.anthropic_calls = 0
            self.together_calls = 0
            self.huggingface_calls = 0
            self.gemini2_calls = 0
            self.deepseek_calls = 0
            self.reset_time = current_time
        
        if api_type == 'gemini':
            time_since_last_call = current_time - self.last_gemini_call
            if time_since_last_call < self.MIN_DELAY_BETWEEN_CALLS:
                await asyncio.sleep(self.MIN_DELAY_BETWEEN_CALLS - time_since_last_call)
            
            if self.gemini_calls >= self.GEMINI_CALLS_PER_MINUTE:
                wait_time = 60 - (current_time - self.reset_time)
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.2f}s for Gemini rate limit reset")
                    await asyncio.sleep(wait_time)
                self.gemini_calls = 0
                self.reset_time = time.time()
            
            self.last_gemini_call = time.time()
            self.gemini_calls += 1
            
        elif api_type == 'gemini2':  # Add new rate limiting for Gemini 2.0
            # Use the same rate limits as Gemini for now
            # Check if we need to reset the counter
            if time.time() - self.reset_time > 60:
                self.gemini2_calls = 0
                self.reset_time = time.time()
                
            # Check if we've exceeded our rate limit
            if self.gemini2_calls >= self.GEMINI_CALLS_PER_MINUTE:
                wait_time = 60 - (time.time() - self.reset_time) + 1
                if wait_time > 0:
                    logger.info(f"Rate limit reached for Gemini 2.0. Waiting {wait_time:.2f} seconds.")
                    await asyncio.sleep(wait_time)
                self.gemini2_calls = 0
                self.reset_time = time.time()
                
            # Check if we need to wait between calls
            time_since_last_call = time.time() - self.last_gemini2_call
            if time_since_last_call < self.MIN_DELAY_BETWEEN_CALLS:
                await asyncio.sleep(self.MIN_DELAY_BETWEEN_CALLS - time_since_last_call)
                
            self.last_gemini2_call = time.time()
            self.gemini2_calls += 1
            
        elif api_type == 'anthropic':
            time_since_last_call = current_time - self.last_anthropic_call
            if time_since_last_call < self.MIN_DELAY_BETWEEN_CALLS:
                await asyncio.sleep(self.MIN_DELAY_BETWEEN_CALLS - time_since_last_call)
            
            if self.anthropic_calls >= self.ANTHROPIC_CALLS_PER_MINUTE:
                wait_time = 60 - (current_time - self.reset_time)
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.2f}s for Anthropic rate limit reset")
                    await asyncio.sleep(wait_time)
                self.anthropic_calls = 0
                self.reset_time = time.time()
            
            self.last_anthropic_call = time.time()
            self.anthropic_calls += 1
            
        elif api_type == 'together':
            time_since_last_call = current_time - self.last_together_call
            if time_since_last_call < self.MIN_DELAY_BETWEEN_CALLS:
                await asyncio.sleep(self.MIN_DELAY_BETWEEN_CALLS - time_since_last_call)
            
            if self.together_calls >= self.TOGETHER_CALLS_PER_MINUTE:
                wait_time = 60 - (current_time - self.reset_time)
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.2f}s for Together AI rate limit reset")
                    await asyncio.sleep(wait_time)
                self.together_calls = 0
                self.reset_time = time.time()
            
            self.last_together_call = time.time()
            self.together_calls += 1
        
        elif api_type == 'huggingface':
            time_since_last_call = current_time - self.last_huggingface_call
            if time_since_last_call < self.MIN_DELAY_BETWEEN_CALLS:
                await asyncio.sleep(self.MIN_DELAY_BETWEEN_CALLS - time_since_last_call)
            
            if self.huggingface_calls >= self.HUGGINGFACE_CALLS_PER_MINUTE:
                wait_time = 60 - (current_time - self.reset_time)
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.2f}s for Hugging Face rate limit reset")
                    await asyncio.sleep(wait_time)
                self.huggingface_calls = 0
                self.reset_time = time.time()
            
            self.last_huggingface_call = time.time()
            self.huggingface_calls += 1

        elif api_type == 'deepseek':
            time_since_last_call = current_time - self.last_deepseek_call
            if time_since_last_call < self.MIN_DELAY_BETWEEN_CALLS:
                await asyncio.sleep(self.MIN_DELAY_BETWEEN_CALLS - time_since_last_call)
            
            if self.deepseek_calls >= self.DEEPSEEK_CALLS_PER_MINUTE:
                wait_time = 60 - (current_time - self.reset_time)
                if wait_time > 0:
                    logger.info(f"Waiting {wait_time:.2f}s for DeepSeek rate limit reset")
                    await asyncio.sleep(wait_time)
                self.deepseek_calls = 0
                self.reset_time = time.time()
            
            self.last_deepseek_call = time.time()
            self.deepseek_calls += 1

    async def _retry_with_backoff(self, func, *args, max_retries=3, initial_delay=1):
        """Retry a function with exponential backoff."""
        delay = initial_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await func(*args)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Check if it's a rate limit error
                    if '429' in str(e):
                        wait_time = delay * (2 ** attempt)
                        logger.info(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}")
                        await asyncio.sleep(wait_time)
                    else:
                        # For other errors, use shorter delays
                        wait_time = delay
                        logger.warning(f"Error occurred, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {str(e)}")
                    raise last_exception

    async def _generate_content_with_gemini(self, text_content: str, subject: str) -> Dict:
        """Generate content using Gemini API with rate limiting."""
        try:
            await self._wait_for_rate_limit('gemini')
            response = await self.model.generate_content_async(
                f"Subject: {subject}\n\nContent: {text_content}\n\n{self.system_prompt}"
            )
            
            # Parse the response
            lines = response.text.split('\n')
            result = {}
            current_key = None
            
            for line in lines:
                if line.startswith('Summary:'):
                    current_key = 'summary'
                    result[current_key] = line.replace('Summary:', '').strip()
                elif line.startswith('Keywords:'):
                    current_key = 'keywords'
                    keywords_text = line.replace('Keywords:', '').strip()
                    result[current_key] = [kw.strip() for kw in keywords_text.split(',')]
                elif line.startswith('Tags:'):
                    current_key = 'tags'
                    tags_text = line.replace('Tags:', '').strip()
                    result[current_key] = [tag.strip() for tag in tags_text.split(',')]
                elif line.startswith('Products:'):
                    current_key = 'products'
                    products_text = line.replace('Products:', '').strip()
                    result[current_key] = [prod.strip() for prod in products_text.split(',')]
                elif line.startswith('Key Insights:'):
                    current_key = 'insights'
                    result[current_key] = []
                elif current_key == 'insights' and line.strip().startswith('-'):
                    result[current_key].append(line.strip()[2:])
                elif current_key and line.strip():
                    if isinstance(result[current_key], list):
                        result[current_key].append(line.strip())
                    else:
                        result[current_key] += ' ' + line.strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None

    async def _generate_content_with_anthropic(self, text_content: str, subject: str) -> Dict:
        """Generate content using Anthropic API with rate limiting."""
        try:
            await self._wait_for_rate_limit('anthropic')
            message = await self.anthropic.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1500,  # Increased for longer summaries
                messages=[{
                    "role": "user",
                    "content": f"Subject: {subject}\n\nContent: {text_content}\n\n{self.system_prompt}"
                }]
            )
            
            # Parse the response
            lines = message.content[0].text.split('\n')
            result = {}
            current_key = None
            
            for line in lines:
                if line.startswith('Summary:'):
                    current_key = 'summary'
                    result[current_key] = line.replace('Summary:', '').strip()
                elif line.startswith('Keywords:'):
                    current_key = 'keywords'
                    keywords_text = line.replace('Keywords:', '').strip()
                    result[current_key] = [kw.strip() for kw in keywords_text.split(',')]
                elif line.startswith('Tags:'):
                    current_key = 'tags'
                    tags_text = line.replace('Tags:', '').strip()
                    result[current_key] = [tag.strip() for tag in tags_text.split(',')]
                elif line.startswith('Products:'):
                    current_key = 'products'
                    products_text = line.replace('Products:', '').strip()
                    result[current_key] = [prod.strip() for prod in products_text.split(',')]
                elif line.startswith('Key Insights:'):
                    current_key = 'insights'
                    result[current_key] = []
                elif current_key == 'insights' and line.strip().startswith('-'):
                    result[current_key].append(line.strip()[2:])
                elif current_key and line.strip():
                    if isinstance(result[current_key], list):
                        result[current_key].append(line.strip())
                    else:
                        result[current_key] += ' ' + line.strip()
            
            return result
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None

    async def _generate_content_with_together(self, text_content: str, subject: str) -> Dict:
        """Generate content using Together AI API with rate limiting."""
        try:
            await self._wait_for_rate_limit('together')
            headers = {
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Format prompt for Mixtral model
            prompt = f"""<s>[INST] Here's a newsletter to analyze:

Subject: {subject}

Content: {text_content}

{self.system_prompt}[/INST]</s>"""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.TOGETHER_API_URL,
                    headers=headers,
                    json={
                        "model": self.TOGETHER_MODEL,
                        "prompt": prompt,
                        "max_tokens": 1500,  # Increased for longer summaries
                        "temperature": 0.7,
                        "top_p": 0.7,
                        "top_k": 50,
                        "repetition_penalty": 1.1
                    }
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Together AI API error: {response.status}")
                    result = await response.json()
                    return self._parse_llm_response(result['output']['choices'][0]['text'])
            
        except Exception as e:
            logger.error(f"Together AI API error: {e}")
            return None

    async def _generate_content_with_huggingface(self, text_content: str, subject: str) -> Dict:
        """Generate content using Hugging Face API with rate limiting."""
        try:
            await self._wait_for_rate_limit('huggingface')
            headers = {
                "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Format prompt for Mixtral model
            prompt = f"""<s>[INST] Here's a newsletter to analyze:

Subject: {subject}

Content: {text_content}

{self.system_prompt}[/INST]</s>"""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.HUGGINGFACE_API_URL,
                    headers=headers,
                    json={
                        "inputs": prompt,
                        "parameters": {
                            "max_new_tokens": 1500,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "return_full_text": False
                        }
                    }
                ) as response:
                    response_text = await response.text()
                    if response.status != 200:
                        logger.error(f"Hugging Face API error status {response.status}: {response_text}")
                        raise Exception(f"Hugging Face API error: {response.status} - {response_text}")
                    
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get('generated_text', '')
                        if generated_text:
                            return self._parse_llm_response(generated_text)
                        else:
                            logger.error("Hugging Face API returned empty response")
                            return None
                    else:
                        logger.error(f"Unexpected Hugging Face API response format: {result}")
                        return None
            
        except Exception as e:
            logger.error(f"Hugging Face API error: {str(e)}")
            return None

    def _parse_llm_response(self, text: str) -> Dict:
        """Parse LLM response into structured format."""
        lines = text.split('\n')
        result = {}
        current_key = None
        
        for line in lines:
            if line.startswith('Summary:'):
                current_key = 'summary'
                result[current_key] = line.replace('Summary:', '').strip()
            elif line.startswith('Keywords:'):
                current_key = 'keywords'
                keywords_text = line.replace('Keywords:', '').strip()
                result[current_key] = [kw.strip() for kw in keywords_text.split(',')]
            elif line.startswith('Tags:'):
                current_key = 'tags'
                tags_text = line.replace('Tags:', '').strip()
                # Split by comma and handle nested commas in tags
                raw_tags = [tag.strip() for tag in tags_text.split(',')]
                processed_tags = []
                for tag in raw_tags:
                    # If tag contains multiple words with first letters capitalized, split it
                    if ' ' in tag and all(word[0].isupper() for word in tag.split()):
                        processed_tags.extend([t.strip() for t in tag.split()])
                    else:
                        processed_tags.append(tag)
                # Remove any empty tags and convert to PascalCase
                result[current_key] = [
                    ''.join(word.capitalize() for word in tag.split())
                    for tag in processed_tags
                    if tag.strip()
                ]
            elif line.startswith('Products:'):
                current_key = 'products'
                products_text = line.replace('Products:', '').strip()
                result[current_key] = [prod.strip() for prod in products_text.split(',')]
            elif line.startswith('Key Insights:'):
                current_key = 'insights'
                result[current_key] = []
            elif current_key == 'insights' and line.strip():
                # Clean the insight line: remove asterisks, dashes, and leading/trailing whitespace
                clean_insight = line.strip()
                clean_insight = re.sub(r'^\s*[\*\-]\s*', '', clean_insight)  # Remove leading * or - and whitespace
                if clean_insight:  # Only add non-empty insights
                    result[current_key].append(clean_insight)
            elif current_key and line.strip():
                if isinstance(result[current_key], list):
                    result[current_key].append(line.strip())
                else:
                    result[current_key] += ' ' + line.strip()
        
        return result

    def _local_fallback_analysis(self, text_content: str, subject: str) -> Dict:
        """Perform basic text analysis when all LLM APIs fail."""
        try:
            # Basic summary: Use subject and first paragraph
            soup = BeautifulSoup(text_content, 'html.parser')
            first_paragraph = soup.find('p')
            summary = f"{subject} - {first_paragraph.get_text()[:200]}..." if first_paragraph else subject
            
            # Basic tag extraction from subject and first paragraph
            text_for_tags = f"{subject} {first_paragraph.get_text() if first_paragraph else ''}"
            words = re.findall(r'\b\w+\b', text_for_tags)
            common_tech_terms = {
                'AI', 'Machine Learning', 'Data', 'Cloud', 'Security',
                'Marketing', 'Business', 'Technology', 'Software', 'Development',
                'Web', 'Mobile', 'Analytics', 'Digital', 'Innovation'
            }
            
            # Extract potential tags from text
            tags = []
            for term in common_tech_terms:
                if term.lower() in text_for_tags.lower():
                    tags.append(term.replace(' ', ''))
            
            # Extract potential product mentions (look for capitalized terms)
            products = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text_for_tags)
            products = [p for p in products if len(p) > 2 and p not in {'I', 'A', 'The'}]
            
            # Extract key sentences as insights
            sentences = re.split(r'[.!?]+', text_content)
            insights = [s.strip() for s in sentences[:3] if len(s.strip()) > 20][:3]
            
            return {
                'summary': summary,
                'tags': tags[:5],  # Limit to 5 tags
                'products': products[:3],  # Limit to 3 products
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Error in local fallback analysis: {e}")
            return {
                'summary': f"Failed to generate summary for: {subject}",
                'tags': [],
                'products': [],
                'insights': []
            }

    async def _generate_content_with_gemini2(self, text_content: str, subject: str) -> Dict:
        """Generate content analysis using Google AI Studio API with Gemini 2.0 model."""
        try:
            # Initialize rate limiting tracker if not exists
            if not hasattr(self, 'gemini2_calls'):
                self.gemini2_calls = 0
                self.last_gemini2_call = 0
            
            await self._wait_for_rate_limit("gemini2")
            
            # Import here to handle potential import errors gracefully
            try:
                import google.generativeai as genai_v2
                
                # Configure the SDK with the API key
                genai_v2.configure(api_key=self.GOOGLE_AI_STUDIO_API)
                
                # Create a client - note the correct initialization pattern
                genai_client = genai_v2.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config={"temperature": 0.7, "max_output_tokens": 1500}
                )
            except ImportError:
                logger.error("Failed to import Google GenAI SDK. Make sure it's installed correctly.")
                return {}
            
            # Prepare prompt
            prompt = f"""
            Subject: {subject}
            
            {self.system_prompt}
            
            Content to analyze:
            {text_content}
            """
            
            # The SDK doesn't have built-in async support, so we need to run it in an executor
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: genai_client.generate_content(prompt)
            )
            
            # Extract text from response
            analysis_text = response.text
            
            # Parse the response
            analysis = self._parse_llm_response(analysis_text)
            self._log_llm_output("Gemini 2.0", analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating content with Gemini 2.0: {e}")
            logger.error(traceback.format_exc())
            return {}

    async def _generate_content_with_deepseek(self, text_content: str, subject: str) -> Dict:
        """Generate content using DeepSeek API with rate limiting."""
        try:
            await self._wait_for_rate_limit('deepseek')
            headers = {
                "Authorization": f"Bearer {self.DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Format prompt for DeepSeek model
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": f"Subject: {subject}\n\nContent: {text_content}"
                }
            ]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.DEEPSEEK_API_URL,
                    headers=headers,
                    json={
                        "model": "deepseek-chat",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 1500
                    }
                ) as response:
                    response_text = await response.text()
                    if response.status != 200:
                        logger.error(f"DeepSeek API error status {response.status}: {response_text}")
                        raise Exception(f"DeepSeek API error: {response.status} - {response_text}")
                    
                    result = await response.json()
                    generated_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    if generated_text:
                        analysis = self._parse_llm_response(generated_text)
                        self._log_llm_output("DeepSeek", analysis)
                        return analysis
                    else:
                        logger.error("DeepSeek API returned empty response")
                        return None
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def process_email(self, email_content: str, subject: str) -> Dict:
        """Process email content to extract information using LLMs."""
        logger.info(f"Processing email: {subject}")
        
        # Extract text content from HTML
        soup = BeautifulSoup(email_content, 'html.parser')
        text_content = self._extract_text_with_structure(soup)
        
        analysis = {}
        try:
            # Try Google AI Studio API (Gemini 2.0) first
            analysis = await self._generate_content_with_gemini2(text_content, subject)
            if analysis:
                return analysis
            
            # Try DeepSeek second
            analysis = await self._generate_content_with_deepseek(text_content, subject)
            if analysis:
                return analysis
                
            # Try Hugging Face next
            analysis = await self._generate_content_with_huggingface(text_content, subject)
            if analysis:
                return analysis
                
            # Try Anthropic Claude next
            analysis = await self._generate_content_with_anthropic(text_content, subject)
            if analysis:
                return analysis
                
            # Try Gemini Pro next
            analysis = await self._generate_content_with_gemini(text_content, subject)
            if analysis:
                return analysis
                
            # Try Together AI next
            analysis = await self._generate_content_with_together(text_content, subject)
            if analysis:
                return analysis
                
            # Fallback to local analysis
            analysis = self._local_fallback_analysis(text_content, subject)
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing email: {e}")
            logger.error(traceback.format_exc())
            return {
                "summary": f"Error processing email: {str(e)}",
                "keywords": [],
                "tags": [],
                "products": [],
                "key_insights": []
            }

    def _extract_text_with_structure(self, soup: BeautifulSoup) -> str:
        for element in soup(['script', 'style']):
            element.decompose()
        
        preserved_tags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol', 'li', 'blockquote']
        for tag in preserved_tags:
            for element in soup.find_all(tag):
                element.name = tag
        
        return str(soup)

    def decode_email_subject(self, email_msg) -> str:
        """Decode email subject properly handling emojis and special characters."""
        try:
            # Get raw subject
            subject_header = email_msg['Subject']
            if not subject_header:
                return "Untitled Newsletter"

            # If it's already a string and doesn't contain encoded parts, just clean it
            if isinstance(subject_header, str) and not '=?' in subject_header:
                return self.clean_subject(subject_header)

            # Decode header parts
            decoded_parts = []
            parts = decode_header(subject_header)
            
            for part, charset in parts:
                if isinstance(part, bytes):
                    try:
                        # Try with provided charset first
                        if charset:
                            decoded_parts.append(part.decode(charset))
                        else:
                            # Try UTF-8 first for emoji support
                            try:
                                decoded_parts.append(part.decode('utf-8'))
                            except UnicodeDecodeError:
                                # Fallback to other encodings
                                try:
                                    decoded_parts.append(part.decode('latin1'))
                                except UnicodeDecodeError:
                                    decoded_parts.append(part.decode('ascii', errors='replace'))
                    except Exception as e:
                        logger.warning(f"Error decoding subject part: {e}")
                        # Last resort fallback
                        decoded_parts.append(part.decode('ascii', errors='replace'))
                else:
                    decoded_parts.append(str(part))

            # Join parts and clean
            subject = ''.join(decoded_parts)
            
            # Handle special cases of Q-encoded text that wasn't properly decoded
            subject = re.sub(r'=\?utf-8\?[Qq]\?(.*?)\?=', r'\1', subject)
            subject = subject.replace('=20', ' ')  # Fix common Q-encoding space
            subject = subject.replace('&=', '&')   # Fix common Q-encoding ampersand
            
            return self.clean_subject(subject)
            
        except Exception as e:
            logger.error(f"Error decoding subject: {e}")
            return "Untitled Newsletter"

    def clean_subject(self, subject: str) -> str:
        """Clean the subject while preserving emojis and special characters."""
        if not subject or not subject.strip():
            return "Untitled Newsletter"
        
        # Remove any null bytes or control characters except newlines
        subject = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', subject)
        # Replace multiple spaces with single space
        subject = re.sub(r'\s+', ' ', subject)
        # Clean up any remaining Q-encoded artifacts
        subject = subject.replace('=?UTF-8?Q?', '')
        subject = subject.replace('?=', '')
        # Strip leading/trailing whitespace
        subject = subject.strip()
        
        return subject if subject else "Untitled Newsletter"

    async def process_and_save_email(self, email_msg) -> None:
        try:
            # Use the new subject decoding method
            subject = self.decode_email_subject(email_msg)
            logger.info(f"Processing email with subject: {subject}")
            
            # Extract sender information properly
            from_header = decode_header(email_msg['From'])[0][0]
            if isinstance(from_header, bytes):
                from_header = from_header.decode()
            
            # Parse sender email and name
            if '<' in from_header and '>' in from_header:
                sender_name = from_header.split('<')[0].strip().replace('"', '')
                sender_email = from_header.split('<')[1].split('>')[0].strip()
            else:
                sender_name = None
                sender_email = from_header.strip()

            html_content = None
            for part in email_msg.walk():
                if part.get_content_type() == "text/html":
                    html_content = part.get_payload(decode=True).decode(part.get_content_charset())
                    break

            if not html_content:
                logger.warning(f"No HTML content found in email: {subject}")
                return

            uuid_val = str(uuid.uuid4())
            analysis = await self.process_email(html_content, subject)
            
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

                        # Extract and create/get brand
                        brand_info = self.extract_brand_info(sender_email, sender_name)
                        brand_id = self.get_or_create_brand(cur, brand_info)
                        logger.info(f"Using brand: {brand_info['name']} (ID: {brand_id})")

                        # Get products link safely - directly from analysis now
                        products = analysis.get('products', [])
                        products_link = products[0] if products else None
                        
                        # Process key insights - join them with commas for database storage
                        key_insights = analysis.get('insights', [])
                        key_insights_string = ", ".join(key_insights) if key_insights else None
                        
                        logger.info(f"Extracted key insights: {key_insights_string}")

                        # Insert newsletter with brand_id and key_insights
                        cur.execute("""
                            INSERT INTO "Newsletter" (
                                user_id, sender, published_at, subject, html_file_url,
                                full_screenshot_url, top_screenshot_url,
                                likes_count, you_rocks_count, created_at,
                                summary, products_link, brand_id, key_insights
                            ) 
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            RETURNING newsletter_id
                        """, (
                            master_user_id, sender_name, email_date, subject,
                            html_s3_link, full_screenshot_url, top_screenshot_url,
                            0, 0, datetime.now(timezone.utc), analysis.get('summary'),
                            products_link, brand_id, key_insights_string
                        ))
                        
                        newsletter_id = cur.fetchone()[0]

                        # Process tags - directly from analysis now
                        tags = analysis.get('tags', [])
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

    def create_brand_slug(self, text: str) -> str:
        """Create a URL-friendly slug from text."""
        # Convert to lowercase and replace spaces/special chars with hyphens
        slug = text.lower()
        slug = re.sub(r'[^a-z0-9]+', '-', slug)
        slug = slug.strip('-')
        return slug

    def get_unique_brand_slug(self, cur, base_slug: str, domain: str = None) -> str:
        """Generate a unique slug for a brand, considering domain if available."""
        slug = base_slug
        counter = 1
        
        while True:
            # Check if this slug is already used
            cur.execute(
                'SELECT domain FROM "Brand" WHERE slug = %s',
                (slug,)
            )
            result = cur.fetchone()
            
            if not result:
                # Slug is unique, we can use it
                return slug
                
            existing_domain = result[0]
            if existing_domain == domain:
                # Same domain means it's the same brand
                return slug
                
            # Add domain-based suffix if available
            if counter == 1 and domain:
                # Extract first part of domain (e.g., 'india' from 'company.india.com')
                domain_parts = domain.split('.')
                if len(domain_parts) > 2:
                    location_hint = domain_parts[-3]  # Get the subdomain
                    slug = f"{base_slug}-{location_hint}"
                    counter += 1
                    continue
                    
            # If still not unique or no domain available, add number
            slug = f"{base_slug}-{counter}"
            counter += 1

    def extract_brand_info(self, sender_email: str, sender_name: str) -> Dict[str, str]:
        """Extract brand information from the email sender."""
        # Extract domain from email
        domain = sender_email.split('@')[1] if '@' in sender_email else None
        
        # Create brand name from sender name or email
        name = sender_name or sender_email.split('@')[0]
        
        # Create initial slug from the name
        base_slug = self.create_brand_slug(name)
        
        return {
            'name': name,
            'slug': base_slug,
            'domain': domain,
            'email': sender_email
        }

    def get_or_create_brand(self, cur, brand_info: Dict[str, str]) -> str:
        """Get existing brand or create a new one with unique slug."""
        try:
            # First try to find by domain (most specific identifier)
            if brand_info['domain']:
                cur.execute(
                    'SELECT brand_id FROM "Brand" WHERE domain = %s',
                    (brand_info['domain'],)
                )
                result = cur.fetchone()
                if result:
                    return result[0]
            
            # Then try to find by email pattern
            email_domain = brand_info['email'].split('@')[1]
            cur.execute(
                'SELECT brand_id FROM "Brand" WHERE domain LIKE %s',
                (f'%.{email_domain}',)
            )
            result = cur.fetchone()
            if result:
                return result[0]
            
            # Generate unique slug
            unique_slug = self.get_unique_brand_slug(cur, brand_info['slug'], brand_info['domain'])
            
            # Create new brand with unique slug
            cur.execute(
                '''INSERT INTO "Brand" (
                    brand_id, name, slug, domain, 
                    is_verified, is_claimed, 
                    created_at, updated_at
                ) VALUES (
                    gen_random_uuid(), %s, %s, %s, 
                    false, false, 
                    NOW(), NOW()
                )
                RETURNING brand_id''',
                (
                    brand_info['name'],
                    unique_slug,
                    brand_info['domain']
                )
            )
            
            brand_id = cur.fetchone()[0]
            
            # Create social links entry for the brand
            cur.execute(
                '''INSERT INTO "SocialLinks" (
                    id, brand_id
                ) VALUES (
                    gen_random_uuid(), %s
                )''',
                (brand_id,)
            )
            
            logger.info(f"Created new brand: {brand_info['name']} (ID: {brand_id}, slug: {unique_slug})")
            return brand_id
            
        except Exception as e:
            logger.error(f"Error in get_or_create_brand: {e}")
            raise

    def _log_llm_output(self, llm_name: str, analysis: Dict) -> None:
        """Log the LLM output in a readable format."""
        logger.info(f"\n{'='*50}\n{llm_name} Output:\n{'='*50}")
        logger.info(f"Summary: {analysis.get('summary', 'N/A')}")
        logger.info(f"Keywords: {', '.join(analysis.get('keywords', []))}")
        logger.info(f"Tags: {', '.join(analysis.get('tags', []))}")
        logger.info(f"Products: {', '.join(analysis.get('products', []))}")
        logger.info("Key Insights:")
        for insight in analysis.get('insights', []):
            logger.info(f"  - {insight}")
        logger.info('='*50)

if __name__ == "__main__":
    processor = NewsletterProcessor()
    asyncio.run(processor.main()) 