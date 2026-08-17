import os
import sys
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

import json
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from PIL import Image
from dotenv import load_dotenv
import time
import psycopg2
from psycopg2.extras import DictCursor
from urllib.parse import urlparse, unquote
from dateutil import parser as dateutil_parser
from typing import Dict, List, Optional
import aiohttp
from pprint import pformat
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Logging setup
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Supabase Storage Configuration
SUPABASE_URL = os.getenv('SUPABASE_URL') or os.getenv('NEXT_PUBLIC_SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY') or os.getenv('SUPABASE_KEY') or os.getenv('NEXT_PUBLIC_SUPABASE_ANON_KEY')
SUPABASE_BUCKET = os.getenv('SUPABASE_STORAGE_BUCKET') or 'newsletters'

supabase_client: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info(f"Initialized Supabase client targeting bucket: '{SUPABASE_BUCKET}'")
    except Exception as _sb_err:
        logger.error(f"Error initializing Supabase client: {_sb_err}")

# Environment variables
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')
IMAP_SERVER = os.getenv('EMAIL_IMAP_SERVER') or os.getenv('IMAP_SERVER') or 'imap.buzondecorreo.com'
IMAP_PORT = int(os.getenv('EMAIL_IMAP_PORT') or os.getenv('IMAP_PORT') or 993)
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

class NewsletterProcessor:
    def __init__(self):
        # Initialize OpenRouter API & Free Models Fallback List
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
        
        default_free_models = [
            "google/gemini-2.0-flash-exp:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "deepseek/deepseek-r1:free",
            "qwen/qwen-2.5-coder-32b-instruct:free",
            "mistralai/mistral-small-24b-instruct-2501:free",
            "google/gemma-2-9b-it:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "google/gemini-2.0-flash-001"
        ]
        
        env_models = os.getenv('OPENROUTER_MODELS') or os.getenv('OPENROUTER_MODEL')
        if env_models:
            self.OPENROUTER_MODELS = [m.strip() for m in env_models.split(',') if m.strip()]
        else:
            self.OPENROUTER_MODELS = default_free_models
        
        # Rate limiting setup
        self.last_openrouter_call = 0
        self.openrouter_calls = 0
        self.reset_time = time.time()
        
        # Rate limits
        self.OPENROUTER_CALLS_PER_MINUTE = 50
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
        db_url = os.getenv('DIRECT_DATABASE_URL') or os.getenv('DATABASE_URL') or os.getenv('DATABASE_URL_POOLED')
        if not db_url:
            raise ValueError("Neither DIRECT_DATABASE_URL nor DATABASE_URL is set in environment variables")
        
        # Strip literal quotes, escaped backslashes, and whitespace
        clean_url = db_url.strip('"\'' + '\\').strip()
        
        # Remove query parameters like ?schema=public
        if '?' in clean_url:
            clean_url = clean_url.split('?')[0]
            
        parsed = urlparse(clean_url)
        dbname = parsed.path.lstrip('/') if parsed.path else 'postgres'
        return {
            'dbname': dbname,
            'user': unquote(parsed.username) if parsed.username else '',
            'password': unquote(parsed.password) if parsed.password else '',
            'host': parsed.hostname or 'supabasenewsletter.oncewerehumans.com',
            'port': parsed.port or 5432
        }

    def get_db_connection(self):
        db_config = self.get_db_config()
        
        # List of potential fallback configurations if connection is refused
        configs_to_try = [db_config]
        
        # If port is 5432, also try Supabase pooler port 6543
        if db_config.get('port') == 5432:
            cfg6543 = db_config.copy()
            cfg6543['port'] = 6543
            configs_to_try.append(cfg6543)

        # If running in Docker on same server, also try host.docker.internal / 172.17.0.1
        for host in ['host.docker.internal', '172.17.0.1']:
            if db_config.get('host') != host:
                cfg_host = db_config.copy()
                cfg_host['host'] = host
                configs_to_try.append(cfg_host)

        last_error = None
        for cfg in configs_to_try:
            try:
                conn = psycopg2.connect(**cfg)
                conn.autocommit = False
                logger.info(f"Successfully connected to PostgreSQL at {cfg['host']}:{cfg['port']}")
                return conn
            except Exception as e:
                last_error = e
                logger.warning(f"Could not connect to PostgreSQL at {cfg['host']}:{cfg['port']} ({e}). Trying next configuration...")

        logger.error(f"Error connecting to database after trying all fallback configurations: {last_error}")
        raise last_error

    def connect_to_imap(self, retry_count=5):
        for attempt in range(retry_count):
            try:
                logger.info(f"Connecting to IMAP server {IMAP_SERVER}:{IMAP_PORT} as {EMAIL_USER}")
                mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
                mail.login(EMAIL_USER, EMAIL_PASS)
                return mail
            except imaplib.IMAP4.abort as e:
                logger.error(f"IMAP connection error: {e}")
                if attempt < retry_count - 1:
                    time.sleep(5)
                else:
                    return None
            except Exception as e:
                logger.error(f"IMAP login/connect error: {e}")
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

    async def upload_to_supabase_storage(self, file_path: str, uuid_val: str, content_type: str = "image/webp") -> str:
        """Upload a file to Supabase Storage bucket and return its public URL."""
        filename = os.path.basename(file_path)
        storage_path = f"{uuid_val}/{filename}"

        if not supabase_client:
            logger.error("Supabase client is not initialized. Please verify SUPABASE_URL and SUPABASE_KEY.")
            return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_path}"

        try:
            with open(file_path, 'rb') as f:
                file_bytes = f.read()

            try:
                supabase_client.storage.from_(SUPABASE_BUCKET).upload(
                    file=file_bytes,
                    path=storage_path,
                    file_options={"content-type": content_type, "upsert": "true"}
                )
            except Exception as upload_err:
                if "Bucket not found" in str(upload_err):
                    logger.info(f"Bucket '{SUPABASE_BUCKET}' not found in Supabase. Attempting auto-creation...")
                    try:
                        supabase_client.storage.create_bucket(SUPABASE_BUCKET, options={"public": True})
                        supabase_client.storage.from_(SUPABASE_BUCKET).upload(
                            file=file_bytes,
                            path=storage_path,
                            file_options={"content-type": content_type, "upsert": "true"}
                        )
                    except Exception as create_err:
                        logger.error(f"Could not auto-create bucket '{SUPABASE_BUCKET}': {create_err}")
                        raise upload_err
                else:
                    raise upload_err

            public_url = supabase_client.storage.from_(SUPABASE_BUCKET).get_public_url(storage_path)
            logger.info(f"Uploaded to Supabase Storage: {public_url}")

            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted local temporary file: {file_path}")

            return public_url
        except Exception as e:
            logger.error(f"Error uploading {file_path} to Supabase Storage: {e}")
            traceback.print_exc()
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{storage_path}"

    async def take_screenshot(self, html_content: str, uuid_val: str) -> Dict[str, str]:
        """Generate full and thumbnail screenshots, convert to WebP, and upload to Supabase Storage."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
            )
            page = await browser.new_page()
            page.set_default_timeout(60000)

            full_url = ""
            top_url = ""

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

                full_webp_path = full_screenshot_path.replace(".png", ".webp")
                thumb_webp_path = thumb_screenshot_path.replace(".png", ".webp")

                full_url = await self.upload_to_supabase_storage(full_webp_path, uuid_val, content_type="image/webp")
                top_url = await self.upload_to_supabase_storage(thumb_webp_path, uuid_val, content_type="image/webp")

            except Exception as e:
                logger.error(f"Error taking screenshots: {e}")
                traceback.print_exc()
            finally:
                if page:
                    await page.close()
                if browser:
                    await browser.close()

            return {
                'full_screenshot_url': full_url or f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{uuid_val}/{uuid_val}_full.webp",
                'top_screenshot_url': top_url or f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{uuid_val}/{uuid_val}_small.webp"
            }

    async def upload_html_and_take_screenshot(self, html_content: str, uuid_val: str) -> Dict[str, str]:
        """Save HTML, upload to Supabase Storage, and trigger Playwright screenshots."""
        html_file_path = f"{uuid_val}.html"
        with open(html_file_path, 'w', encoding='utf-8') as file:
            file.write(html_content)

        try:
            html_public_url = await self.upload_to_supabase_storage(html_file_path, uuid_val, content_type="text/html")
            screenshot_assets = await self.take_screenshot(html_content, uuid_val)

            return {
                'html_url': html_public_url,
                'full_screenshot_url': screenshot_assets.get('full_screenshot_url', ''),
                'top_screenshot_url': screenshot_assets.get('top_screenshot_url', '')
            }

        except Exception as e:
            logger.error(f"Error uploading HTML or taking screenshots: {e}")
            traceback.print_exc()
            if os.path.exists(html_file_path):
                try:
                    os.remove(html_file_path)
                except Exception:
                    pass
            return {
                'html_url': f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{uuid_val}/{uuid_val}.html",
                'full_screenshot_url': f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{uuid_val}/{uuid_val}_full.webp",
                'top_screenshot_url': f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{uuid_val}/{uuid_val}_small.webp"
            }

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

    async def _wait_for_rate_limit(self, api_type: str = 'openrouter') -> None:
        """Wait if necessary to respect OpenRouter rate limits."""
        current_time = time.time()
        
        if current_time - self.reset_time >= 60:
            self.openrouter_calls = 0
            self.reset_time = current_time
        
        time_since_last_call = current_time - self.last_openrouter_call
        if time_since_last_call < self.MIN_DELAY_BETWEEN_CALLS:
            await asyncio.sleep(self.MIN_DELAY_BETWEEN_CALLS - time_since_last_call)
        
        if self.openrouter_calls >= self.OPENROUTER_CALLS_PER_MINUTE:
            wait_time = 60 - (current_time - self.reset_time)
            if wait_time > 0:
                logger.info(f"Waiting {wait_time:.2f}s for OpenRouter rate limit reset")
                await asyncio.sleep(wait_time)
            self.openrouter_calls = 0
            self.reset_time = time.time()
        
        self.last_openrouter_call = time.time()
        self.openrouter_calls += 1

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

    async def _generate_content_with_openrouter(self, text_content: str, subject: str) -> Optional[Dict]:
        """Generate content by attempting free OpenRouter models sequentially until one succeeds."""
        if not self.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY is not set.")
            return None
            
        headers = {
            "Authorization": f"Bearer {self.OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://newsletterzilla.online",
            "X-Title": "Newsletterzilla Scraper",
            "Content-Type": "application/json"
        }
        
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
        
        for model in self.OPENROUTER_MODELS:
            try:
                await self._wait_for_rate_limit('openrouter')
                logger.info(f"Attempting OpenRouter model: '{model}'")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.OPENROUTER_API_URL,
                        headers=headers,
                        json={
                            "model": model,
                            "messages": messages,
                            "temperature": 0.7,
                            "max_tokens": 1500
                        }
                    ) as response:
                        response_text = await response.text()
                        if response.status != 200:
                            logger.warning(f"OpenRouter model '{model}' status {response.status}: {response_text}. Trying next free model...")
                            continue
                        
                        result = await response.json()
                        generated_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                        
                        if generated_text:
                            analysis = self._parse_llm_response(generated_text)
                            self._log_llm_output(f"OpenRouter ({model})", analysis)
                            return analysis
                        else:
                            logger.warning(f"OpenRouter model '{model}' returned empty response. Trying next free model...")
                            continue
                            
            except Exception as e:
                logger.warning(f"Error with OpenRouter model '{model}': {e}. Trying next free model...")
                continue
                
        logger.error("All configured free OpenRouter models failed.")
        return None

    async def process_email(self, email_content: str, subject: str) -> Dict:
        """Process email content to extract summary and metadata using OpenRouter."""
        logger.info(f"Processing email with OpenRouter: {subject}")
        
        # Extract text content from HTML
        soup = BeautifulSoup(email_content, 'html.parser')
        text_content = self._extract_text_with_structure(soup)
        
        try:
            analysis = await self._generate_content_with_openrouter(text_content, subject)
            if analysis:
                return analysis

            # Fallback to local heuristic analysis if OpenRouter fails
            logger.warning("OpenRouter analysis returned empty, using local fallback analysis")
            return self._local_fallback_analysis(text_content, subject)
            
        except Exception as e:
            logger.error(f"Error processing email: {e}")
            logger.error(traceback.format_exc())
            return {
                "summary": f"Error processing email: {str(e)}",
                "keywords": [],
                "tags": [],
                "products": [],
                "insights": []
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
            
            # Upload HTML and take screenshots to Supabase Storage
            storage_assets = await self.upload_html_and_take_screenshot(html_content, uuid_val)
            html_s3_link = storage_assets.get('html_url')
            full_screenshot_url = storage_assets.get('full_screenshot_url')
            top_screenshot_url = storage_assets.get('top_screenshot_url')

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
        run_once = os.getenv('RUN_ONCE', 'false').lower() in ('true', '1', 't', 'yes')
        check_interval = int(os.getenv('CHECK_INTERVAL', '120'))

        logger.info(f"Starting Newsletter Processor (RUN_ONCE={run_once}, CHECK_INTERVAL={check_interval}s)...")
        while True:
            mail = self.connect_to_imap()
            if not mail:
                logger.error(f"Failed to connect to IMAP server. Retrying in {check_interval} seconds...")
                if run_once:
                    break
                await asyncio.sleep(check_interval)
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

            if run_once:
                logger.info("Single run completed. Exiting.")
                break

            logger.info(f"Waiting {check_interval} seconds before next check...")
            await asyncio.sleep(check_interval)

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