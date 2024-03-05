import os
import logging
import traceback
import imaplib
import email
from email.header import decode_header
import boto3
import botocore.session
import botocore.exceptions
from botocore.exceptions import NoCredentialsError
import datetime
from supabase import create_client
from bs4 import BeautifulSoup
import asyncio
from pyppeteer import launch, errors
from PIL import Image

# Configure logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Supabase client
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BASE_URL = "https://newslettermonster.com/"

# Email credentials
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# S3 credentials
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
S3_BUCKET = os.getenv('AWS_S3_BUCKET_NAME')


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


# Get email subject
def get_email_subject(email_msg):
    """
    Extracts the subject of the email message.
    """
    subject, _ = decode_header(email_msg["Subject"])[0]
    return subject.decode() if isinstance(subject, bytes) else subject


# Extract HTML content from email
def extract_html(email_msg):
    """
    Extracts the HTML content from the email message.
    """
    html = None
    for part in email_msg.walk():
        if part.get_content_type() == "text/html":
            charset = part.get_content_charset()
            html = part.get_payload(decode=True).decode(charset)
            break
    if html is None:
        logger.warning("No HTML content found in the email.")
    else:
        logger.info("HTML content extracted successfully")
        logger.debug(f"HTML content: {html}")  # Log the HTML content
        soup = BeautifulSoup(html, 'html.parser')
        html = soup.prettify()  # Convert BeautifulSoup object back to prettified HTML string
    return html


# Upload HTML content to S3
def upload_to_s3(html, uuid):
    """
    Uploads the HTML content to S3 and returns the S3 object URL.
    """
    try:
        s3 = boto3.client('s3',
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        # Generate S3 key with UUID
        key = f"{uuid}.html"
        # Upload HTML content to S3 bucket
        s3.put_object(Body=html.encode(), Bucket=S3_BUCKET, Key=key)
        logger.info("HTML uploaded to S3")

        # Construct URL of the uploaded object
        object_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
        logger.info(f"S3 object URL: {object_url}")
        
        # Update Supabase with S3 object URL
        update_supabase_with_s3_link(uuid, object_url)
        
        return object_url
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == 'InvalidAccessKeyId':
            logger.error("Invalid AWS access key ID provided.")
        elif error_code == 'InvalidSecretAccessKey':
            logger.error("Invalid AWS secret access key provided.")
        else:
            logger.error(f"S3 upload error: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        logger.error(f"S3 upload error: {e}")
        traceback.print_exc()
        return None

# Update Supabase with S3 object URL
def update_supabase_with_s3_link(uuid, object_url):
    try:
        # Update record in Supabase table with the S3 object URL
        supabase.table("TableN1").update({
            "S3link": object_url,
        }).eq("id", uuid).execute()
        logger.info("S3 object URL updated in Supabase")
    except Exception as e:
        logger.error(f"Error updating S3 object URL in Supabase: {e}")
        traceback.print_exc()

# Get email sender
def get_email_sender(email_msg):
    return email.utils.parseaddr(email_msg.get("From"))[1]

# Get email received date
def get_email_date(email_msg):
    return email.utils.parsedate_to_datetime(email_msg.get("Date"))

# Generate slug
def generate_slug(uuid):
    return f"{BASE_URL}{uuid}.html"

# Update slug in Supabase
def update_slug(uuid, slug):
    try:
        # Update record in Supabase table with the generated slug
        supabase.table("TableN1").update({
            "slug": slug,
        }).eq("id", uuid).execute()
        logger.info("Slug updated in Supabase")
    except Exception as e:
        logger.error(f"Error updating slug in Supabase: {e}")
        traceback.print_exc()

# Process an email
async def process_email(email_msg, msg_id):
    try:
        # Extract info from email
        subject = get_email_subject(email_msg)
        sender = get_email_sender(email_msg)
        date = get_email_date(email_msg)

        # Get HTML content
        html = extract_html(email_msg)
        logger.debug(f"HTML content after extraction: {html}")  # log the HTML content

        # Skip processing if html is None
        if html is None:
            logger.warning("Skipping email processing due to missing HTML content.")
            return

        # Insert record into Supabase
        uuid = insert_to_supabase(subject, sender, date)
        logger.debug(f"UUID: {uuid}")  # log the UUID

        # Save HTML content to a local file
        file_path = f"{uuid}.html"
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(html)
        logger.info("HTML content saved to local file")

        # Generate and update slug
        if uuid:
            slug = generate_slug(uuid)
            update_slug(uuid, slug)

        # Capture the current date and time when the email is processed
        processed_time = datetime.datetime.now()

        # Update Supabase with processing time
        if uuid:
            update_processing_time(uuid, processed_time)

        # Upload file to S3
        if uuid:
            s3_key = upload_to_s3(html, uuid)
            logger.debug(f"S3 key: {s3_key}")  # log the S3 key
            
            # Delete local HTML file after uploading to S3
            os.remove(file_path)
            logger.info("Local HTML file deleted after uploading to S3")
        
        # Take screenshots
        await take_screenshots(html, uuid)

        # Mark email as "read"
        uid = mail.fetch(msg_id, '(UID)')[1][0].split()[2]  # Extract UID
        mail.store(msg_id, '+FLAGS', '\\Seen')
        logger.info("Email marked as read")

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
            "date_processed": processing_time.isoformat(),  # Convert to ISO format string
        }).eq("id", uuid).execute()
        logger.info("Processing time updated in Supabase")
    except Exception as e:
        logger.error(f"Error updating processing time in Supabase: {e}")
        traceback.print_exc()

# Define browser variable in the global scope
browser = None

# Take screenshots
async def take_screenshots(html, uuid):
    global browser  # Use the global browser variable
    
    try:
        browser = await launch(headless=True)
        page = await browser.newPage()
        await page.setContent(html)
        
        # Wait for the content to load
        await page.waitForSelector('body')
        
        # Take full-height screenshot
        full_height_path = f"{uuid}_full.webp"
        await page.setViewport({'width': 900, 'height': 100})
        await page.screenshot({'path': full_height_path, 'fullPage': True})
        logger.info(f"Full-height screenshot saved: {full_height_path}")
        
        # Take 900x900 screenshot
        thumb_path = f"{uuid}_thumb.webp"
        await page.setViewport({'width': 900, 'height': 900})
        await page.evaluate('window.scrollTo(0, 125)')  # Scroll down to leave a gap of 125px
        await page.screenshot({'path': thumb_path})
        logger.info(f"900x900 screenshot saved: {thumb_path}")
        
        # Upload screenshots to S3
        upload_to_s3_image(full_height_path)
        upload_to_s3_image(thumb_path)
        
    except Exception as e:
        logger.error(f"Error taking screenshots: {e}")
        traceback.print_exc()
    finally:
        if browser:
            await browser.close()

# Upload image to S3
def upload_to_s3_image(image_path):
    try:
        s3 = boto3.client('s3', 
                          aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        
        # Generate S3 key with filename
        key = os.path.basename(image_path)
        
        # Upload image to S3 bucket
        with open(image_path, 'rb') as f:
            s3.put_object(Body=f, Bucket=S3_BUCKET, Key=key, ContentType='image/webp')
        
        # Construct URL of the uploaded image
        object_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
        logger.info(f"Image uploaded to S3: {object_url}")
        
        # Remove local image file after uploading to S3
        os.remove(image_path)
        logger.info("Local image file deleted after uploading to S3")
        
        # Update Supabase with S3 object URL
        update_supabase_with_s3_link(uuid, object_url)
        
    except Exception as e:
        logger.error(f"Error uploading image to S3: {e}")
        traceback.print_exc()

# Main function
if __name__ == "__main__":
    mail = connect_to_imap()
    if mail:
        try:
            # Select inbox
            mail.select('INBOX')

            # Get all message IDs
            _, msg_ids = mail.search(None, 'UNSEEN')  # Only fetch unseen emails
            msg_ids = msg_ids[0].split()

            # Process each email
            loop = asyncio.get_event_loop()
            for msg_id in msg_ids:
                _, msg_data = mail.fetch(msg_id, '(RFC822)')
                email_msg = email.message_from_bytes(msg_data[0][1])
                # Log the subject of the email message
                logger.info(f"Processing email with subject: {get_email_subject(email_msg)}")
                loop.run_until_complete(process_email(email_msg, msg_id))

        except Exception as e:
            logger.error(f"Error processing emails: {e}")
            traceback.print_exc()
        finally:
            # Close IMAP connection
            mail.logout()