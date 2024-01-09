import asyncio
import os
from pyppeteer import launch
from PIL import Image
from datetime import datetime

async def capture_screenshot(html_file_path, screenshot_path):
    browser = await launch(headless=True)
    page = await browser.newPage()
    
    try:
        # Load the HTML file
        await page.goto(f'file://{os.path.abspath(html_file_path)}', waitUntil='domcontentloaded')

        # Wait for a fixed duration (e.g., 5 seconds)
        await asyncio.sleep(5)

        # Skip screenshot capture if HTML file is monsterrealemailsubject.html
        if os.path.basename(html_file_path).lower() != 'monsterrealemailsubject.html':
            # Take a screenshot of the whole page
            await page.screenshot({'path': screenshot_path, 'fullPage': True})
    except Exception as e:
        print(f"An error occurred while taking a screenshot: {e}")
    finally:
        await browser.close()

def crop_to_1200px(input_path, output_path):
    try:
        image = Image.open(input_path)
        width, height = image.size
        bottom_crop = max(0, height - 1200)  # Crop from the bottom, leaving the top
        cropped_image = image.crop((0, 0, width, height - bottom_crop))
        cropped_image.save(output_path)
    except Exception as e:
        print(f"An error occurred while cropping the image: {e}")

# Set the path to the extracted folder
extracted_folder = 'extracted'
archived_folder = 'archived'

try:
    for folder_name in os.listdir(extracted_folder):
        folder_path = os.path.join(extracted_folder, folder_name)

        if os.path.isdir(folder_path):
            html_file = next((f for f in os.listdir(folder_path) if f.endswith('.html') and f.lower() != 'monsterrealemailsubject.html'), None)

            if html_file:
                subject_name = os.path.splitext(html_file)[0]
                html_file_path = os.path.join(folder_path, html_file)

                screenshot_file = f"{subject_name}.png"
                screenshot_path = os.path.join(folder_path, screenshot_file)

                asyncio.get_event_loop().run_until_complete(capture_screenshot(html_file_path, screenshot_path))

                cropped_screenshot_file = f"{subject_name}_cropped.png"
                cropped_screenshot_path = os.path.join(folder_path, cropped_screenshot_file)
                crop_to_1200px(screenshot_path, cropped_screenshot_path)

                print(f"Screenshots saved: {screenshot_path}, {cropped_screenshot_path}")

                # Move the folder to archived_folder
                current_date = datetime.now().strftime("%d%m%Y")
                current_date_folder = os.path.join(archived_folder, current_date)
                os.makedirs(current_date_folder, exist_ok=True)
                os.rename(folder_path, os.path.join(current_date_folder, folder_name))
                print(f"Folder moved to: {current_date_folder}")

except Exception as e:
    print(f"An error occurred while processing folders: {e}")