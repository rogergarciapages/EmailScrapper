# Email IMAP Scraper

* Pulls new emails from IMAP folder.
* downloads emails locally.
* extracts sender, subject, HTML & CSS content from email, date sent.
* updates Supabase database.
* renames HTML file with UUID+.html.
* uploads to S3 bucket.
* takes screenshots of the email.
* updates Supabase db with S3 Objects url.
* maks emails as read.

[x] mark html files as text
[] connect supabase api to frontend
[] create categories db

Using these libraries: Playwright, Beautifulsoup, PIL, imaplib, boto, s3, supabase (among others)

Taking screenshots in headlessmode.

![Default_a_computer_script_scraping_pulling_out_emails_like_cr_0](https://github.com/rogergarciaseo/EmailScrapper/assets/96830104/94c97def-fe30-4b3f-940f-4ea39326d562)
