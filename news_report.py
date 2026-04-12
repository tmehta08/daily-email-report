import json
import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText

import feedparser  # third-party
import ollama  # third-party

# --- Config from environment ---
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]

# --- RSS feeds to pull from ---
FEEDS = [
    "https://news.google.com/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
]


def fetch_headlines() -> list[dict]:
    """Pull recent headlines from RSS feeds."""
    articles = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "source": feed.feed.get("title", "Unknown"),
            })
    return articles


def pick_top_5(articles: list[dict]) -> str:
    """Use the local LLM to pick and summarize the top 5 stories."""
    headlines = "\n".join(
        f"- {a['title']} ({a['source']})" for a in articles
    )

    response = ollama.chat(
        model="qwen3:1.7b",
        messages=[
            {
                "role": "user",
                "content": (
                    "Here are today's news headlines:\n\n"
                    f"{headlines}\n\n"
                    "Pick the 5 most important and interesting stories. "
                    "For each, write:\n"
                    "1. A clear headline\n"
                    "2. A 1-2 sentence summary of why it matters\n\n"
                    "Format it as a numbered list. Be concise. "
                    "Do not include any thinking or reasoning tags."
                ),
            }
        ],
    )
    return response.message.content


def send_email(subject: str, body: str):
    """Send an email via Gmail SMTP."""
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = GMAIL_ADDRESS

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)


def main():
    today = datetime.now().strftime("%B %d, %Y")
    print(f"Fetching headlines for {today}...")

    articles = fetch_headlines()
    print(f"Got {len(articles)} articles. Asking AI to pick top 5...")

    summary = pick_top_5(articles)
    print(f"Summary ready. Sending email...")

    subject = f"Daily News Briefing - {today}"
    send_email(subject, summary)
    print(f"Email sent to {GMAIL_ADDRESS}!")


if __name__ == "__main__":
    main()
