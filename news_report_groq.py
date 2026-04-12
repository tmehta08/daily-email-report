import json
import os
import smtplib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.mime.text import MIMEText

import feedparser
from dotenv import load_dotenv
from groq import Groq

# --- Load .env file (locally), falls back to env vars (GitHub Actions) ---
load_dotenv()
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# --- RSS feeds to pull from ---
NEWS_FEEDS = [
    "https://news.google.com/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
]

STOCK_FEEDS = [
    "https://finance.yahoo.com/news/rssindex",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
]


def fetch_from_feeds(feeds: list[str], limit: int = 20) -> list[dict]:
    """Pull recent headlines from RSS feeds."""
    articles = []
    for url in feeds:
        feed = feedparser.parse(url)
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "source": feed.feed.get("title", "Unknown"),
            })
    return articles


SYSTEM_PROMPT = (
    "You write like a sharp, well-informed friend who works in finance. "
    "Your tone is chill but direct — no fluff, no hype, no corporate speak. "
    "You're professional enough to trust, casual enough to enjoy reading "
    "over morning coffee. Keep it real and informative."
)


def ask_groq(prompt: str) -> str:
    """Send a prompt to Groq and return the response."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content


def pick_top_5(articles: list[dict]) -> list[dict]:
    """Use Groq to pick and summarize the top 5 stories."""
    headlines = "\n".join(
        f"- {a['title']} [source: {a['source']}] [link: {a['link']}]"
        for a in articles
    )

    content = ask_groq(
        "Here are today's news headlines:\n\n"
        f"{headlines}\n\n"
        "Pick the 5 most important and interesting stories. "
        "Return ONLY valid JSON — no other text, no markdown. "
        "Use this exact format:\n"
        '[{"headline": "...", "summary": "...", "link": "..."}]\n'
        "Each summary should be 2 sentences on why it matters."
    )

    return _parse_json_response(content)


def pick_top_5_stocks(articles: list[dict]) -> list[dict]:
    """Use Groq to pick 5 stocks to watch."""
    headlines = "\n".join(
        f"- {a['title']} [source: {a['source']}]"
        for a in articles
    )

    content = ask_groq(
        "Here are today's US stock market headlines:\n\n"
        f"{headlines}\n\n"
        "Based on these headlines, pick 10 US stocks to watch today. "
        "Return ONLY valid JSON — no other text, no markdown. "
        "Use this exact format:\n"
        '[{"ticker": "AAPL", "company": "Apple", "reason": "..."}]\n'
        "Each reason should be 3 sentences on why this stock is "
        "interesting today."
    )

    return _parse_json_response(content)


def _parse_json_response(content: str) -> list[dict]:
    """Parse JSON from LLM response, stripping code fences if needed."""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"Failed to parse LLM response as JSON:\n{content}")
        raise


def build_html(stories: list[dict], stocks: list[dict], date: str) -> str:
    """Build a clean HTML email."""
    items = ""
    for i, s in enumerate(stories, 1):
        items += f"""
        <tr>
            <td style="padding: 16px 0; border-bottom: 1px solid #eee;">
                <h2 style="margin: 0 0 6px 0; font-size: 17px;
                    color: #1a1a1a;">
                    {i}. {s['headline']}
                </h2>
                <p style="margin: 0 0 8px 0; font-size: 14px;
                    color: #555; line-height: 1.5;">
                    {s['summary']}
                </p>
                <a href="{s.get('link', '#')}"
                   style="font-size: 13px; color: #2563eb;
                   text-decoration: none;">
                    Read full story &rarr;
                </a>
            </td>
        </tr>"""

    stock_items = ""
    for s in stocks:
        stock_items += f"""
        <tr>
            <td style="padding: 12px 0; border-bottom: 1px solid #eee;">
                <span style="font-size: 16px; font-weight: bold;
                    color: #0d9488;">
                    {s['ticker']}
                </span>
                <span style="font-size: 14px; color: #888;">
                    &mdash; {s['company']}
                </span>
                <p style="margin: 4px 0 0 0; font-size: 14px;
                    color: #555; line-height: 1.4;">
                    {s['reason']}
                </p>
            </td>
        </tr>"""

    return f"""
    <html>
    <body style="margin: 0; padding: 0; background: #f5f5f5;
        font-family: -apple-system, Helvetica, Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0">
        <tr><td align="center" style="padding: 24px;">
            <table width="560" cellpadding="0" cellspacing="0"
                style="background: #fff; border-radius: 8px;
                overflow: hidden; box-shadow: 0 1px 3px
                rgba(0,0,0,0.1);">
                <tr>
                    <td style="background: #7c3aed; padding: 20px 24px;">
                        <h1 style="margin: 0; font-size: 20px;
                            color: #fff;">
                            Daily News Briefing
                        </h1>
                        <p style="margin: 4px 0 0 0; font-size: 13px;
                            color: #ddd;">
                            {date} &bull; powered by Llama 3.3 70B via Groq
                        </p>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 24px 24px 24px;">
                        <table width="100%">{items}</table>
                    </td>
                </tr>
                <tr>
                    <td style="background: #0d9488; padding: 16px 24px;">
                        <h2 style="margin: 0; font-size: 18px;
                            color: #fff;">
                            Stocks to Watch
                        </h2>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 24px 24px 24px;">
                        <table width="100%">{stock_items}</table>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 16px 24px; background: #fafafa;
                        border-top: 1px solid #eee; font-size: 12px;
                        color: #999; text-align: center;">
                        Generated by your AI agent with Groq + Llama 3.3
                    </td>
                </tr>
            </table>
        </td></tr>
        </table>
    </body>
    </html>"""


RECIPIENTS = [GMAIL_ADDRESS, "jmmeht01@yahoo.com"]


def send_email(subject: str, html_body: str):
    """Send an HTML email via Gmail SMTP."""
    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = ", ".join(RECIPIENTS)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)


def main():
    today = datetime.now().strftime("%B %d, %Y")
    print(f"Fetching headlines for {today}...")

    # Fetch both RSS feeds in parallel
    with ThreadPoolExecutor() as pool:
        news_future = pool.submit(fetch_from_feeds, NEWS_FEEDS)
        stock_future = pool.submit(fetch_from_feeds, STOCK_FEEDS)
        articles = news_future.result()
        stock_articles = stock_future.result()

    print(f"Got {len(articles)} news + {len(stock_articles)} stock articles.")
    print("Asking AI to pick top 10 of each (in parallel)...")

    # Both LLM calls in parallel — this is the big speedup
    with ThreadPoolExecutor() as pool:
        stories_future = pool.submit(pick_top_5, articles)
        stocks_future = pool.submit(pick_top_5_stocks, stock_articles)
        stories = stories_future.result()
        stocks = stocks_future.result()

    print(f"Got {len(stories)} stories + {len(stocks)} stocks. Building email...")

    html = build_html(stories, stocks, today)
    subject = f"News AI Agent by Tanmay - {today}"
    send_email(subject, html)
    print(f"Email sent to {GMAIL_ADDRESS}!")


if __name__ == "__main__":
    main()
