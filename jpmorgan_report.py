import json
import os
import smtplib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.mime.text import MIMEText

import feedparser
import groq
from dotenv import load_dotenv
from groq import Groq

# --- Load .env file (locally), falls back to env vars (GitHub Actions) ---
load_dotenv()
GMAIL_ADDRESS = os.environ["GMAIL_ADDRESS"]
GMAIL_APP_PASSWORD = os.environ["GMAIL_APP_PASSWORD"]
JPM_RECIPIENT = GMAIL_ADDRESS
client = Groq(api_key=os.environ["GROQ_API_KEY"])

# --- RSS feeds focused on JPMorgan / finance ---
FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=JPM&region=US&lang=en-US",
    "https://news.google.com/rss/search?q=JP+Morgan+OR+JPMorgan&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Jamie+Dimon&hl=en-US&gl=US&ceid=US:en",
]

SYSTEM_PROMPT = (
    "You write like a sharp, well-informed colleague at a major bank. "
    "Your tone is chill but direct — no fluff, no hype, no corporate speak. "
    "You're professional enough to trust, casual enough to enjoy reading "
    "over morning coffee. Keep it real and informative. "
    "The reader works at JP Morgan, so focus on what matters to them — "
    "company news, leadership moves, earnings, deals, regulation, "
    "and competitive landscape. Provide latest headline about company"
)


def fetch_articles() -> list[dict]:
    """Pull recent headlines from JPM-focused RSS feeds."""
    articles = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:15]:
            articles.append({
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "source": feed.feed.get("title", "Unknown"),
            })
    return articles


MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.1-8b-instant",
]


def ask_groq(prompt: str) -> str:
    """Send a prompt to Groq, falling back to the next model on rate limit."""
    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
            )
            return response.choices[0].message.content
        except (groq.RateLimitError, groq.APIStatusError) as e:
            print(f"{e.__class__.__name__} on {model}, trying next model...")
    raise RuntimeError("All models failed")


def pick_top_stories(articles: list[dict]) -> list[dict]:
    """Pick the most relevant JPM stories."""
    headlines = "\n".join(
        f"- {a['title']} [source: {a['source']}] [link: {a['link']}]"
        for a in articles
    )

    content = ask_groq(
        "Here are today's headlines related to JP Morgan:\n\n"
        f"{headlines}\n\n"
        "Pick the 5 most important stories for a JP Morgan employee. "
        "Prioritize: earnings, deals, leadership, regulation, "
        "competitive moves, and anything that affects day-to-day work. "
        "Return ONLY valid JSON — no other text, no markdown. "
        "Use this exact format:\n"
        '[{"headline": "...", "summary": "...", "link": "..."}]\n'
        "Each summary should be 2 sentences on why it matters to "
        "someone at JPM."
    )

    return _parse_json_response(content)


def pick_competitor_watch(articles: list[dict]) -> list[dict]:
    """Identify competitor and industry moves from the headlines."""
    headlines = "\n".join(
        f"- {a['title']} [source: {a['source']}]"
        for a in articles
    )

    content = ask_groq(
        "Here are today's headlines related to JP Morgan:\n\n"
        f"{headlines}\n\n"
        "Identify up to 3 stories about competitors (Goldman Sachs, "
        "Morgan Stanley, Citi, Bank of America, Wells Fargo) or "
        "industry-wide trends that a JP Morgan employee should know. "
        "Return ONLY valid JSON — no other text, no markdown. "
        "Use this exact format:\n"
        '[{"company": "...", "headline": "...", "takeaway": "..."}]\n'
        "Each takeaway should be 1 sentence. If there are no relevant "
        "competitor stories, return an empty array []."
    )

    return _parse_json_response(content)


def _parse_json_response(content: str) -> list[dict]:
    """Parse JSON from LLM response, with cleanup for common LLM quirks."""
    import re

    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]

    # Try parsing as-is first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Fix common LLM quirk: missing closing } on last object
    # e.g. ..."sector."]  should be  ..."sector."}]
    content = re.sub(r'"\s*\]\s*\]?\s*$', '"}]', content.strip())

    # Retry after fix
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Extract the first valid JSON array by trying progressively
    start = content.find("[")
    if start == -1:
        print(f"No JSON array found in response:\n{content}")
        return []

    # Try each ']' from left to right until one parses
    for match in re.finditer(r"\]", content[start:]):
        candidate = content[start : start + match.end()]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    print(f"Failed to parse LLM response as JSON:\n{content}")
    return []


def build_html(
    stories: list[dict], competitors: list[dict], date: str
) -> str:
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

    comp_items = ""
    for s in competitors:
        comp_items += f"""
        <tr>
            <td style="padding: 12px 0; border-bottom: 1px solid #eee;">
                <span style="font-size: 16px; font-weight: bold;
                    color: #1e40af;">
                    {s['company']}
                </span>
                <span style="font-size: 14px; color: #888;">
                    &mdash; {s['headline']}
                </span>
                <p style="margin: 4px 0 0 0; font-size: 14px;
                    color: #555; line-height: 1.4;">
                    {s['takeaway']}
                </p>
            </td>
        </tr>"""

    comp_section = ""
    if competitors:
        comp_section = f"""
                <tr>
                    <td style="background: #1e40af; padding: 16px 24px;">
                        <h2 style="margin: 0; font-size: 18px;
                            color: #fff;">
                            Competitor Watch
                        </h2>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 24px 24px 24px;">
                        <table width="100%">{comp_items}</table>
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
                    <td style="background: #0a2540; padding: 20px 24px;">
                        <h1 style="margin: 0; font-size: 20px;
                            color: #fff;">
                            JPMorgan Daily Brief
                        </h1>
                        <p style="margin: 4px 0 0 0; font-size: 13px;
                            color: #8899aa;">
                            {date} &bull; powered by Groq
                        </p>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 24px 24px 24px;">
                        <table width="100%">{items}</table>
                    </td>
                </tr>
                {comp_section}
                <tr>
                    <td style="padding: 16px 24px; background: #fafafa;
                        border-top: 1px solid #eee; font-size: 12px;
                        color: #999; text-align: center;">
                        News AI Agent by Tanmay
                    </td>
                </tr>
            </table>
        </td></tr>
        </table>
    </body>
    </html>"""


def send_email(subject: str, html_body: str):
    """Send an HTML email via Gmail SMTP."""
    msg = MIMEText(html_body, "html")
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = JPM_RECIPIENT

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(msg)


def main():
    today = datetime.now().strftime("%B %d, %Y")
    print(f"Fetching JPMorgan headlines for {today}...")

    articles = fetch_articles()
    print(f"Got {len(articles)} articles. Analyzing in parallel...")

    with ThreadPoolExecutor() as pool:
        stories_future = pool.submit(pick_top_stories, articles)
        comp_future = pool.submit(pick_competitor_watch, articles)
        stories = stories_future.result()
        competitors = comp_future.result()

    print(f"Got {len(stories)} stories + {len(competitors)} competitor items.")

    html = build_html(stories, competitors, today)
    subject = f"JPMorgan Daily Brief - {today}"
    send_email(subject, html)
    print(f"Email sent to {JPM_RECIPIENT}!")


if __name__ == "__main__":
    main()
