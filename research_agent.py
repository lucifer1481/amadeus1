# research_agent.py
# =================================================================================================
#   Project Amadeus: The Evolved Grand Unified Core
#   Module: Research Agent
#   Description: A dedicated module for deep web scraping and summarization,
#                based on the user's final corrected script.
#   Version: 3.0 (User-Finalized Logic)
# =================================================================================================

import requests
from bs4 import BeautifulSoup
from litellm import completion
from urllib.parse import urljoin

# --- Using the DDGS library for reliable search ---
from ddgs import DDGS

from config import Config # Import configuration from our main project

def _scrape_page_for_links(url: str):
    """
    (Internal) Scrapes a page to find all links and their text.
    """
    print(f"[Research Agent] Scraping initial page for links: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = []
        for a_tag in soup.find_all('a', href=True):
            link_text = a_tag.get_text(strip=True)
            if link_text:
                absolute_url = urljoin(url, a_tag['href'])
                links.append((link_text, absolute_url))
        return links
    except Exception as e:
        print(f"[Research Agent] Error scraping for links: {e}")
        return None

def _find_most_relevant_link(links: list, query: str) -> str:
    """
    (Internal) Uses an LLM to choose the most relevant link.
    """
    print("[Research Agent] Using LLM to find the most relevant link...")
    link_map = {text: url for text, url in links}
    link_options = "\n".join([f"- '{text}'" for text, url in links[:40]])

    system_prompt = "You are an intelligent navigation assistant. Your task is to analyze a list of link texts from a webpage and select the single most relevant one based on the user's query. Respond with ONLY the exact text of the link you choose."
    user_prompt = f"""Based on my original query, which of the following links is the most likely to contain the information I need?

My query: "{query}"

Link texts found on the page:
{link_options}

The exact text of the single most relevant link is:"""

    try:
        response = completion(
            model=f"ollama/{Config.MODEL}",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        chosen_text = response.choices[0].message.content.strip().strip("'\"")
        
        if chosen_text in link_map:
            return link_map[chosen_text]
        else:
            print(f"[Research Agent] LLM chose a link text ('{chosen_text}') that was not in the original list. Aborting.")
            return None
    except Exception as e:
        print(f"[Research Agent] An error occurred while choosing a link: {e}")
        return None

def _scrape_final_content(url: str) -> str:
    """
    (Internal) Fetches the content of the final target URL.
    """
    print(f"[Research Agent] Performing deep scrape on final URL: {url}")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script_or_style in soup(["script", "style", "nav", "footer", "header"]):
            script_or_style.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        full_text = '\n'.join(chunk for chunk in chunks if chunk)

        return full_text if full_text else "Could not find any meaningful text on the final page."
    except Exception as e:
        return f"Error during final content scrape: {e}"

def _summarize_text(text: str, query: str) -> str:
    """
    (Internal) Uses an LLM to summarize the provided text.
    """
    print("[Research Agent] Summarizing final content with LLM...")
    text_to_summarize = text[:8000]
    system_prompt = "You are an expert research assistant. Your goal is to provide a concise, helpful summary of the provided text, focusing on the user's original query."
    user_prompt = f"""Based on the following text, please provide a summary that directly answers my question: "{query}"

Text to summarize:
---
{text_to_summarize}
---

Concise summary:"""
    try:
        response = completion(
            model=f"ollama/{Config.MODEL}",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred during summarization: {e}"

def run_research(query: str) -> str:
    """
    The main public function for the research agent. It orchestrates
    the entire search -> navigate -> scrape -> summarize pipeline.
    """
    try:
        # Step 1: Find the landing page using DDGS
        print(f"[Research Agent] Searching for: '{query}'")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=1))
            search_results = [r['href'] for r in results] if results else []

        if not search_results:
            return "I couldn't find any initial search results for that query, Lucifer."
        landing_page_url = search_results[0]

        # Step 2: Find links on the page
        links_on_page = _scrape_page_for_links(landing_page_url)
        if not links_on_page:
            # If no links, assume the landing page has the content
            final_content_url = landing_page_url
        else:
            # Step 3: Choose the best link to follow
            final_content_url = _find_most_relevant_link(links_on_page, query)
            if not final_content_url:
                # If LLM fails, fall back to the main page
                print("[Research Agent] LLM link selection failed. Falling back to scraping the main page.")
                final_content_url = landing_page_url

        # Step 4: Scrape the final page
        final_text = _scrape_final_content(final_content_url)
        if "Error" in final_text or "Could not find" in final_text:
            return f"I was unable to read the content from the final page ({final_content_url}). It might be protected or use a format I can't parse."

        # Step 5: Summarize
        summary = _summarize_text(final_text, query)
        
        return f"{summary}\n\nSource: {final_content_url}"

    except Exception as e:
        return f"A critical error occurred in the research agent: {e}"

