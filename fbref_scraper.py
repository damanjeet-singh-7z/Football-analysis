"""
FBref Web Scraper for Football Statistics
Scrapes player statistics, team data, and match results from FBref.com
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from typing import Optional, List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FBrefScraper:
    """Scraper for FBref.com football statistics"""
    
    BASE_URL = "https://fbref.com"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    def __init__(self, delay: float = 3.0):
        """
        Initialize scraper with rate limiting
        
        Args:
            delay: Seconds to wait between requests (FBref asks for 3+ seconds)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def _get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse URL with error handling"""
        try:
            time.sleep(self.delay)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logging.error(f"Error fetching {url}: {e}")
            return None
    
    def scrape_league_stats(self, league: str, season: str) -> pd.DataFrame:
        """
        Scrape player statistics for a league season
        
        Args:
            league: League code (e.g., 'Premier-League', 'La-Liga', 'Serie-A')
            season: Season year (e.g., '2023-2024')
        
        Returns:
            DataFrame with player statistics
        """
        url = f"{self.BASE_URL}/en/comps/9/{season}/stats/{season}-{league}-Stats"
        logging.info(f"Scraping {league} {season} statistics...")
        
        soup = self._get_soup(url)
        if not soup:
            return pd.DataFrame()
        
        # Find the stats table
        table = soup.find('table', {'id': 'stats_standard'})
        if not table:
            logging.warning("Stats table not found")
            return pd.DataFrame()
        
        # Parse table to DataFrame
        df = pd.read_html(str(table))[0]
        
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # Clean column names
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        # Add metadata
        df['league'] = league
        df['season'] = season
        
        logging.info(f"Scraped {len(df)} player records")
        return df
    
    def scrape_team_stats(self, league_url: str) -> pd.DataFrame:
        """
        Scrape team statistics from a league page
        
        Args:
            league_url: Full URL to league stats page
        
        Returns:
            DataFrame with team statistics
        """
        logging.info(f"Scraping team statistics from {league_url}")
        
        soup = self._get_soup(league_url)
        if not soup:
            return pd.DataFrame()
        
        # Find all stat tables
        tables = soup.find_all('table', class_='stats_table')
        
        if not tables:
            logging.warning("No stats tables found")
            return pd.DataFrame()
        
        # Parse first table (usually team standings/stats)
        df = pd.read_html(str(tables[0]))[0]
        
        # Flatten columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        logging.info(f"Scraped {len(df)} team records")
        return df
    
    def scrape_player_search(self, player_name: str) -> List[Dict]:
        """
        Search for a player and get their profile links
        
        Args:
            player_name: Name of the player to search
        
        Returns:
            List of player info dictionaries
        """
        search_url = f"{self.BASE_URL}/en/search/search.fcgi?search={player_name.replace(' ', '+')}"
        logging.info(f"Searching for player: {player_name}")
        
        soup = self._get_soup(search_url)
        if not soup:
            return []
        
        results = []
        search_results = soup.find('div', {'id': 'search-results'})
        
        if search_results:
            items = search_results.find_all('div', class_='search-item-name')
            for item in items:
                link = item.find('a')
                if link:
                    results.append({
                        'name': link.text.strip(),
                        'url': self.BASE_URL + link['href']
                    })
        
        logging.info(f"Found {len(results)} results")
        return results
    
    def scrape_player_profile(self, player_url: str) -> Dict:
        """
        Scrape detailed player profile data
        
        Args:
            player_url: Full URL to player's FBref page
        
        Returns:
            Dictionary with player information and statistics
        """
        logging.info(f"Scraping player profile: {player_url}")
        
        soup = self._get_soup(player_url)
        if not soup:
            return {}
        
        data = {'url': player_url}
        
        # Get player info
        info_box = soup.find('div', {'id': 'info'})
        if info_box:
            # Extract name
            h1 = info_box.find('h1')
            if h1:
                data['name'] = h1.text.strip()
            
            # Extract other info (position, age, etc.)
            paragraphs = info_box.find_all('p')
            for p in paragraphs:
                text = p.text.strip()
                if 'Position:' in text:
                    data['position'] = text.split('Position:')[1].strip().split('▪')[0].strip()
                if 'Born:' in text:
                    data['birth_info'] = text.split('Born:')[1].strip()
        
        # Get career stats tables
        stats_tables = soup.find_all('table', class_='stats_table')
        data['tables'] = {}
        
        for table in stats_tables:
            table_id = table.get('id', 'unknown')
            try:
                df = pd.read_html(str(table))[0]
                data['tables'][table_id] = df
            except:
                continue
        
        logging.info(f"Scraped profile with {len(data['tables'])} stat tables")
        return data
    
    def scrape_match_results(self, league: str, season: str) -> pd.DataFrame:
        """
        Scrape match results for a league season
        
        Args:
            league: League name
            season: Season year
        
        Returns:
            DataFrame with match results
        """
        url = f"{self.BASE_URL}/en/comps/9/{season}/schedule/{season}-{league}-Scores-and-Fixtures"
        logging.info(f"Scraping match results: {league} {season}")
        
        soup = self._get_soup(url)
        if not soup:
            return pd.DataFrame()
        
        table = soup.find('table', {'id': 'sched_ks_all'})
        if not table:
            logging.warning("Match schedule table not found")
            return pd.DataFrame()
        
        df = pd.read_html(str(table))[0]
        
        # Clean columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        logging.info(f"Scraped {len(df)} match records")
        return df
    
    def scrape_big5_leagues(self, season: str = '2023-2024') -> pd.DataFrame:
        """
        Scrape player statistics from all Big 5 European leagues
        
        Args:
            season: Season to scrape
        
        Returns:
            Combined DataFrame with all leagues
        """
        leagues = {
            'Premier-League': '9',
            'La-Liga': '12',
            'Serie-A': '11',
            'Bundesliga': '20',
            'Ligue-1': '13'
        }
        
        all_data = []
        
        for league_name, league_id in leagues.items():
            url = f"{self.BASE_URL}/en/comps/{league_id}/{season}/stats/{season}-{league_name}-Stats"
            
            soup = self._get_soup(url)
            if not soup:
                continue
            
            table = soup.find('table', {'id': 'stats_standard'})
            if table:
                df = pd.read_html(str(table))[0]
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                
                df['league'] = league_name
                df['season'] = season
                all_data.append(df)
                logging.info(f"Scraped {len(df)} records from {league_name}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            logging.info(f"Total records scraped: {len(combined)}")
            return combined
        
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    scraper = FBrefScraper(delay=3.5)
    
    # Example 1: Scrape Big 5 leagues
    print("\n=== Scraping Big 5 Leagues ===")
    big5_data = scraper.scrape_big5_leagues(season='2023-2024')
    if not big5_data.empty:
        big5_data.to_csv('big5_leagues_2023_24.csv', index=False)
        print(f"Saved {len(big5_data)} records to big5_leagues_2023_24.csv")
    
    # Example 2: Search for a player
    print("\n=== Searching for Player ===")
    players = scraper.scrape_player_search("Erling Haaland")
    for player in players[:3]:
        print(f"Found: {player['name']} - {player['url']}")
    
    # Example 3: Scrape match results
    print("\n=== Scraping Match Results ===")
    matches = scraper.scrape_match_results('Premier-League', '2023-2024')
    if not matches.empty:
        print(f"Scraped {len(matches)} matches")
        matches.to_csv('premier_league_matches.csv', index=False)
