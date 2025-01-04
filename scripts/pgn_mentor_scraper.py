#!/usr/bin/env python3
"""
Script to scrape chess games from PGN Mentor.
Downloads and extracts ZIP files containing PGN games for various players.
"""

import os
import time
import requests
import logging
from urllib.parse import urljoin
import zipfile
from io import BytesIO

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PGNMentorScraper:
    """Scraper for PGNMentor website to download chess games."""
    
    BASE_URL = "https://www.pgnmentor.com/players/"
    
    # Known player list with game counts
    PLAYERS = {
        'Abdusattorov': 2554,
        'Adams': 3558,
        'Akobian': 1755,
        'Akopian': 1987,
        'Alburt': 776,
        'Alekhine': 1661,
        'Alekseev': 2890,
        'Almasi': 2041,
        'Anand': 4162,
        'Anderssen': 681,
        'Andersson': 2736,
        'Andreikin': 5794,
        'Aronian': 4888,
        'Ashley': 414,
        'Averbakh': 885,
        'Azmaiparashvili': 1324,
        'Bacrot': 4529,
        'Bareev': 2012,
        'BecerraRivero': 1194,
        'Beliavsky': 3300,
        'Benjamin': 1784,
        'Benko': 1376,
        'Berliner': 64,
        'Bernstein': 261,
        'Bird': 353,
        'Bisguier': 1186,
        'Blackburne': 738,
        'Blatny': 1918,
        'Bogoljubow': 973,
        'Boleslavsky': 651,
        'Bologan': 3222,
        'Botvinnik': 891,
        'Breyer': 176,
        'Bronstein': 1930,
        'Browne': 1621,
        'Bruzon': 690,
        'Bu': 1977,
        'Byrne': 1013,
        'Capablanca': 597,
        'Carlsen': 6011,
        'Caruana': 4946,
        'Chiburdanidze': 1346,
        'Chigorin': 688,
        'Christiansen': 1637,
        'DeFirmian': 1712,
        'DeLaBourdonnais': 101,
        'Denker': 404,
        'Ding': 2063,
        'DominguezPerez': 2516,
        'Dreev': 4252,
        'Duda': 3688,
        'Dzindzichashvili': 780,
        'Ehlvest': 3366,
        'Eljanov': 3074,
        'Erigaisi': 3027,
        'Euwe': 1122,
        'Evans': 685,
        'Fedorowicz': 1438,
        'Fine': 305,
        'Finegold': 980,
        'Firouzja': 3713,
        'Fischer': 827,
        'Fishbein': 1107,
        'Flohr': 986,
        'Gaprindashvili': 1225,
        'Gashimov': 953,
        'Gelfand': 3897,
        'Geller': 2198,
        'Georgiev': 3096,
        'Giri': 3824,
        'Gligoric': 2898,
        'Goldin': 1095,
        'GrandaZuniga': 1917,
        'Grischuk': 5638,
        'Gukesh': 1979,
        'Gulko': 1876,
        'Gunsberg': 319,
        'GurevichD': 3590,
        'GurevichM': 2523,
        'Harikrishna': 2843,
        'Hort': 3152,
        'Horwitz': 113,
        'Hou': 2275,
        'Huebner': 2157,
        'Ibragimov': 1457,
        'IllescasCordoba': 1440,
        'Inarkiev': 2150,
        'Ivanchuk': 4839,
        'IvanovA': 1286,
        'IvanovI': 784,
        'Ivkov': 2505,
        'Jakovenko': 1943,
        'Janowski': 769,
        'Jobava': 4395,
        'Jussupow': 2098,
        'Kaidanov': 1433,
        'Kamsky': 6422,
        'Karjakin': 3486,
        'Karpov': 3529,
        'Kasimdzhanov': 1772,
        'Kasparov': 2128,
        'Kavalek': 1328,
        'Keres': 1571,
        'Khalifman': 2348,
        'Kholmov': 1900,
        'Koneru': 1957,
        'Korchnoi': 4569,
        'Korobov': 3318,
        'Kosteniuk': 5127,
        'Kotov': 641,
        'Kramnik': 4110,
        'Krasenkow': 2987,
        'Krush': 2199,
        'Kudrin': 1243,
        'Lahno': 919,
        'Larsen': 2383,
        'Lasker': 900,
        'Lautier': 1809,
        'Le': 2111,
        'Leko': 2672,
        'Levenfish': 354,
        'Li': 1323,
        'Lilienthal': 649,
        'Ljubojevic': 1944,
        'Lputian': 1552,
        'MacKenzie': 198,
        'Malakhov': 2163,
        'Mamedyarov': 4684,
        'Maroczy': 756,
        'Marshall': 1027,
        'McDonnell': 106,
        'McShane': 1702,
        'Mecking': 626,
        'Mikenas': 497,
        'Miles': 2614,
        'Milov': 1398,
        'Morozevich': 2729,
        'Morphy': 211,
        'Motylev': 2522,
        'Movsesian': 3284,
        'Muzychuk': 1678,
        'Najdorf': 1604,
        'Najer': 2488,
        'Nakamura': 8025,
        'Navara': 3678,
        'Negi': 1053,
        'Nepomniachtchi': 3941,
        'Ni': 1581,
        'Nielsen': 2128,
        'Nikolic': 2587,
        'Nimzowitsch': 512,
        'Nisipeanu': 2260,
        'Novikov': 1398,
        'Nunn': 1784,
        'Olafsson': 916,
        'Oll': 1034,
        'Onischuk': 1624,
        'Pachman': 1462,
        'Paehtz': 3143,
        'Panno': 1873,
        'Paulsen': 322,
        'Petrosian': 1893,
        'Philidor': 6,
        'Pillsbury': 388,
        'Pilnik': 899,
        'PolgarJ': 1825,
        'PolgarS': 856,
        'PolgarZ': 909,
        'Polugaevsky': 1890,
        'Ponomariov': 2469,
        'Portisch': 3030,
        'Praggnanandhaa': 2228,
        'Psakhis': 1963,
        'Quinteros': 1136,
        'Radjabov': 2579,
        'Rapport': 1974,
        'Reshevsky': 1267,
        'Reti': 646,
        'Ribli': 2231,
        'Rohde': 445,
        'Rubinstein': 797,
        'Rublevsky': 1853,
        'Saemisch': 551,
        'Sakaev': 1788,
        'Salov': 766,
        'Sasikiran': 2123,
        'Schlechter': 739,
        'Seirawan': 1519,
        'Serper': 848,
        'Shabalov': 2681,
        'Shamkovich': 977,
        'Shirov': 5380,
        'Short': 3216,
        'Shulman': 1191,
        'Smirin': 2718,
        'Smyslov': 2627,
        'So': 4161,
        'Sokolov': 3170,
        'Soltis': 370,
        'Spassky': 2231,
        'Speelman': 2356,
        'Spielmann': 1057,
        'Stahlberg': 964,
        'Staunton': 284,
        'Stefanova': 3486,
        'Stein': 699,
        'Steinitz': 590,
        'Suetin': 1284,
        'SultanKhan': 157,
        'Sutovsky': 1397,
        'Svidler': 4312,
        'Szabo': 1777,
        'Taimanov': 2117,
        'Tal': 2431,
        'Tarrasch': 704,
        'Tartakower': 1290,
        'Teichmann': 536,
        'Timman': 3612,
        'Tiviakov': 3076,
        'Tkachiev': 1598,
        'Tomashevsky': 1978,
        'Topalov': 2612,
        'TorreRepetto': 175,
        'Uhlmann': 2537,
        'Unzicker': 1265,
        'Ushenina': 2077,
        'VachierLagrave': 4862,
        'Vaganian': 2444,
        'VallejoPons': 2834,
        'VanWely': 4060,
        'Vitiugov': 2102,
        'Volokitin': 1768,
        'Waitzkin': 385,
        'Wang': 1648,
        'WangH': 2015,
        'Wei': 1711,
        'Winawer': 241,
        'Wojtaszek': 2717,
        'Wojtkiewicz': 1564,
        'Wolff': 640,
        'Xie': 701,
        'Xu': 491,
        'Ye': 662,
        'Yermolinsky': 1593,
        'Yu': 2323,
        'Yudasin': 1585,
        'Zhu': 1178,
        'Zukertort': 265,
        'Zvjaginsev': 2537,
    }
    
    def __init__(self, output_dir="data/pgn_games"):
        """Initialize scraper with output directory."""
        self.output_dir = output_dir
        self._create_output_dir()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
    
    def _create_output_dir(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory set to: {self.output_dir}")
    
    def _download_file(self, url):
        """Download file with error handling and retries."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                return response.content
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch {url}: {e}")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def download_player_games(self, player_name):
        """Download and extract ZIP file for a specific player."""
        zip_url = urljoin(self.BASE_URL, f"{player_name}.zip")
        
        try:
            # Download the ZIP file
            logger.info(f"Downloading games for {player_name}...")
            zip_content = self._download_file(zip_url)
            
            # Create a BytesIO object from the ZIP content
            zip_buffer = BytesIO(zip_content)
            
            # Try to open it as a ZIP file
            try:
                with zipfile.ZipFile(zip_buffer) as zip_ref:
                    # Extract the PGN file
                    for file in zip_ref.namelist():
                        if file.endswith('.pgn'):
                            zip_ref.extract(file, self.output_dir)
                            extracted_path = os.path.join(self.output_dir, file)
                            # Rename to player name if different
                            final_path = os.path.join(self.output_dir, f"{player_name}.pgn")
                            if extracted_path != final_path:
                                os.rename(extracted_path, final_path)
                            
                            logger.info(f"Successfully downloaded: {player_name}.pgn ({self.PLAYERS.get(player_name, 'unknown')} games)")
                            return True
            except zipfile.BadZipFile:
                logger.error(f"Downloaded content is not a valid ZIP file for {player_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading games for {player_name}: {e}")
            return False
    
    def download_all_games(self):
        """Download games for all known players."""
        successful = 0
        failed = 0
        
        for player_name in self.PLAYERS:
            if self.download_player_games(player_name):
                successful += 1
            else:
                failed += 1
            time.sleep(2)  # Be nice to the server
        
        logger.info(f"Download complete. Success: {successful}, Failed: {failed}")
        return successful, failed

def main():
    """Main function to run the scraper."""
    scraper = PGNMentorScraper()
    
    try:
        logger.info("Starting PGN download from PGN Mentor")
        logger.info(f"Will attempt to download games for {len(scraper.PLAYERS)} players")
        successful, failed = scraper.download_all_games()
        
        if successful > 0:
            logger.info(f"Successfully downloaded {successful} PGN files")
            logger.info(f"Files saved in: {scraper.output_dir}")
        
        if failed > 0:
            logger.warning(f"Failed to download {failed} PGN files")
            
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
