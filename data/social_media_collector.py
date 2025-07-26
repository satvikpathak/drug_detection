"""
Social Media Data Collector for Drug Detection Research
Collects real social media data from Twitter and Reddit for drug use and overdose detection.

This module implements:
- Twitter API integration for real-time data collection
- Reddit API integration for forum discussions
- Privacy-aware data collection
- Ethical data handling
- Real-time filtering and preprocessing

Author: Research Team
Date: 2024
"""

import tweepy
import praw
import pandas as pd
import numpy as np
import re
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
import logging
from collections import defaultdict
import hashlib
import os
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SocialMediaPost:
    """Data class for social media posts."""
    id: str
    text: str
    platform: str
    timestamp: datetime
    user_id: str
    hashtags: List[str]
    mentions: List[str]
    urls: List[str]
    emojis: List[str]
    language: str
    location: Optional[str]
    engagement_metrics: Dict[str, int]
    anonymized: bool = True


class PrivacyManager:
    """Manages privacy and ethical considerations for data collection."""
    
    def __init__(self):
        self.anonymization_salt = os.urandom(32)
        self.collected_hashes = set()
        
    def anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID using SHA-256."""
        salted = user_id.encode() + self.anonymization_salt
        return hashlib.sha256(salted).hexdigest()[:16]
    
    def anonymize_text(self, text: str) -> str:
        """Anonymize text while preserving drug-related content."""
        # Remove personal identifiers
        text = re.sub(r'@\w+', '@USER', text)
        text = re.sub(r'#\w+', lambda m: m.group(0), text)  # Keep hashtags
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        return text
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.collected_hashes:
            return True
        self.collected_hashes.add(text_hash)
        return False
    
    def meets_ethical_guidelines(self, text: str) -> bool:
        """Check if post meets ethical guidelines."""
        # Exclude posts with personal information
        personal_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{10}\b',  # Phone numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
        ]
        
        for pattern in personal_patterns:
            if re.search(pattern, text):
                return False
        
        return True


class DrugKeywordFilter:
    """Filters and categorizes drug-related content."""
    
    def __init__(self):
        self.drug_keywords = {
            'opioids': [
                'heroin', 'fentanyl', 'oxycodone', 'morphine', 'hydrocodone',
                'dope', 'smack', 'junk', 'horse', 'brown', 'china white',
                'oxy', 'percs', 'roxies', 'blues', '30s', '80s', 'fent'
            ],
            'stimulants': [
                'cocaine', 'methamphetamine', 'amphetamine', 'crack',
                'coke', 'blow', 'snow', 'powder', 'white', 'nose candy',
                'meth', 'crystal', 'ice', 'glass', 'tina', 'crank', 'speed'
            ],
            'symptoms': [
                'overdose', 'od', 'overdosing', 'nodding out', 'falling out',
                'nausea', 'puking', 'throwing up', 'sick to stomach',
                'dizziness', 'dizzy', 'lightheaded', 'woozy',
                'seizure', 'seizing', 'convulsing', 'shaking',
                'anxiety', 'panicking', 'freaking out', 'nervous'
            ],
            'slang': [
                'lit', 'baked', 'fried', 'zoned', 'faded', 'wasted',
                'dope sick', 'sick', 'kicking', 'cold turkey', 'detoxing',
                'plug', 'connect', 'guy', 'man', 'source'
            ]
        }
        
        # Create comprehensive keyword set
        self.all_keywords = set()
        for category, keywords in self.drug_keywords.items():
            self.all_keywords.update(keywords)
    
    def is_drug_related(self, text: str) -> bool:
        """Check if text contains drug-related keywords."""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.all_keywords)
    
    def categorize_content(self, text: str) -> Dict[str, bool]:
        """Categorize content by drug type and symptoms."""
        text_lower = text.lower()
        categories = {}
        
        for category, keywords in self.drug_keywords.items():
            categories[category] = any(keyword in text_lower for keyword in keywords)
        
        return categories
    
    def extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        return re.findall(r'#\w+', text)
    
    def extract_mentions(self, text: str) -> List[str]:
        """Extract user mentions from text."""
        return re.findall(r'@\w+', text)
    
    def extract_emojis(self, text: str) -> List[str]:
        """Extract emojis from text."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.findall(text)


class SocialMediaCollector(ABC):
    """Abstract base class for social media collectors."""
    
    def __init__(self, privacy_manager: PrivacyManager, keyword_filter: DrugKeywordFilter):
        self.privacy_manager = privacy_manager
        self.keyword_filter = keyword_filter
        self.collected_posts = []
        
    @abstractmethod
    def collect_posts(self, query: str, limit: int = 1000) -> List[SocialMediaPost]:
        """Collect posts from the platform."""
        pass
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the platform."""
        pass


class TwitterCollector(SocialMediaCollector):
    """Twitter data collector using Twitter API v2."""
    
    def __init__(self, bearer_token: str, privacy_manager: PrivacyManager, keyword_filter: DrugKeywordFilter):
        super().__init__(privacy_manager, keyword_filter)
        self.bearer_token = bearer_token
        self.client = None
        
    def authenticate(self) -> bool:
        """Authenticate with Twitter API."""
        try:
            self.client = tweepy.Client(bearer_token=self.bearer_token, wait_on_rate_limit=True)
            return True
        except Exception as e:
            logging.error(f"Twitter authentication failed: {e}")
            return False
    
    def collect_posts(self, query: str, limit: int = 1000) -> List[SocialMediaPost]:
        """Collect tweets using Twitter API v2."""
        if not self.client:
            if not self.authenticate():
                return []
        
        posts = []
        try:
            # Search tweets
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'lang', 'public_metrics', 'entities'],
                user_fields=['location'],
                expansions=['author_id', 'entities.mentions.username'],
                max_results=100
            ).flatten(limit=limit)
            
            for tweet in tweets:
                if self._should_collect_tweet(tweet):
                    post = self._process_tweet(tweet)
                    if post:
                        posts.append(post)
                        
        except Exception as e:
            logging.error(f"Error collecting tweets: {e}")
        
        return posts
    
    def _should_collect_tweet(self, tweet) -> bool:
        """Determine if tweet should be collected."""
        # Check if drug-related
        if not self.keyword_filter.is_drug_related(tweet.text):
            return False
        
        # Check ethical guidelines
        if not self.privacy_manager.meets_ethical_guidelines(tweet.text):
            return False
        
        # Check for duplicates
        if self.privacy_manager.is_duplicate(tweet.text):
            return False
        
        return True
    
    def _process_tweet(self, tweet) -> Optional[SocialMediaPost]:
        """Process tweet into SocialMediaPost object."""
        try:
            # Extract entities
            hashtags = []
            mentions = []
            urls = []
            
            if tweet.entities:
                if 'hashtags' in tweet.entities:
                    hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
                if 'mentions' in tweet.entities:
                    mentions = [mention['username'] for mention in tweet.entities['mentions']]
                if 'urls' in tweet.entities:
                    urls = [url['url'] for url in tweet.entities['urls']]
            
            # Extract emojis
            emojis = self.keyword_filter.extract_emojis(tweet.text)
            
            # Anonymize
            anonymized_text = self.privacy_manager.anonymize_text(tweet.text)
            anonymized_user_id = self.privacy_manager.anonymize_user_id(str(tweet.author_id))
            
            return SocialMediaPost(
                id=str(tweet.id),
                text=anonymized_text,
                platform='twitter',
                timestamp=tweet.created_at,
                user_id=anonymized_user_id,
                hashtags=hashtags,
                mentions=mentions,
                urls=urls,
                emojis=emojis,
                language=tweet.lang or 'en',
                location=None,  # Could be extracted from user data
                engagement_metrics={
                    'retweets': tweet.public_metrics.get('retweet_count', 0),
                    'likes': tweet.public_metrics.get('like_count', 0),
                    'replies': tweet.public_metrics.get('reply_count', 0)
                },
                anonymized=True
            )
            
        except Exception as e:
            logging.error(f"Error processing tweet: {e}")
            return None


class RedditCollector(SocialMediaCollector):
    """Reddit data collector using PRAW."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str,
                 privacy_manager: PrivacyManager, keyword_filter: DrugKeywordFilter):
        super().__init__(privacy_manager, keyword_filter)
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent
        self.reddit = None
        
    def authenticate(self) -> bool:
        """Authenticate with Reddit API."""
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            return True
        except Exception as e:
            logging.error(f"Reddit authentication failed: {e}")
            return False
    
    def collect_posts(self, subreddits: List[str], limit: int = 1000) -> List[SocialMediaPost]:
        """Collect posts from specified subreddits."""
        if not self.reddit:
            if not self.authenticate():
                return []
        
        posts = []
        try:
            for subreddit_name in subreddits:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Collect hot posts
                for submission in subreddit.hot(limit=limit // len(subreddits)):
                    if self._should_collect_submission(submission):
                        post = self._process_submission(submission)
                        if post:
                            posts.append(post)
                
                # Collect comments
                for submission in subreddit.hot(limit=limit // len(subreddits)):
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        if self._should_collect_comment(comment):
                            post = self._process_comment(comment)
                            if post:
                                posts.append(post)
                                
        except Exception as e:
            logging.error(f"Error collecting Reddit posts: {e}")
        
        return posts
    
    def _should_collect_submission(self, submission) -> bool:
        """Determine if submission should be collected."""
        text = submission.title + " " + (submission.selftext or "")
        
        if not self.keyword_filter.is_drug_related(text):
            return False
        
        if not self.privacy_manager.meets_ethical_guidelines(text):
            return False
        
        if self.privacy_manager.is_duplicate(text):
            return False
        
        return True
    
    def _should_collect_comment(self, comment) -> bool:
        """Determine if comment should be collected."""
        if not self.keyword_filter.is_drug_related(comment.body):
            return False
        
        if not self.privacy_manager.meets_ethical_guidelines(comment.body):
            return False
        
        if self.privacy_manager.is_duplicate(comment.body):
            return False
        
        return True
    
    def _process_submission(self, submission) -> Optional[SocialMediaPost]:
        """Process submission into SocialMediaPost object."""
        try:
            text = submission.title + " " + (submission.selftext or "")
            
            # Extract entities
            hashtags = self.keyword_filter.extract_hashtags(text)
            mentions = self.keyword_filter.extract_mentions(text)
            emojis = self.keyword_filter.extract_emojis(text)
            
            # Anonymize
            anonymized_text = self.privacy_manager.anonymize_text(text)
            anonymized_user_id = self.privacy_manager.anonymize_user_id(str(submission.author))
            
            return SocialMediaPost(
                id=str(submission.id),
                text=anonymized_text,
                platform='reddit',
                timestamp=datetime.fromtimestamp(submission.created_utc),
                user_id=anonymized_user_id,
                hashtags=hashtags,
                mentions=mentions,
                urls=[submission.url] if submission.url else [],
                emojis=emojis,
                language='en',  # Reddit is primarily English
                location=None,
                engagement_metrics={
                    'upvotes': submission.score,
                    'comments': submission.num_comments,
                    'downvotes': 0  # Not available in PRAW
                },
                anonymized=True
            )
            
        except Exception as e:
            logging.error(f"Error processing submission: {e}")
            return None
    
    def _process_comment(self, comment) -> Optional[SocialMediaPost]:
        """Process comment into SocialMediaPost object."""
        try:
            # Extract entities
            hashtags = self.keyword_filter.extract_hashtags(comment.body)
            mentions = self.keyword_filter.extract_mentions(comment.body)
            emojis = self.keyword_filter.extract_emojis(comment.body)
            
            # Anonymize
            anonymized_text = self.privacy_manager.anonymize_text(comment.body)
            anonymized_user_id = self.privacy_manager.anonymize_user_id(str(comment.author))
            
            return SocialMediaPost(
                id=str(comment.id),
                text=anonymized_text,
                platform='reddit',
                timestamp=datetime.fromtimestamp(comment.created_utc),
                user_id=anonymized_user_id,
                hashtags=hashtags,
                mentions=mentions,
                urls=[],
                emojis=emojis,
                language='en',
                location=None,
                engagement_metrics={
                    'upvotes': comment.score,
                    'comments': 0,
                    'downvotes': 0
                },
                anonymized=True
            )
            
        except Exception as e:
            logging.error(f"Error processing comment: {e}")
            return None


class DataCollectionManager:
    """Manages the overall data collection process."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.privacy_manager = PrivacyManager()
        self.keyword_filter = DrugKeywordFilter()
        self.collectors = {}
        self.collected_data = []
        
        self._initialize_collectors()
    
    def _initialize_collectors(self):
        """Initialize social media collectors."""
        # Twitter collector
        if 'twitter' in self.config and self.config['twitter'].get('enabled', False):
            twitter_config = self.config['twitter']
            self.collectors['twitter'] = TwitterCollector(
                bearer_token=twitter_config['bearer_token'],
                privacy_manager=self.privacy_manager,
                keyword_filter=self.keyword_filter
            )
        
        # Reddit collector
        if 'reddit' in self.config and self.config['reddit'].get('enabled', False):
            reddit_config = self.config['reddit']
            self.collectors['reddit'] = RedditCollector(
                client_id=reddit_config['client_id'],
                client_secret=reddit_config['client_secret'],
                user_agent=reddit_config['user_agent'],
                privacy_manager=self.privacy_manager,
                keyword_filter=self.keyword_filter
            )
    
    def collect_data(self, queries: List[str], subreddits: List[str] = None, 
                    limit_per_platform: int = 1000) -> pd.DataFrame:
        """Collect data from all platforms."""
        all_posts = []
        
        # Collect from Twitter
        if 'twitter' in self.collectors:
            for query in queries:
                posts = self.collectors['twitter'].collect_posts(query, limit_per_platform)
                all_posts.extend(posts)
        
        # Collect from Reddit
        if 'reddit' in self.collectors and subreddits:
            posts = self.collectors['reddit'].collect_posts(subreddits, limit_per_platform)
            all_posts.extend(posts)
        
        # Convert to DataFrame
        df = self._posts_to_dataframe(all_posts)
        self.collected_data = df
        
        return df
    
    def _posts_to_dataframe(self, posts: List[SocialMediaPost]) -> pd.DataFrame:
        """Convert posts to DataFrame."""
        data = []
        for post in posts:
            data.append({
                'id': post.id,
                'text': post.text,
                'platform': post.platform,
                'timestamp': post.timestamp,
                'user_id': post.user_id,
                'hashtags': ' '.join(post.hashtags),
                'mentions': ' '.join(post.mentions),
                'urls': ' '.join(post.urls),
                'emojis': ' '.join(post.emojis),
                'language': post.language,
                'location': post.location,
                'retweets': post.engagement_metrics.get('retweets', 0),
                'likes': post.engagement_metrics.get('likes', 0),
                'upvotes': post.engagement_metrics.get('upvotes', 0),
                'comments': post.engagement_metrics.get('comments', 0),
                'anonymized': post.anonymized
            })
        
        return pd.DataFrame(data)
    
    def save_data(self, filepath: str):
        """Save collected data to file."""
        if self.collected_data is not None:
            self.collected_data.to_csv(filepath, index=False)
            logging.info(f"Data saved to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about collected data."""
        if self.collected_data is None or len(self.collected_data) == 0:
            return {}
        
        stats = {
            'total_posts': len(self.collected_data),
            'platforms': self.collected_data['platform'].value_counts().to_dict(),
            'languages': self.collected_data['language'].value_counts().to_dict(),
            'date_range': {
                'start': self.collected_data['timestamp'].min(),
                'end': self.collected_data['timestamp'].max()
            }
        }
        
        return stats


# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    config = {
        'twitter': {
            'enabled': True,
            'bearer_token': 'YOUR_TWITTER_BEARER_TOKEN'
        },
        'reddit': {
            'enabled': True,
            'client_id': 'YOUR_REDDIT_CLIENT_ID',
            'client_secret': 'YOUR_REDDIT_CLIENT_SECRET',
            'user_agent': 'DrugDetectionResearch/1.0'
        }
    }
    
    # Initialize manager
    manager = DataCollectionManager(config)
    
    # Define search queries
    queries = [
        'overdose OR "drug use" OR heroin OR fentanyl OR cocaine',
        'withdrawal OR "dope sick" OR detox',
        'nausea OR vomiting OR "throwing up"',
        'anxiety OR panic OR "freaking out"'
    ]
    
    # Define subreddits
    subreddits = [
        'opiates',
        'drugs',
        'addiction',
        'recovery'
    ]
    
    # Collect data
    df = manager.collect_data(queries, subreddits, limit_per_platform=500)
    
    # Save data
    manager.save_data('social_media_drug_data.csv')
    
    # Print statistics
    stats = manager.get_statistics()
    print("Collection Statistics:")
    print(json.dumps(stats, indent=2, default=str))