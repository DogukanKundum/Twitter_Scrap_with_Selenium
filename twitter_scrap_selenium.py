import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import re
import time
import string
import warnings

# for all NLP related operations on text
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.classify import NaiveBayesClassifier
# from wordcloud import WordCloud

# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB

# To mock web-browser and scrap tweets
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# To consume Twitter's API
import tweepy
from tweepy import OAuthHandler

# To identify the sentiment of text
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.np_extractors import ConllExtractor

# ignoring all the warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class SeleniumClient(object):
    def __init__(self):
        #Initialization method.
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-setuid-sandbox')

        # you need to provide the path of chromdriver in your system
        self.browser = webdriver.Chrome(options=self.chrome_options)

        self.base_url = 'https://twitter.com/search?q='

    def get_tweets(self, query):
        #Function to fetch tweets.
        try:
            self.browser.get(self.base_url+query)
            time.sleep(0.2)

            body = self.browser.find_element_by_tag_name('body')

            for _ in range(50000):
                body.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.1)
            timeline = self.browser.find_element_by_id('timeline')
            tweet_nodes = timeline.find_elements_by_css_selector('.tweet-text')
            tweet_times = timeline.find_elements_by_css_selector('.tweet-timestamp')
            test_times = [element.get_attribute('title') for element in self.browser.find_elements_by_xpath('//a[starts-with(@class, "tweet-timestamp js-permalink js-nav js-tooltip")]')]
            return pd.DataFrame({'tweets': [tweet_node.text for tweet_node in tweet_nodes], 'time' :  test_times})
        except:
            print("Selenium - An error occured while fetching tweets.")


selenium_client = SeleniumClient()

# calling function to get tweets
tweets_df = selenium_client.get_tweets('turktelekom')
print('tweets_df Shape - {tweets_df.shape}')
tweets_df.to_csv('test_turktelekom_tweet.csv', sep='\t', encoding='utf-8')
tweets_df.head(10)