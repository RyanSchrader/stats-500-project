import csv
import requests
import bs4
from textblob import TextBlob
import numpy as np
import lda
import thread
import threading
import sys
from time import sleep

def collect_article_urls(filename):
    #return a list
    urls_videos = list()
    mycsv = csv.reader(open(filename))
    first = True
    for row in mycsv:
        if first:   # used to ignore the first row in the csv,
                    # which is the var names
            first = False
            continue
        url = row[0]
        videos = row[10]
        #urls.append(url)
        urls_videos.append( (url,videos) )
    return urls_videos


def write_to_csv(filename, article_statistics):
    keys = ["url", "date", "time", "author", "title_num_words", "title_num_characters", "content_num_paragraphs", "content_num_words", "content_num_characters", "avg_word_length", "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos", "num_topics", "data_channel_is_business", "data_channel_is_entertainment", "data_channel_is_lifestyle", "data_channel_is_social-media"]
    keys.extend(["data_channel_is_tech",  "data_channel_is_watercooler", "data_channel_is_world",  "self_reference_min_shares", "self_reference_max_shares", "self_reference_avg_shares", "weekday_is_Mon", "weekday_is_Tue", "weekday_is_Wed", "weekday_is_Thu", "weekday_is_Fri", "weekday_is_Sat", "weekday_is_Sun", "global_sentiment_subjectivity", "global_sentiment_polarity", "global_rate_positive_words", "global_rate_negative_words", "rate_positive_words", "rate_negative_words"])
    keys.extend(["min_positive_polarity", "max_positive_polarity", "avg_positive_polarity", "min_negative_polarity", "max_negative_polarity", "avg_negative_polarity", "title_sentiment_subjectivity", "title_sentiment_polarity", "shares_facebook", "shares_google_plus", "shares_linked_in", "shares_pinterest", "shares_stumble_upon", "shares_twitter", "shares_total"])

    with open(filename, 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(article_statistics)


def calc_shares(str_shares):
    if "k" in str_shares:
        shares_thousands = str_shares.replace("k","")
        return int(float(shares_thousands) * 1000)
    else:
        return int(str_shares)


def main():
    input_filename = "OnlineNewsPopularity/OnlineNewsPopularity.csv"
    article_urls_videos = collect_article_urls(input_filename)
    #article_urls_videos = list()
    #article_urls_videos.append(("http://mashable.com/2013/01/25/data-vs-nature-infographic/#uYW0Pfm2f5qj",3))
    #article_urls_videos.append(("http://mashable.com/2013/01/16/facebook-graph-search-name/#3HJqUDApA8qQ",1))
    #article_urls_videos.append(("http://mashable.com/2013/01/15/facebook-graph-search-great/%20/",2))
    #article_urls_videos.append(("http://mashable.com/2013/03/06/mashbash-sxswi",1))

    #article_statistics = dict()
    article_stats = list()

    keys = ["url", "date", "time", "author", "title_num_words", "title_num_characters", "content_num_paragraphs", "content_num_words", "content_num_characters", "avg_word_length", "num_hrefs", "num_self_hrefs", "num_imgs", "num_videos", "num_topics", "data_channel_is_business", "data_channel_is_entertainment", "data_channel_is_lifestyle", "data_channel_is_social-media"]
    keys.extend(["data_channel_is_tech",  "data_channel_is_watercooler", "data_channel_is_world",  "self_reference_min_shares", "self_reference_max_shares", "self_reference_avg_shares", "weekday_is_Mon", "weekday_is_Tue", "weekday_is_Wed", "weekday_is_Thu", "weekday_is_Fri", "weekday_is_Sat", "weekday_is_Sun", "global_sentiment_subjectivity", "global_sentiment_polarity", "global_rate_positive_words", "global_rate_negative_words", "rate_positive_words", "rate_negative_words"])
    keys.extend(["min_positive_polarity", "max_positive_polarity", "avg_positive_polarity", "min_negative_polarity", "max_negative_polarity", "avg_negative_polarity", "title_sentiment_subjectivity", "title_sentiment_polarity", "shares_facebook", "shares_google_plus", "shares_linked_in", "shares_pinterest", "shares_stumble_upon", "shares_twitter", "shares_total"])

    with open('scraped_dataset_all_samples.csv', 'wb') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        #dict_writer.writerows(article_statistics)
        for url_video in article_urls_videos:
            #article_statistics[url_video[0]] = collect_article_variables(url_video)
            row = collect_article_variables(url_video)
            if row == -1:
                continue
            else:
                #article_stats.append(val)
                dict_writer.writerow(row)
    #write_to_csv('scraped_dataset_all_samples.csv', article_stats)

#####
counter = 0
def collect_article_variables(url_video):
    try:
        #url = "http://mashable.com/2013/01/07/amazon-instant-video-browser/"
        #url = "http://mashable.com/2013/01/07/creature-cups/"
        #url = "http://mashable.com/2013/01/09/gopro-videos/"
        article_statistics = dict()
        article_statistics["url"] = url_video[0]
        article_statistics["num_videos"] = url_video[1]

        sleep(1)

        try:
            res = requests.get(url_video[0])
        except:
            return -1
        #res.raise_for_status()
        if res.status_code != requests.codes.ok:
            return -1

        soup = bs4.BeautifulSoup(res.text)

        # grab total shares
        shares_total = soup.find(class_="total-shares")
        #article_statistics['shares_total'] = calc_shares(shares_total.find('em').string)
        article_statistics['shares_total'] = 0

        # grab shares per social media platform
        social_media_platforms = ["facebook", "twitter", "google_plus", "linked_in", "stumble_upon", "pinterest"]
        for platform in social_media_platforms:
            platform_shares = calc_shares(soup.find(class_="social-stub social-share " + platform)['data-shares'])
            article_statistics["shares_" + platform] = platform_shares
            article_statistics['shares_total'] += platform_shares

        # grab author
        author_name = soup.find(class_="author_name")
        if author_name:
            article_statistics['author'] = author_name.string.replace("By ", "")
        else:
            article_statistics['author'] = None

        # grab date and time of publication
        time_field =    soup.find('time')
        date =          time_field.string[0:10]
        time =          time_field.string[11:19]
        date_time =     time_field['datetime']
        weekday =   time_field['datetime'][0:3]
        article_statistics['date'] = date
        article_statistics['time'] = time
        #article_statistics['datetime'] = date_time
        #article_statistics['weekday'] = weekday
        possible_days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for day in possible_days_of_week:
            if day == weekday:
                article_statistics['weekday_is_' + day] = 1
            else:
                article_statistics['weekday_is_' + day] = 0

        # grab data channel
        possible_data_channels = ["lifestyle", "entertainment", "business", "social-media", "tech", "world", "watercooler"]

        data_channel = soup.find(id="body").find(class_="page-header channel")["data-channel"]
        # article_statistics['data_channel'] = data_channel
        for channel in possible_data_channels:
            if channel == data_channel:
                article_statistics['data_channel_is_' + channel] = 1
            else:
                article_statistics['data_channel_is_' + channel] = 0

        # grab {n_tokens_title, n_tokens_content, n_unique_tokens, n_non_stop_words, n_non_stop_unique_tokens}
        title = soup.find(class_="article-header").find(class_="title").string
        article_statistics["title_num_characters"] = len(title)
        article_statistics["title_num_words"]      = len(title.split())

        body = soup.find(class_="article-content")
        body_paragraphs = body.find_all('p', recursive=False)
        #print body_paragraphs
        body_text = ""
        #num_paragraphs = 0
        for paragraph in body_paragraphs:
            #first = True
            for content in paragraph.contents:
                if content.string:
                    body_text += content.string + " "
                    #if first:
                    #    num_paragraphs += 1
                    #    first = False
        body_text = " ".join(body_text.split())    #removes double spaces
        #print body.text
        article_statistics["content_num_characters"] = len(body_text)
        article_statistics["content_num_words"]      = len(body_text.split())
        #article_statistics["content_num_paragraphs"] = num_paragraphs
        article_statistics["content_num_paragraphs"] = len(body_paragraphs)

        # grab topics
        topics = soup.find(class_="article-topics")
        article_statistics["num_topics"] = len(topics.find_all('a'))

        # grab {num_hrefs, num_self_hrefs, num_imgs, num_videos}
        # TODO: this is wrong for the fisrt article, I have 0, but csv_answer is 1
        article_statistics["num_hrefs"] = len(body.find_all('a'))
        self_links = [link for link in body.find_all('a') if link.has_attr('href') and "mashable.com/20" in link['href']]
        article_statistics["num_self_hrefs"] = len(self_links)
        num_imgs = len(soup.find(class_="article-image").find_all('img')) # find_all forces it to be a list, which allows us to call len()
        num_imgs += len(body.find_all('img'))   #this has an issue of double counting the first img on link: http://mashable.com/2013/01/07/downton-abbey-tumblrs/#bB_h2E4PpmqZ
        article_statistics["num_imgs"] = num_imgs


        # grab {global_subjectivity, global_sentiment_polarity, global_rate_positive_words, global_rate_negative_words, rate_positive_words, rate_negative_words, avg_positive_polarity, min_positive_polarity, max_positive_polarity, avg_negative_polarity, min_negative_polarity	 max_negative_polarity, title_subjectivity, title_sentiment_polarity, abs_title_subjectivity, abs_title_sentiment_polarity}
        blob_body_text = TextBlob(body_text)
        article_statistics["global_sentiment_subjectivity"] = blob_body_text.sentiment.subjectivity
        article_statistics["global_sentiment_polarity"]     = blob_body_text.sentiment.polarity
        blob_title = TextBlob(title)
        article_statistics["title_sentiment_subjectivity"]  = blob_title.sentiment.subjectivity
        article_statistics["title_sentiment_polarity"]      = blob_title.sentiment.polarity

        words = body_text.split()
        num_words = len(words)
        positive_words = [word for word in words if TextBlob(word).sentiment.polarity > 0]
        negative_words = [word for word in words if TextBlob(word).sentiment.polarity < 0]
        if len(words) == 0:
            article_statistics["global_rate_positive_words"] = 0
            article_statistics["global_rate_negative_words"] = 0
        else:
            article_statistics["global_rate_positive_words"] = len(positive_words)/float(len(words))
            article_statistics["global_rate_negative_words"] = len(negative_words)/float(len(words))

        if len(positive_words)+len(negative_words) == 0:
            article_statistics["rate_positive_words"]        = 0
            article_statistics["rate_negative_words"]        = 0
        else:
            article_statistics["rate_positive_words"]        = len(positive_words)/float(len(positive_words)+len(negative_words))
            article_statistics["rate_negative_words"]        = len(negative_words)/float(len(positive_words)+len(negative_words))

        positive_word_polarities = [TextBlob(word).sentiment.polarity for word in positive_words]
        negative_word_polarities = [TextBlob(word).sentiment.polarity for word in negative_words]
        if len(positive_word_polarities) == 0:
            article_statistics["avg_positive_polarity"]      = 0
            article_statistics["min_positive_polarity"]      = 0
            article_statistics["max_positive_polarity"]      = 0
        else:
            article_statistics["avg_positive_polarity"]      = sum(positive_word_polarities)/float(len(positive_word_polarities))
            article_statistics["min_positive_polarity"]      = min(positive_word_polarities)
            article_statistics["max_positive_polarity"]      = max(positive_word_polarities)

        if len(negative_word_polarities) == 0:
            article_statistics["avg_negative_polarity"]      = 0
            article_statistics["min_negative_polarity"]      = 0
            article_statistics["max_negative_polarity"]      = 0
        else:
            article_statistics["avg_negative_polarity"]      = sum(negative_word_polarities)/float(len(negative_word_polarities))
            article_statistics["min_negative_polarity"]      = min(negative_word_polarities)
            article_statistics["max_negative_polarity"]      = max(negative_word_polarities)

        condensed_text = ''.join(char for char in body_text if char.isalnum())
        if num_words == 0:
            article_statistics["avg_word_length"] = 0
        else:
            article_statistics["avg_word_length"] = len(condensed_text)/float(num_words)

        self_ref_shares = []
        for link in self_links:
            if "http://mashable.com/20" not in link['href']:
                continue
            ref_url = link['href']
            ref_url = 'http'+ref_url.split('http')[1] #removes everything before http
            try:
                ref_res = requests.get(ref_url)
            except:
                continue
            #ref_res.raise_for_status()
            if ref_res.status_code != requests.codes.ok:
                continue
            ref_soup = bs4.BeautifulSoup(ref_res.text)
            ref_shares = 0
            for platform in social_media_platforms:
                ref_platform_shares = calc_shares(ref_soup.find(class_="social-stub social-share " + platform)['data-shares'])
                ref_shares += ref_platform_shares
            self_ref_shares.append(ref_shares)
        if len(self_ref_shares) == 0:
            article_statistics["self_reference_avg_shares"] = None
            article_statistics["self_reference_min_shares"] = None
            article_statistics["self_reference_max_shares"] = None
        else:
            article_statistics["self_reference_avg_shares"] = sum(self_ref_shares)/float(len(self_ref_shares))
            article_statistics["self_reference_min_shares"] = min(self_ref_shares)
            article_statistics["self_reference_max_shares"] = max(self_ref_shares)

        #things not certain about: num_imgs, num_videos


        #print
        #print
        #print "PRINTING 'key: value' CONTENTS OF DICTIONARY"
        #print "------------------------------------------------------"
        #for key,value in sorted(article_statistics.iteritems()):
        #    num_spaces = 35-len(key)
        #    print "    " + key + ":" + " "*num_spaces + str(value)
        #print "------------------------------------------------------"

        global counter
        counter += 1
        print counter, url_video[0]


        return article_statistics

    except:
        return -1
    #blob = TextBlob("What did you think of the movie? I thought it was boring.")
    #print blob.tags
    #print blob.noun_phrases
    #   for sentence in blob.sentences:
    #print sentence.sentiment.subjectivity


#collect_article_variables(("http://mashable.com/2013/01/09/gopro-videos/",11))

main()

#for i in range(120):
#    test_one_article()

#for i in range(120):
#    thread.start_new_thread(test_one_article, ())

#for i in range(120):
#    t1 = threading.Thread(target=test_one_article)
#    t1.start()
#    t1.join()
