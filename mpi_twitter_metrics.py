#!/usr/bin/env python
import sys
import json
import pandas as pd
from mpi4py import MPI

def extract_tweet_info(tweet):
    """Extract date, hour, and sentiment from tweet. Returns None if any info is missing."""
    try:
        doc = tweet.get("doc", {})
        data = doc.get("data", {})
        created_at = data["created_at"].split("T")
        date = created_at[0]
        hour = created_at[1][:2]  # Get only the hour part

        # Extract sentiment safely
        sentiment_value = data.get("sentiment")
        if isinstance(sentiment_value, dict):
            # If sentiment is a dict, extract the numerical value
            sentiment = float(sentiment_value.get("score", 0))
        else:
            # If sentiment is already a numerical value (or string that can be converted)
            sentiment = float(sentiment_value)

        return date, hour, sentiment
    except (KeyError, IndexError, ValueError, TypeError):
        # KeyError for missing keys, IndexError for split issues,
        # ValueError or TypeError for conversion errors
        return None

def process_tweets(tweets):
    """Process a list of tweets to extract relevant info and create a DataFrame."""
    # Extract info from each tweet
    extracted_info = [extract_tweet_info(tweet) for tweet in tweets]
    # Remove None entries
    extracted_info = [info for info in extracted_info if info is not None]
    # Convert to DataFrame
    df = pd.DataFrame(extracted_info, columns=['Date', 'Hour', 'Sentiment'])
    return df

def main():
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running on {size} cores")
        if len(sys.argv) != 2:
            print("Usage: mpiexec -n <num_processes> python mpi_twitter_metrics.py <json_file_path>")
            sys.exit(1)
        json_file_path = sys.argv[1]
        # Read and distribute data only on root
        with open(json_file_path, 'r') as file:
            tweet_data = json.load(file)["rows"]
        # Calculate chunk size and distribute data
        chunks = [tweet_data[i::size] for i in range(size)]
    else:
        chunks = None

    # Scatter chunks of tweet data to all processes
    local_tweets = comm.scatter(chunks, root=0)
    print(f"Rank {rank} is processing {len(local_tweets)} tweets")

    # Process local chunk of tweets
    df_local = process_tweets(local_tweets)

    # Gather all DataFrames on root
    gathered_data = comm.gather(df_local, root=0)

    # Only root process will aggregate and analyze the data
    if rank == 0:
        # Concatenate all DataFrames into a single one
        df_all = pd.concat(gathered_data)
        # Aggregate and analyze the data for activity
        most_tweets_hour = df_all.groupby(['Date', 'Hour']).size().idxmax()
        most_tweets_day = df_all.groupby('Date').size().idxmax()
        
        # Aggregate and analyze the data for sentiment
        happiest_hour_data = df_all.groupby(['Date', 'Hour'])['Sentiment'].mean().idxmax()
        happiest_day_data = df_all.groupby('Date')['Sentiment'].mean().idxmax()
        
        most_tweets_hour_count = df_all.groupby(['Date', 'Hour']).size().max()
        most_tweets_day_count = df_all.groupby('Date').size().max()
        happiest_hour_score = df_all.groupby(['Date', 'Hour'])['Sentiment'].mean().max()
        happiest_day_score = df_all.groupby('Date')['Sentiment'].mean().max()

        print("\n")
        print(f"Total number of tweets: {len(df_all)}\n")
        print(f"The happiest hour ever: {happiest_hour_data[1]}:00 on {happiest_hour_data[0]} with an overall sentiment score of {happiest_hour_score}")
        print(f"The happiest day ever: {happiest_day_data} was the happiest day with an overall sentiment score of {happiest_day_score}")
        print(f"The most active hour ever: {most_tweets_hour[1]}:00 on {most_tweets_hour[0]} had the most tweets (#{most_tweets_hour_count})")
        print(f"The most active day ever: {most_tweets_day} had the most tweets (#{most_tweets_day_count})")

if __name__ == "__main__":
    main()