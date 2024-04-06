#!/usr/bin/env python
import sys
import json
import numpy as np
import pandas as pd
from mpi4py import MPI

def extract_tweet_info(tweet):
    # This function attempts to extract date, hour, and sentiment from a tweet.
    # It handles exceptions to ensure the program continues even if data is missing or format is unexpected.
    try:
        doc = tweet.get("doc", {})
        data = doc.get("data", {})
        created_at = data["created_at"].split("T")
        date = created_at[0]
        hour = created_at[1][:2]  # Extracts only the hour part from the timestamp.

        # Attempt to parse sentiment data, which might be stored directly as a float or inside a dictionary.
        sentiment_value = data.get("sentiment")
        sentiment = np.float32(sentiment_value.get("score", 0)) if isinstance(sentiment_value, dict) else np.float32(sentiment_value)
        return (date, hour, sentiment)
    except (KeyError, IndexError, ValueError, TypeError):
        # If there's any error in data extraction, return None.
        return None

def main():
    comm = MPI.COMM_WORLD  # Initialize MPI environment.
    rank = comm.Get_rank()  # Get the rank of the process within the communicator.
    size = comm.Get_size()  # Get the total number of processes in the communicator.

    if rank == 0:
        print(f"Running on {size} cores")
        if len(sys.argv) != 2:
            # If the filename is not provided, terminate the program with an error message.
            print("Usage: mpiexec -n <num_processes> python mpi_twitter_metrics.py <json_file_path>")
            sys.exit(1)
        json_file_path = sys.argv[1]  # Path to the JSON file containing tweets.
        with open(json_file_path, 'r') as file:
            tweet_data = json.load(file)["rows"]  # Load tweet data from the JSON file.
        chunks = [tweet_data[i::size] for i in range(size)]  # Split data into chunks for each process.
    else:
        chunks = None

    # Distribute data chunks to each process.
    local_tweets = comm.scatter(chunks, root=0)
    print(f"Rank {rank} is processing {len(local_tweets)} tweets")

    # Process tweets locally on each process and extract relevant information.
    extracted_info = np.array([extract_tweet_info(tweet) for tweet in local_tweets if extract_tweet_info(tweet) is not None], dtype=[('Date', 'U10'), ('Hour', 'U2'), ('Sentiment', 'f4')])
    
    # Convert numpy array to DataFrame if there is data.
    if extracted_info.size > 0:
        df_local = pd.DataFrame(extracted_info)
    else:
        df_local = pd.DataFrame(columns=['Date', 'Hour', 'Sentiment'])

    # Gather all local DataFrames at the root process.
    gathered_data = comm.gather(df_local, root=0)

    if rank == 0:
        # Combine all DataFrames into a single DataFrame.
        df_all = pd.concat(gathered_data)
        print(f"\nTotal number of tweets: {len(df_all)}\n")
        activity_analysis(df_all)  # Analyze the combined data.

def activity_analysis(df):
    # This function calculates and prints out various statistics about tweet activity.
    # Determine the hour and day with the most tweets and the highest average sentiment.
    most_tweets_hour = df.groupby(['Date', 'Hour']).size().idxmax()
    happiest_hour = df.groupby(['Date', 'Hour'])['Sentiment'].mean().idxmax()
    most_tweets_day = df.groupby('Date').size().idxmax()
    happiest_day = df.groupby('Date')['Sentiment'].mean().idxmax()

    # Calculate the actual maximum values for tweet counts and sentiment scores.
    most_tweets_hour_count = df.groupby(['Date', 'Hour']).size().max()
    happiest_hour_score = df.groupby(['Date', 'Hour'])['Sentiment'].mean().max()
    most_tweets_day_count = df.groupby('Date').size().max()
    happiest_day_score = df.groupby('Date')['Sentiment'].mean().max()

    # Print detailed results.
    print("\n")
    print(f"Total number of tweets: {len(df)}\n")
    print(f"The happiest hour ever: {happiest_hour[1]}:00 on {happiest_hour[0]} with an overall sentiment score of {happiest_hour_score}")
    print(f"The happiest day ever: {happiest_day} was the happiest day with an overall sentiment score of {happiest_day_score}")
    print("\n")
    print(f"The most active hour ever: {most_tweets_hour[1]}:00 on {most_tweets_hour[0]} had the most tweets (#{most_tweets_hour_count})")
    print(f"The most active day ever: {most_tweets_day} had the most tweets (#{most_tweets_day_count})")


if __name__ == "__main__":
    main()
