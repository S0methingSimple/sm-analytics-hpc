#!/usr/bin/env python
import os
import sys
import json
import numpy as np
import pandas as pd
from mpi4py import MPI

# 1 Tweet: 1KB
# 1MB: 1000 tweets
# 50MB: 50,000 tweets
# 100GB: 100,000,000 tweets
BUFFER = 2048 # buffer is 2kb (2 tweets length)
# BATCH_SIZE = 1000 * 1024 # 1MB

def extract_tweet_info(tweet):
    # This function attempts to extract date, hour, and sentiment from a tweet.
    # It handles exceptions to ensure the program continues even if data is missing or format is unexpected.
    try:
        doc = tweet.get("doc", {})
        id = doc.get("_id")
        data = doc.get("data", {})
        created_at = data["created_at"].split("T")
        date = created_at[0]
        hour = created_at[1][:2]  # Extracts only the hour part from the timestamp.

        # Attempt to parse sentiment data, which might be stored directly as a float or inside a dictionary.
        sentiment_value = data.get("sentiment")
        sentiment = np.float32(sentiment_value.get("score", 0)) if isinstance(sentiment_value, dict) else np.float32(sentiment_value)
        return (id, date, hour, sentiment)
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
        file_size = os.path.getsize(json_file_path)

        # Calculate chunk size and adjust for last process
        chunk_size = (file_size + size - 1) // size

        # Calculate starting and ending byte offsets (start, end) for each chunk
        offsets = [(rank * chunk_size, min(file_size, (rank + 1) * chunk_size)) for rank in range(size)]
        offsets[-1] = (offsets[-1][0], file_size) # last offset should not be larger than the file size

        # Add buffer to the start of each chunk except the first one
        buffered_offsets = [(max(offset[0] - BUFFER, 0), offset[1], json_file_path) if i != 0 else (offset[0], offset[1], json_file_path) for i, offset in enumerate(offsets)]

    else:
        buffered_offsets = None

    # Distribute data chunks to each process.
    local_offsets = comm.scatter(buffered_offsets, root=0)
    # df_local = pd.DataFrame(columns=['Id', 'Date', 'Hour', 'Sentiment'])
    print(f"Rank {rank} is processing data from {local_offsets[0]} to {local_offsets[1]}")

    # # Read JSON file in batches and process tweets starting from the given offset.
    # for offset in range(local_offsets[0], local_offsets[1], BATCH_SIZE):
    #     with open(local_offsets[2], 'r') as file:
    #         file.seek(offset)
    #         data = file.read(BATCH_SIZE)

    #         # Remove last line if rank is last
    #         if rank == size - 1:
    #             data = data[:data.rfind("\n")]

    #         # for all processes read from the first \n to the last \n
    #         data = data[data.find("\n") + 1 : data.rfind("\n")]

    #         # print last 30 characters of the data
    #         tweets = json.loads(f"[{data[:-1]}]")

    #         if tweets:
    #             # Extract relevant information from each tweet and append to the current DataFrame.
    #             extracted_info = np.array([extract_tweet_info(tweet) for tweet in tweets if extract_tweet_info(tweet) is not None], dtype=[('Id', 'U20'), ('Date', 'U10'), ('Hour', 'U2'), ('Sentiment', 'f4')])
    #             if extracted_info.size > 0:
    #                 df_local = pd.concat([df_local, pd.DataFrame(extracted_info)])

    # Process tweets locally on each process and extract relevant information.
    local_tweets = []
    with open(local_offsets[2], 'r') as file:
        file.seek(local_offsets[0])
        data = file.read(local_offsets[1] - local_offsets[0])

        # Remove last line if rank is last
        if rank == size - 1:
            data = data[:data.rfind("\n")]

        # for all processes read from the first \n to the last \n
        data = data[data.find("\n") + 1 : data.rfind("\n")]

        # print last 30 characters of the data
        local_tweets = json.loads(f"[{data[:-1]}]")

    extracted_info = np.array([extract_tweet_info(tweet) for tweet in local_tweets if extract_tweet_info(tweet) is not None], dtype=[('Id', 'U20'), ('Date', 'U10'), ('Hour', 'U2'), ('Sentiment', 'f4')])
    
    # Convert numpy array to DataFrame if there is data.
    if extracted_info.size > 0:
        df_local = pd.DataFrame(extracted_info)
    else:
        df_local = pd.DataFrame(columns=['Id', 'Date', 'Hour', 'Sentiment'])

    # Print the number of tweets processed by each process.
    print(f"Rank {rank} processed {len(df_local)} tweets")

    # Gather all local DataFrames at the root process.
    gathered_data = comm.gather(df_local, root=0)

    if rank == 0:
        # Combine all DataFrames into a single DataFrame.
        df_all = pd.concat(gathered_data)
        df_all.drop_duplicates(subset='Id', keep='last', inplace=True)  # Remove duplicate tweets based on ID.
        print(f"\nTotal number of tweets: {len(df_all)}")
        most_tweets_hour, happiest_hour, most_tweets_day, happiest_day, most_tweets_hour_count, happiest_hour_score, most_tweets_day_count, happiest_day_score = activity_analysis(df_all)  # Analyze the combined data.
        
        # Print detailed results.
        print("\n")
        print(f"The happiest hour ever: {happiest_hour[1]}:00 on {happiest_hour[0]} with an overall sentiment score of {happiest_hour_score}")
        print(f"The happiest day ever: {happiest_day} was the happiest day with an overall sentiment score of {happiest_day_score}")
        print("\n")
        print(f"The most active hour ever: {most_tweets_hour[1]}:00 on {most_tweets_hour[0]} had the most tweets (#{most_tweets_hour_count})")
        print(f"The most active day ever: {most_tweets_day} had the most tweets (#{most_tweets_day_count})")

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

    # Return the results as a tuple.
    return most_tweets_hour, happiest_hour, most_tweets_day, happiest_day, most_tweets_hour_count, happiest_hour_score, most_tweets_day_count, happiest_day_score


if __name__ == "__main__":
    main()
