#!/usr/bin/env python
import os
import sys
import json
import re
import time
import numpy as np
import pandas as pd
from mpi4py import MPI

# 1 Tweet: 1KB
# 1MB: 1000 tweets
# 50MB: 50,000 tweets
# 100GB: 100,000,000 tweets
# 5.6GB for 100,000,000 tweets (60 bytes/record)

BUFFER = 1024 # buffer is 1kb (1 tweet length)
BATCH_SIZE = 5000 * 1024 # 5MB (5000 tweets)

# def extract_tweet_info(tweet):
#     # This function attempts to extract date, hour, and sentiment from a tweet.
#     # It handles exceptions to ensure the program continues even if data is missing or format is unexpected.
#     try:
#         doc = tweet.get("doc", {})
#         id = doc.get("_id")
#         data = doc.get("data", {})
#         created_at = data["created_at"].split("T")
#         date = created_at[0]
#         hour = created_at[1][:2]  # Extracts only the hour part from the timestamp.

#         # Attempt to parse sentiment data, which might be stored directly as a float or inside a dictionary.
#         sentiment_value = data.get("sentiment")
#         sentiment = np.float32(sentiment_value.get("score", 0)) if isinstance(sentiment_value, dict) else np.float32(sentiment_value)
#         return (id, date, hour, sentiment)
#     except (KeyError, IndexError, ValueError, TypeError):
#         # If there's any error in data extraction, return None.
#         return None
    
def extract_tweets(data):
    # Sample of tweet: {"id":"1406813874750300166","key":[2021,6,21,"1405853521107324933","1055330142313046016","1406813874750300166"],"value":{"text":"Such a cute pupper 😍"},"doc":{"_id":"1406813874750300166","_rev":"1-1a7924e962def1c92ecd08ee1537ea1b","data":{"author_id":"1055330142313046016","conversation_id":"1405853521107324933","created_at":"2021-06-21T03:18:59.000Z","entities":{"mentions":[{"start":0,"end":13,"username":"pup_chazable"}]},"geo":{},"lang":"en","public_metrics":{"retweet_count":0,"reply_count":0,"like_count":0,"quote_count":0},"text":"@pup_chazable Such a cute pupper 😍","sentiment":0.7142857142857143},"matching_rules":[{"id":1406789634793697300,"tag":"Australia-based users or Australia-located tweets, but no re-tweets"}]}},
    # This function attempts to extract date, hour, and sentiment from a tweet using RegEx.
    data = data.split("\n")
    tweets = []
    pattern = r'"id":"(.+?)".+?"created_at":"(\d{4}-\d{2}-\d{2})T(\d{2}).+?"sentiment":(-?\d+\.\d+|0)'
    for tweet in data:
        matches = re.findall(pattern, tweet)
        if matches and len(matches[0]) == 4:
            tweets.append(matches[0])

    return tweets
            


def batch_process_tweets(rank, size, local_offsets, json_file_path):
    df_local = pd.DataFrame(columns=['Id', 'Date', 'Hour', 'Sentiment'])

    # Read JSON file in batches and process tweets starting from the given offset.
    for batch_offset in range(local_offsets[0], local_offsets[1], BATCH_SIZE):
        with open(json_file_path, 'r') as file:
            try:
                file.seek(batch_offset)
                
                # local batch should not be larger than the local offset end
                local_batch = min(BATCH_SIZE, local_offsets[1] - batch_offset)
                data = file.read(local_batch)

                # Remove last line if rank is last
                if rank == size - 1:
                    data = data[:data.rfind("\n")]

                # for all processes read from the first \n to the last \n
                data = data[data.find("\n") + 1 : data.rfind("\n")]

                # print last 30 characters of the data
                tweets = extract_tweets(data)

                if tweets:
                    # Extract relevant information from each tweet and append to the current DataFrame.
                    extracted_info = np.array(tweets, dtype=[('Id', 'u8'), ('Date', 'U10'), ('Hour', 'u4'), ('Sentiment', 'f4')])
                    if extracted_info.size > 0:
                        df_local = pd.concat([df_local, pd.DataFrame(extracted_info)])

            except UnicodeDecodeError:
                # Handle the error (e.g., skip the chunk, log a warning)
                print(f"Error decoding data at offset {batch_offset}")

    # Print the number of tweets processed by each process.
    print(f"Rank {rank} processed {len(df_local)} tweets from {local_offsets[0]} to {local_offsets[1]} bytes.")

    return df_local

def generate_offsets(file_path, size):
    # Calculate chunk size and adjust for last process
    file_size = os.path.getsize(file_path)
    chunk_size = (file_size + size - 1) // size

    # Calculate starting and ending byte offsets (start, end) for each chunk
    offsets = [(rank * chunk_size, min(file_size, (rank + 1) * chunk_size)) for rank in range(size)]
    offsets[-1] = (offsets[-1][0], file_size) # last offset should not be larger than the file size

    return [((max(offset[0] - BUFFER, 0), offset[1]), file_path) if i != 0 else (offset, file_path) for i, offset in enumerate(offsets)]

def activity_analysis(df):
    # Remove duplicate tweets based on ID.
    df.drop_duplicates(subset='Id', keep='last', inplace=True)  
    print(f"\nTotal number of tweets: {len(df)}\n")
    
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
    print(f"The happiest hour ever: {happiest_hour[1]}:00 on {happiest_hour[0]} with an overall sentiment score of {happiest_hour_score}")
    print(f"The happiest day ever: {happiest_day} was the happiest day with an overall sentiment score of {happiest_day_score}\n")

    print(f"The most active hour ever: {most_tweets_hour[1]}:00 on {most_tweets_hour[0]} had the most tweets (#{most_tweets_hour_count})")
    print(f"The most active day ever: {most_tweets_day} had the most tweets (#{most_tweets_day_count})\n")

def main():
    start_time = time.time()  # Record start time

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
       
        # Generate byte offsets for each process to read from the JSON file.
        buffered_offsets = generate_offsets(json_file_path, size)
    else:
        buffered_offsets = None

    # Distribute data chunks to each process.
    local_offsets, json_file_path = comm.scatter(buffered_offsets, root=0)

    # Process tweets in batches and return a local DataFrame.
    df_local = batch_process_tweets(rank, size, local_offsets, json_file_path)

    # Gather all local DataFrames at the root process.
    gathered_data = comm.gather(df_local, root=0)

    if rank == 0:
        # Combine all DataFrames into a single DataFrame.
        df_all = pd.concat(gathered_data)
        activity_analysis(df_all)  # Analyze the combined data.

        end_time = time.time()  # Record end time
        total_time = end_time - start_time
        print(f"Total processing time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
