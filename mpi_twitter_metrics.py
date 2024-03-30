#!/usr/bin/env python
import sys
import json
import numpy as np
import pandas as pd
from mpi4py import MPI

# Function to extract hour and date from tweet 2021-06-21T03:18:59.000Z
def extract_date_hour(tweet):
    date = tweet["doc"]["data"]["created_at"].split("T")
    return date[0], date[1].split(":")[0]

# Main function
def main():
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running on {size} cores")

    # Check arguments
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: mpiexec -n <num_processes> python mpi_twitter_metrics.py <json_file_path>")
        sys.exit(1)

    json_file_path = sys.argv[1]

    # Read JSON file
    if rank == 0:
        with open(json_file_path, 'r') as file:
            tweet_data = json.load(file)["rows"]

        # Divide the data into chunks
        chunk_size = len(tweet_data) // size
        chunked_tweets = [tweet_data[i:i+chunk_size] for i in range(0, len(tweet_data), chunk_size)]
    else:
        chunked_tweets = None

    # Scatter tweet data across processes
    local_tweets = comm.scatter(chunked_tweets, root=0)

    # Print how many tweets are being processed
    print(f"Rank {rank} is processing {len(local_tweets)} tweets")

    # Extract date-hours from tweets
    date_hour = [extract_date_hour(tweet) for tweet in local_tweets]

    # Combine days and hours to form unique date-hour pairs
    date_hour_count = np.unique(date_hour, axis=0, return_counts=True)

    # Gather date_hour_count from all processes
    all_date_hour = comm.gather(date_hour_count, root=0)

    # Process results on root process
    if rank == 0:

        # Extract date-hour pairs and counts from all_date_hour
        date_hour_counts = np.concatenate([arr[0] for arr in all_date_hour])
        counts = np.concatenate([arr[1] for arr in all_date_hour])
        
        # Create a DataFrame for aggregation
        df = pd.DataFrame(date_hour_counts, columns=['Date', 'Hour'])
        df['Count'] = counts

        # Show the first 5 rows of the dataframe
        print(df.head())

        # Aggregate counts for each date-hour pair and date
        aggregated_hour_counts = df.groupby(['Date', 'Hour'])['Count'].sum()
        aggregated_date_counts = df.groupby('Date')['Count'].sum()

        # Find the most active date and hour
        most_active_hour = aggregated_hour_counts.idxmax()
        most_active_hour_count = aggregated_hour_counts.loc[most_active_hour]

        most_active_date =  aggregated_date_counts.idxmax()
        most_active_date_count = aggregated_date_counts.loc[most_active_date]

        # Print most active date-hour pair
        print(f"The most active hour: {most_active_hour[1]}:00 on {most_active_hour[0]} had the most tweets (#{most_active_hour_count})")
        print(f"The most active date: {most_active_date} had the most tweets (#{most_active_date_count})")


# Execute main function
if __name__ == "__main__":
    main()