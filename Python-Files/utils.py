import pandas as pd

def correct_sequences(s):
    seq = list(s)
    temp = s[0]
    for i in range(1, len(seq)):
        # if 1 then start again
        if seq[i] == 1:
            temp = 1
        elif seq[i] == temp+1:
            temp = seq[i]
        else :
            seq[i] = temp+1
            temp = seq[i]
    return seq

def fingerhut_data_cleaner(og_df, defs):
    """
    Function to drop duplicates, reindex journey steps, convert timestamps, and merge event definitions.

    args:
     - og_df: This is the original Fingerhut data
     - defs: This is the Event Definitions data frame (also provided by fingerhut)
    
    output:
     - df: This is the cleaned Fingerhut data
    """
    # Dropping duplicate (ignoring journey steps variable)
    df = og_df[['customer_id',
             'account_id',
             'ed_id',
             'event_name',
             'event_timestamp',
             'ed_id']]

    df = df.drop_duplicates()
    df = df.reset_index(drop=True) # re-indexing

    # Re-adding journey_steps_until_end (Axel's way)
    j_steps = df['journey_steps_until_end']
    s_corrected = correct_sequences(j_steps)
    df['journey_steps_until_end'] = s_corrected

    # Convert event_timestamps to datetime objects
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], format='mixed')
    
    # Adding a `stage` variable based on the event definitions
    df_stages = defs[['event_name', 'stage']]
    
    df = pd.merge(df, df_stages, on ='event_name', how = 'left')
    
    return df

def add_n_accounts(df):
    """
    Adds a new column representing the number of accounts each customer has.
    """
    account_counts = df.groupby('customer_id').size().reset_index(name='n_accounts')

    return pd.merge(df, account_counts, on='customer_id')

def add_has_discover(df):
    """
    Adds a new column representing whether a customer has gone through the 'Discover' phase.
    """
    discover_customers = df.groupby('customer_id')['stage'].apply(lambda x: 'Discover' in x.values).reset_index(name='has_discover')

    return pd.merge(df, discover_customers, on='customer_id')

def add_has_first_purchase(df):
    """
    Adds a new column representing whether a customer has made their first purchase. 
    """
    first_purchase_customers = df.groupby('customer_id')['stage'].apply(lambda x: 'First Purchase' in x.values).reset_index(name='has_first_purchase')

    return pd.merge(df, first_purchase_customers, on='customer_id')