import pandas as pd

def has_correct_sequence(s):
    """Function that checks if the sequence is correct.

    Args:
        s (list of ints): This is the journey steps until end variable in the Fingerhut data.

    Returns:
        Bool : This is a boolean that is True if the sequence is correct and False if it is not.
    """
    
    s = list(s)
    temp = s[0]
    for i in range(1, len(s)):
        if s[i] == temp+1:
            temp = s[i]
        elif s[i] == 1:
            temp = 1
        else :
            print('error because temp is ', temp, ' and x[i] is ', s[i])
            print('i is ', i)
            return False
    return True

def correct_sequences(s):
    """Function that corrects the sequences (journey steps until end) in the Fingerhut data.

    Args:
        s (list of ints): This is the journey steps until end variable in the Fingerhut data.

    Returns:
        seq : This is the corrected journey steps until end variable.
    """
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
             'ed_id',
             'journey_steps_until_end']]


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