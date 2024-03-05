import pandas as pd
import numpy as np

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
             'journey_steps_until_end',
             'journey_id',
             'milestone_number',]]
    
    # Filling in missing milestone numbers with 0
    df.loc[:,['milestone_number']] = df['milestone_number'].copy().fillna(0)

    df = df.drop_duplicates(subset=['customer_id', 'account_id', 'ed_id', 'event_name', 'event_timestamp'])
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
    
    # Setting positive values for account_ids
    df['account_id'] = remove_if(df, 'account_id')

    # Setting positive values for customer_ids
    df['customer_id'] = remove_if(df, 'customer_id')
    
    return df

def add_n_accounts(df):
    """
    Adds a new column representing the number of accounts each customer has.    
    """
    # Counting the unique number of account_ids for each customer_id
    unique_account_counts = df.groupby('customer_id')['account_id'].nunique().reset_index(name='n_accounts')

    # Merging the unique account counts back into the original dataframe
    return pd.merge(df, unique_account_counts, on='customer_id')

def add_has_discover(df):
    """
    Adds a new column representing whether a customer has gone through the 'Discover' phase.
    """
    discover_customers = df.groupby('customer_id')['stage'].apply(lambda x: 'Discover' in x.values).reset_index(name='has_discover')

    return pd.merge(df, discover_customers, on='customer_id')

def add_has_first_purchase(df):
    """
    Adds a new column representing whether a customer has made their first purchase. 
    
    WE ARE ADDING THE BOOLEAN VALUE WITHOUT TAKING CARE IF IT WAS A MILESTONE OR NOT I.E
    WE ARE NOT TAKING INTO ACCOUNT THAT 'FIRST PURCHASE' COULD BE JUST BROWSING PRODUCTS AND NOT
    ACTUALLY BUYING SOMETHING

    """
    first_purchase_customers = df.groupby('customer_id')['stage'].apply(lambda x: 'First Purchase' in x.values).reset_index(name='has_first_purchase')

    return pd.merge(df, first_purchase_customers, on='customer_id')

def split_sequences(df):
    """Function that given the dataframe, returns a list of lists with the sequences of events

    Args:
        df (dataframe): The dataframe with the data

    Returns:
        result: A list of lists with the sequences of events
    """
    result = []
    current_sequence = []
    
    for idx, step in enumerate(df['journey_steps_until_end']):
        if step == 1:
            # If the list is not empty, i.e. we have a new 1 in
            # the journey we append the current sequence to the result
            if current_sequence:
                result.append(current_sequence)
            current_sequence = [df['ed_id'].iloc[idx]]
        else:
            current_sequence.append(df['ed_id'].iloc[idx])
    
    # In case the last sequence is not empty we append the remaining sequence
    if current_sequence:
        result.append(current_sequence)
    
    return result

def remove_if(df, col_name):
    """Function that removes the negative number of a customer_id

    Args:
        a (str): number of a customer_id

    Returns:
        str : number of a customer_id without the negative sign
    """
    values = df[col_name].apply(lambda x : (-1)*x if x < 0 else x).astype('int64')
    return values

def number_journeys_and_max(cus_df):
    """Function to check the number of journeys in a sequence

    Args:
        seq (list): List of values

    Returns:
        int: Number of journeys in the sequence
    """
    j_steps = cus_df['journey_steps_until_end']
    ones = [i for i, x in enumerate(j_steps) if x == 1]
    return len(ones), max(j_steps)

def has_discover(cust_df):
    """Function to check if a sequence has the discovery event

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        bool: True if the sequence has the discovery event, False otherwise
    """
    return 'Discover' in list(cust_df['stage'])

def number_accounts(cust_df):
    """Function to add the number of accounts to the dataset

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        pd.DataFrame: Dataset with the number of accounts in a new column
    """
    return cust_df['account_id'].nunique()

def has_more_one_journey(j_steps):
    """Function to check if a sequence has repeated values

    Args:
        seq (list): List of values

    Returns:
        bool: True if there are repeated values, False otherwise
    """
    return len(j_steps) != len(set(j_steps))

def most_repeated_event(cust_df):
    """Function that returns the most repeated event in a sequence

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        str: The most repeated event in the sequence
    """
    return cust_df['ed_id'].mode()[0]

def average_length_seq(cust_df):
    """Function to add the average length of the sequences to the dataset

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        pd.DataFrame: Dataset with the average length of the sequences in a new column
    """
    new_df = cust_df.copy()
    # Split the sequences
    sequences = split_sequences(new_df)
    return np.mean([len(seq) for seq in sequences])

def has_prospecting(cust_df):
    """Function to check if a sequence has the prospecting event

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        bool: True if the sequence has the prospecting event, False otherwise
    """
    evnts = list(cust_df['ed_id'])
    return 20 in evnts or 21 in evnts or 24 in evnts

def has_pre_application(cust_df):
    """Function to check if a sequence has the pre-application event

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        bool: True if the sequence has the pre-application event, False otherwise
    """
    return 22 in list(cust_df['ed_id'])

def initial_device(cust_df):
    """Function to get the initial device of a customer
    """
    events = set(cust_df['event_name'])
    phone = ['phone' in event for event in events]
    web = ['web' in event for event in events]
    
    if np.array(phone).any() and np.array(web).any():
        return 3
    elif np.array(phone).any():
        return 1
    elif np.array(web).any():
        return 2
    
def has_approved(cust_df):
    """Function to check if a sequence has the approved event

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        bool: True if the sequence has the approved event, False otherwise
    """
    x = set(cust_df['ed_id'])
    return 15 in x or 12 in x

def get_first_n_events(cust_df, n = 10):
    """Function that returns the first 10 events of a sequence

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        list: The first 10 events of the sequence, padded with np.nan if necessary
    """
    events = cust_df['ed_id'].head(n).tolist()
    # Pad with np.nan if the sequence has fewer than 10 events
    events += [0] * (n - len(events))
    return np.array(events)

def get_time_since_last_event(cust_df, n=10):
    cust_df = cust_df.head(n)
    x = cust_df.groupby(['customer_id', 'journey_id'])['event_timestamp'].diff()
    x = x.fillna(pd.Timedelta(seconds=0))
    x = x.dt.total_seconds()
    x = x.tolist() + [0] * (n - len(x))
    return np.array(x)

def which_milestones(cust_df):
    """Function that returns in a tuple in the following sequence the next statemens:
    - If the customer has applied for credit and it has been approved (milestone 1)
    - If the customer has first purchase (milestone 2)
    - If the customer has account activitation (milestone 3)
    - If the customer has downpayment received (milestone 4)
    - If the customer has downpayment cleared (milestone 5)
    - If the customer has order shipped (milestone 6)

    Args:
        cust_df (_type_): _description_
    """
    milestones = set(cust_df['milestone_number'].unique())
    max_milestone = max(milestones)
    return (1 in milestones, 2 in milestones, 3 in milestones, 4 in milestones, 5 in milestones, 6 in milestones), max_milestone

# Functions for time
def get_idxs(cust_df, stage, milestone = -1):
    """Function to get the indexes of a certain stage

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        list: List with the indexes of a certain stage
    """
    if milestone != -1:
        return list(cust_df[cust_df['milestone_number'] == milestone].index)
    
    return list(cust_df[cust_df['stage'] == stage].index)

def time_in_discover(cust_df, seconds_differences):
    """Function to calculate the time between events

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        list: List with the time between events
    """
    idxs = get_idxs(cust_df, 'Discover')
    
    time_in = []
    for idx in idxs:
        if idx + 1 < len(seconds_differences):
            time_in.append(seconds_differences[idx + 1])
        else:
            time_in.append(0)
    return sum(time_in)

def time_in_apply(cust_df, seconds_differences):
    """Function to calculate the time between events

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        list: List with the time between events
    """
    idxs = get_idxs(cust_df, 'Apply for Credit')
    
    time_in = []
    for idx in idxs:
        if idx + 1 < len(seconds_differences):
            time_in.append(seconds_differences[idx + 1])
        else:
            time_in.append(0)
    return sum(time_in)

def time_reach_milestone1(cust_df, seconds_differences):
    """Function to calculate the time between events

    Args:
        cust_df (pd.DataFrame): Dataset of a certain customer (not all the dataset, just one customer)

    Returns:
        list: List with the time between events
    """
    idxs = get_idxs(cust_df, 'Apply for Credit', 1)
    
    # sum all the times before the milestone
    return sum(seconds_differences[1:idxs[0]+1])

def group_by_approach(cust_df):
    cust_df = cust_df.reset_index(drop=True)
    # applying all the functions to get the data
    num_journeys, max_journey = number_journeys_and_max(cust_df)
    discover = has_discover(cust_df)
    numb_accs = number_accounts(cust_df)
    more_one_journey = has_more_one_journey(cust_df['journey_steps_until_end'])
    repeated_event = most_repeated_event(cust_df)
    avg_length_journey = average_length_seq(cust_df)
    has_pros = has_prospecting(cust_df)
    pre_applic = has_pre_application(cust_df)
    device = initial_device(cust_df)
    x = cust_df['event_timestamp'].diff().dt.total_seconds().tolist()
    time_disc = time_in_discover(cust_df, x)
    time_apply = time_in_apply(cust_df, x)
    # time_milestone1 = time_reach_milestone1(cust_df, x)
    
    milestones, max_milestone = which_milestones(cust_df)
    
    # Creating the new data frame
    new_df = pd.DataFrame({'num_journeys': num_journeys,
                           'max_journey': max_journey,
                           'discover': discover, 
                           'number_accounts': numb_accs,
                           'one_more_journey': more_one_journey,
                           'most_repeated_event': repeated_event,
                           'average_length_seq': avg_length_journey,
                           'approved_credit': milestones[0],
                           'first_purchase': milestones[1],
                           'account_activitation': milestones[2],
                           'downpayment_received': milestones[3],
                           'downpayment_cleared': milestones[4],
                           'order_ships': milestones[5],
                           'max_milestone': max_milestone,
                            'has_prospecting': has_pros,
                            'has_pre_application': pre_applic,
                            'initial_device': device,
                            'time_in_discover': time_disc,
                            'time_in_apply': time_apply,
                            #'time_reach_milestone_1': time_milestone1,
                           'index':[0]})
    return new_df    

def get_classification_dataset(data, event_defs, n_events = 10):
    """This function is the one that gives you the final dataset with the features and the target variable

    Args:
        data (_type_): the original dataset (without any cleaning or anything like that)
        event_defs (_type_): the event definitions dataset
        n_events (int, optional): Number of events in the last column. Defaults to 10.

    Returns:
        df : The final dataset with the features and the target variable
    """
    
    
    df = fingerhut_data_cleaner(data, event_defs)
    # drop the promotion_created event
    idxs = list(df[df['event_name'] == 'promotion_created'].index)
    df.drop(idxs, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Grouping by the customer id and gathering the data
    new_df = df.groupby('customer_id').apply(group_by_approach)
    new_df.drop(columns=['index'], inplace=True)
    
    # Adding the first n events
    x = list(df.groupby('customer_id').apply(get_first_n_events, n = n_events))
    new_df['first_' + str(n_events) + '_events'] = x
    
    # Adding the time of this first n events
    x = list(df.groupby('customer_id').apply(get_time_since_last_event, n = n_events))
    new_df['time_since_last_event'] = x
    
    return new_df