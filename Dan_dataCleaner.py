import pandas as pd

def fingerhut_data_cleaner(og_df, defs):
    # Inputs: 
    # og_df: This is the original Fingerhut data
    # defs: This is the Event Definitions data frame (also provided by fingerhut)
    # Outputs:
    # df: This is the cleaned Fingerhut data
    
    
    # Dropping duplicate (ignoring journey steps variable)
    df = og_df[['customer_id',
             'account_id',
             'ed_id',
             'event_name',
             'event_timestamp',
             'ed_id']]

    df = df.drop_duplicates()
    df = df.reset_index(drop=True) # re-indexing
    df.head()
    
    # Re-adding journey_steps_until_end (This has to be reset as well)
    
    journey_steps_until_end = []
    step = 1
    for i in range(len(df)-1):
        if df['customer_id'][i] == df['customer_id'][i+1]:
            journey_steps_until_end.append(step)
            step = step + 1
        else:
            step = 1
            journey_steps_until_end.append(step)
    if df['customer_id'][len(df)-2] == df['customer_id'][len(df)-1]:
        journey_steps_until_end.append(step)
        step = step + 1
    else:
        step = 1
        journey_steps_until_end.append(step)
    df['journey_steps_until_end'] = journey_steps_until_end
    
    # Adding a `stage` variable based on the event definitions
    
    df_stages = defs[['event_name', 'stage']]
    
    df = pd.merge(df, df_stages, on ='event_name', how = 'left')
    
    return df
    
    
    
    
    
    
    
    
    
    
    