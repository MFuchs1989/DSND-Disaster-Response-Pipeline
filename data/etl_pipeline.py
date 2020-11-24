import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    '''
    Load in the messages and categories datasets and merge them into one dataframe.

    Args:
        messages_filepath (str): path to the messages.csv file
        categories_filepath (str): path to the categories.csv file

    Returns:
        (Pandas dataframe) merged dataset
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='left')
    return df


def clean_data(df):

    '''
    Clean the merged dataframe.

    Args:
        df (Pandas dataframe): merged data

    Returns:
        (Pandas dataframe) clean data
    '''


    # Expand categories into separate columns
    categories = df.categories.str.split(';', expand=True)
    colnames = categories.iloc[0].str.split('-', expand=True)[0].tolist()
    categories.columns = colnames
    
    # Convert to numeric
    for column in categories.columns:
            categories[column] = categories[column].apply(lambda r: r[-1]).astype(int)            
            
    # Combine original df and expanded categories
    df = pd.concat([df.drop('categories', axis=1), categories], axis=1).drop_duplicates()
    df = df[df.related != 2]
    return df
    

def save_data(df, database_filepath):    

    '''
    Save the clean data into a SQLite database.

    Args:
        df (Pandas dataframe): clean data
        database_filepath (str): path to the SQLite database

    Returns:
        (SQLAlchemy engine) SQLite engine connected to the database
    '''

    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('messages', engine, index=False, if_exists='replace')
    return engine


def main():

    '''
    This file is the ETL pipeline that cleans the data and store them into a SQLite database.

    From this project's root directory, run this file with:
    python data/etl_pipeline.py data/messages.csv data/categories.csv data/messages.db
    '''

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n    DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python etl_pipeline.py '\
              'messages.csv categories.csv '\
              'messages.db')


if __name__ == '__main__':
    main()




