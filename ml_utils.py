import json
from pyexpat import model
import anthropic
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os


# Initialize your Claude (Anthropic) API client
client = anthropic.Anthropic(api_key="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")



def reshape_dataframe(df):
    base_columns = ['DATE', 'DESCRIPTION',  'TOTAL']
    base_columns = [col for col in base_columns if col in df.columns]

    category_columns = df['CATEGORY'].unique().tolist()

    reshaped_df = df[base_columns].copy()

    for category in category_columns:
        reshaped_df[category] = np.where(df['CATEGORY'] == category, df['TOTAL'], '')

    column_order = base_columns + sorted(category_columns)
    reshaped_df = reshaped_df[column_order]

    return reshaped_df


def categorize_transactions(CSV, description, out, suggestedCategory, customize):
    transactions = pd.read_csv(CSV)
    print("Loaded transactions from CSV")

    # Check for required columns
    if 'DESCRIPTION' not in transactions.columns or 'TOTAL' not in transactions.columns or 'DATE' not in transactions.columns:
        raise ValueError("Test data must have 'DESCRIPTION', 'DATE', and 'TOTAL' columns.")


    response_df = pd.DataFrame(columns=['DATE', 'DESCRIPTION', 'TOTAL', 'CATEGORY'])
    # Process in batches
    batch_size = 10
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i + batch_size]
        batch_df = pd.DataFrame(batch, )
        # Constructing the JSON string for the current batch
        json_obj = batch_df.to_json(orient='records')  # Ensure it's a valid JSON array
        
        # Create a prompt for Claude AI
        prompt = f"""
            Please help categorize these business bank transactions for accounting purposes. The business trade description is as follows: {description}.
            Read carefully through the description and decide on what business category to put it in. Start with these categories: {suggestedCategory}.
            If something falls outside those categories, use your analytical skills to categorize it, coming up with a Profit & Loss Category for financial statements.
            Ensure that each response item includes the fields: date, description, total, and category.
            The response format should be a JSON array containing one JSON object for each transaction, like this:
            {{
                "DATE": "dd-mm-yyyy",
                "DESCRIPTION": "Transaction description here",
                "TOTAL": "Transaction total here",
                "CATEGORY": "Transaction category here"
            }}
            Below are the transactions:
            {json_obj} 
            
            Please return only the categorized transactions in the specified JSON format.
        """
        
        print(f"Sending prompt to Claude AI for batch {i//batch_size + 1} of {len(transactions)//batch_size + 1}")
        print("\n\n")

        # Send the prompt to Claude AI
        response = claude_ai(prompt)
        print(f"Received response from Claude AI for batch {i//batch_size + 1}")
        
        # Transform the response into a DataFrame
        temp_df = pd.DataFrame(response)
        response_df = pd.concat([response_df, temp_df], ignore_index=True)

        print(f"Transformed Claude AI response to DataFrame for batch {i//batch_size + 1}")
        
       

    reshaped_df = reshape_dataframe(response_df)
    print("Reshaped transactions DataFrame")
    reshaped_df.to_csv(out, index=False)
    print(f"Saved reshaped DataFrame to {out}")

    return out

def transform_response_to_dataframe(response_text):
    response_text = str(response_text)  # Ensure the response is a string
    # Extract the JSON portion from the response text
    # Assuming the response starts with "Here's the categorized transactions in the specified JSON format:"
    json_start = response_text.find('[')  # Find the start of the JSON array
    json_end = response_text.rfind(']') + 1  # Find the end of the JSON array

    # Check if both indices are valid
    if json_start == -1 or json_end == -1 or json_end <= json_start:
        raise ValueError("JSON data not found in the response text.")

    json_str = response_text[json_start:json_end].strip()  # Extract and trim the JSON string

    print("Extracted JSON String:", json_str)  # Debug print

    # Ensure the extracted string is valid JSON
    if not json_str:
        raise ValueError("Extracted JSON string is empty.")

    # Parse the JSON string into a Python list
    try:
        transactions_list = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")

    # Convert the list into a DataFrame
    df = pd.DataFrame(transactions_list)

    return df

def extract_text_from_response(response):
    # Initialize an empty string to store the concatenated text
    full_text = ""
    
    # Iterate through the list and concatenate the text
    for item in response:
        full_text += item.text + "\n"  # Add each text block with a newline
    
    return full_text.strip()

def claude_ai(prompt):
    # Call Claude API to get the recipient's information
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    print("Sent prompt to Claude AI")
    print("\n\n")
    print(message.content)
    print("\n\n")
    # Extract the response from the message
    response = extract_text_from_response(message.content)

 
    return transform_response_to_dataframe(response)


    



def transform_csv(input_csv_path, transform_request, output_dir='outputs'):
    df = pd.read_csv(input_csv_path)

    if 'moneyIn' not in transform_request and 'moneyOut' not in transform_request:
        raise ValueError("The transformation request must include at least one 'moneyIn' or 'moneyOut' mapping.")

    log_df = pd.DataFrame(columns=['date', 'description', 'total', 'Issue'])

    if 'categorizeMoneyIn' in transform_request and transform_request['categorizeMoneyIn']:
        money_in_col = transform_request['moneyIn']
        
        # Convert the column to numeric, coercing errors to NaN
        df[money_in_col] = pd.to_numeric(df[money_in_col], errors='coerce')
        
        # Apply transformation based on the positive value flag
        if transform_request.get('moneyInPositive', False):
            df['moneyIn'] = df[money_in_col].apply(lambda x: x if x > 0.00 else np.nan)
        else:
            df['moneyIn'] = df[money_in_col]

    if 'categorizeMoneyOut' in transform_request and transform_request['categorizeMoneyOut']:
        money_out_col = transform_request['moneyOut']
        
        # Convert the column to numeric, coercing errors to NaN
        df[money_out_col] = pd.to_numeric(df[money_out_col], errors='coerce')
        
        # Apply transformation based on the negative value flag
        if transform_request.get('moneyOutNegative', False):
            df['moneyOut'] = df[money_out_col].apply(lambda x: -x if x < 0.00 else np.nan)
        else:
            df['moneyOut'] = df[money_out_col]

    if 'date' in transform_request:
        df['DATE'] = df[transform_request['date']]
    if 'description' in transform_request:
        description_cols = transform_request['description']
        if isinstance(description_cols, list) and len(description_cols) > 1:
            df['DESCRIPTION'] = df[description_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
        else:
            df['DESCRIPTION'] = df[description_cols[0]]

    money_in_df = False
    money_out_df = False
    if transform_request['categorizeMoneyIn'] : 
        money_in_df = df.dropna(subset=['moneyIn'])[['DATE', 'DESCRIPTION', 'moneyIn']].rename(columns={'moneyIn': 'TOTAL'})
    if transform_request['categorizeMoneyOut']:
        money_out_df = df.dropna(subset=['moneyOut'])[['DATE', 'DESCRIPTION', 'moneyOut']].rename(columns={'moneyOut': 'TOTAL'})

    for _, row in df.iterrows():
        issues = []
        if pd.isna(row['DATE']):
            issues.append("Missing date")
        if pd.isna(row['DESCRIPTION']):
            issues.append("Missing description")
           # Check if both moneyIn and moneyOut are missing when both are required
        money_in_value = row[transform_request['moneyIn']] if transform_request['categorizeMoneyIn'] else None
        money_out_value = row[transform_request['moneyOut']] if transform_request['categorizeMoneyOut'] else None

        if pd.isna(money_in_value) or pd.isna(money_out_value):
            issues.append("Missing both moneyIn and moneyOut values")
        
        if issues:
            log_df = pd.concat([log_df, pd.DataFrame([{
                'date': row.get('date', ''),
                'description': row.get('description', ''),
                'total': row.get('moneyIn', row.get('moneyOut', '')),
                'Issue': ', '.join(issues)
            }])], ignore_index=True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_files = {}

    try: 
        if not money_in_df.empty:
            money_in_path = os.path.join(output_dir, 'money_in.csv')
            money_in_df.to_csv(money_in_path, index=False)
            output_files['money_in'] = money_in_path
    except Exception: 
        print()


    try: 
        if not money_out_df.empty:
            money_out_path = os.path.join(output_dir, 'money_out.csv')
            money_out_df.to_csv(money_out_path, index=False)
            output_files['money_out'] = money_out_path
    except Exception: 
        print()
        

    if not log_df.empty:
        log_path = os.path.join(output_dir, 'log.csv')
        log_df.to_csv(log_path, index=False)
        output_files['log'] = log_path

    return output_files


# Function to process the log file and handle NaN values
def process_log_file(log_filepath):
    # Read the log file into a DataFrame
    log_df = pd.read_csv(log_filepath)
    
    # Ensure that 'Issue' is a column and handle NaN values
    if 'Issue' in log_df.columns:
        # Replace NaN values with default values
        log_df.fillna({'date': '', 'description': '', 'total': 0, 'Issue': ''}, inplace=True)
        
        # Convert the DataFrame to a list of dictionaries
        errors = log_df[['date', 'description', 'total', 'Issue']].to_dict(orient='records')
        
        return errors
    
    return []


# if name == '__main__': run the categorize_transactions function

if __name__ == '__main__':
    categorize_transactions('transactions.csv', 'DESCRIPTION', 'output.csv', 'Predicted Category', True, 'model.pkl')
    transform_csv('transactions.csv', {'categorizeMoneyIn': True, 'categorizeMoneyOut': True, 'moneyIn': 'moneyIn', 'moneyOut': 'moneyOut', 'date': 'date', 'description': ['description1', 'description2']}, 'outputs')
    process_log_file('log.csv')
