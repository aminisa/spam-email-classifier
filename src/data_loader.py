import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    
    df = df.dropna(subset=['text', 'label_num'])
    
    distribution = df['label_num'].value_counts().reset_index()
    distribution.columns = ['Label', 'Count']
    
    print("Dataset Distribution:\n")
    print("0 indicates a real email, 1 indicates a spam email.\n")
    print("| Label | Count   |")
    print("|-------|---------|")
    for index, row in distribution.iterrows():
        print(f"| {row['Label']}     | {row['Count']}    |")

    print("\n")
    
    return df

'''
Purpose: Loads the raw dataset from data/spam_ham_dataset.csv.
'''

'''
Role: Cleans the dataset by removing missing values.
Displays the dataset distribution (spam vs. ham) in a Markdown-like table for easy visualization.
'''

'''
Dependencies: Works independently but provides the cleaned DataFrame (df) as input to preprocess.py and other modules.
'''