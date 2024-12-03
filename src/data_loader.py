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