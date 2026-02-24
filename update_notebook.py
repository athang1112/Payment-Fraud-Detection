import json
import os

notebook_path = r'c:\Users\acer\Payment fraud detection\payment_fraud_detection.ipynb'
backup_path = notebook_path + '.bak'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Backup
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # 1. Update imports
            if 'import pandas as pd' in source and 'import os' not in source:
                cell['source'] = [s.replace('import pandas as pd', 'import pandas as pd\nimport os') for s in cell['source']]
            
            # 2. Update XGBClassifier
            if 'model = XGBClassifier()' in source:
                cell['source'] = [s.replace('model = XGBClassifier()', "model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')") for s in cell['source']]
            
            # 3. Update Prediction Logic with scaling
            if 'new_data = [[2,9839.64,170136.0,160296.36]]' in source and 'scaler.transform' not in source:
                new_source = []
                for s in cell['source']:
                    if 'new_data = [[2,9839.64,170136.0,160296.36]]' in s:
                        new_source.append(s)
                        new_source.append("new_data_scaled = scaler.transform(new_data)\n")
                    elif 'model.predict(new_data)' in s:
                        new_source.append(s.replace('model.predict(new_data)', 'model.predict(new_data_scaled)'))
                    else:
                        new_source.append(s)
                cell['source'] = new_source

    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook updated successfully.")

except Exception as e:
    print(f"Error: {e}")
