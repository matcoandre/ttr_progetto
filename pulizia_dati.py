import pandas as pd

def carica_e_pulisci():
    df = pd.read_csv('data/data_oecd.csv')
    df.columns = df.columns.str.strip()
    
    filtered = df[df['Inequality'].str.contains('Tot', na=False, case=False)].copy()
    
    val_col = 'Observation Value'
    if filtered[val_col].isnull().all():
        val_col = 'OBS_VALUE'
        
    filtered['v_clean'] = (
        filtered[val_col]
        .astype(str)
        .str.replace(',', '.')
        .str.extract(r'(\d+\.?\d*)')[0]
    )
    
    filtered['v_clean'] = pd.to_numeric(filtered['v_clean'], errors='coerce')
    filtered = filtered.dropna(subset=['v_clean', 'Country', 'Indicator'])
    
    pivot = filtered.pivot_table(
        index='Country', 
        columns='Indicator', 
        values='v_clean',
        aggfunc='mean'
    )
    
    return pivot.fillna(pivot.mean())

if __name__ == "__main__":
    res = carica_e_pulisci()
    print(res.index.tolist())