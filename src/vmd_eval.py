import pandas as pd

def find_fc_max(df:pd.DataFrame):
    df_out = df[['K', 'fc']].groupby('K', as_index=False).agg('max')
    df_out = df_out.rename(columns={'fc':'fc_max'})
    return df_out

def find_cfr_min_max(df:pd.DataFrame):
    df_out = df[['K', 'fc']].groupby('K').agg('max') / df[['K', 'fc']].groupby('K').agg('min')
    df_out = df_out.reset_index().rename(columns={'fc':'CFR (max/min)'})
    return df_out

def find_cfr_roc(df:pd.DataFrame):
    df_out = df[['K', 'fc']].groupby('K').agg('max') / df[['K', 'fc']].groupby('K').agg('min')
    df_out = df_out.reset_index().rename(columns={'fc':'CFR (max/min)'})
    df_out['CFR (RoC)'] = df_out['CFR (max/min)'] / df_out['CFR (max/min)'].shift(1)
    df_out = df_out[['K', 'CFR (RoC)']]
    return df_out

def find_cfr_K(df:pd.DataFrame):
    df_out = df[['K', 'fc']].groupby('K', as_index=False).agg('max').sort_values('K', ascending=True)
    df_out['CFR (K/(K-1))'] = df_out['fc'] / df_out['fc'].shift(1)
    df_out = df_out[['K', 'CFR (K/(K-1))']]
    return df_out

def find_fc_sd(df:pd.DataFrame):
    df_out = df[['K', 'fc']].groupby('K', as_index=False).agg('std')
    df_out = df_out.rename(columns={'fc':'fc_sd'})
    return df_out