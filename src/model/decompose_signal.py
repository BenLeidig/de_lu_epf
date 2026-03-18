# if __name__ == "main":
#     #### #### ANN-Specific Data Adjusting #### ####
#     ann_df = df.copy(deep=True)

#     # Lagging non-price covariates to abide by day-head constraint
#     ## i.e., prices are determined at noon the day before, so we have price data up until last
#     ## hour of the day (since prices are pre-determined), but not exogenous (non-price) data.
#     for col in ann_df.columns:
#         if "price" not in col:
#             ann_df.loc[:, col] = ann_df[col].shift(12)
