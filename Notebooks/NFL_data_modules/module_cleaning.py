def clean_data(df):
    df = df.assign(field_goal = np.where(df['field_goal_result'] == 'made', 1, 0))
    df = df.assign(extra_point = np.where(df['extra_point_result'] == 'good', 1, 0))
    df = df.assign(two_point_conversion = np.where(df['two_point_conv_result'] == 'success', 1, 0))
    
    missing_values = df.isnull().sum()
    mv_ratio = missing_values[missing_values.gt(0)]/len(df)
    list_mv_50 = mv_ratio.loc[mv_ratio.gt(0.5)].index
    
    train_01 = df.drop( columns = list_mv_50).copy()
    
    mv_01 = train_01.isnull().sum()
    list_cols = mv_01.loc[mv_01.gt(0)].index
    
    for col in list_cols:
        train_01[col] = train_01[col].fillna(0)
    
    return train_01

    def process_data(df):
    df['succesful_play'] = (np.where((df['play_type'] == 'pass')
                                        & (df['yards_gained']>= 6.3), 1,
                                   np.where((df['play_type'] == 'run') 
                                            & (df['yards_gained'] >= 4.4), 1, 0)))
    
    df['game_setting'] = df['home_team'].str.cat(df['away_team'])
    dummies = pd.get_dummies(df['game_setting'])
    df = df.join(dummies)
    
    df['posteam_is_home'] = (np.where(df['posteam_type'] == 'home', 1 , 0))
    
    playtype = pd.get_dummies(df['play_type'], prefix= 'pt', drop_first = True )
    df = df.join(playtype)
    
    df['game_half'] = np.where(df['game_half'] == 'Half1', 1, 2)
    
    df['side_of_field_is_hometeam'] = np.where(df['side_of_field'] == df['home_team'], 1, 0)
    
    object_cols = ['home_team','away_team' ,'posteam' ,'posteam_type',
               'defteam', 'side_of_field','game_date','time','yrdln', 'desc','play_type', 'game_setting']
    df = df.drop(columns = object_cols)
    return df 