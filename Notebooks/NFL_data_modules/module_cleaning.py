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

def process_data_v1(df):
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

def process_data_v2(df):
    df['succesful_play'] = (np.where((df['play_type'] == 'pass')
                                      & (df['yards_gained']>= 6.3), 1,
                                        np.where((df['play_type'] == 'run') 
                                          & (df['yards_gained'] >= 4.4), 1, 0)))
    
    df['posteam_is_home'] = (np.where(df['posteam_type'] == 'home', 1 , 0))
    
    playtype = pd.get_dummies(df['play_type'], prefix= 'playtype', drop_first = True )
    df = df.join(playtype)
    
    df['side_of_field_is_hometeam'] = np.where(df['side_of_field'] == df['home_team'], 1, 0)
    
    object_cols = ['posteam', 'posteam_type','defteam',
                   'side_of_field','game_date','time','yrdln', 'desc','play_type']
    df = df.drop(columns = object_cols)
    
    return df 

def to_gamesdata(df):
    df['succesful_play'] = (np.where((df['play_type'] == 'pass')
                                      & (df['yards_gained']>= 6.3), 1,
                                 np.where((df['play_type'] == 'run') 
                                          & (df['yards_gained'] >= 4.4), 1, 0)))
    
    df['posteam_is_home'] = (np.where(df['posteam_type'] == 'home', 1 , 0))
    
    playtype = pd.get_dummies(df['play_type'], prefix= 'playtype', drop_first = True )
    df = df.join(playtype)
    
    df['side_of_field_is_hometeam'] = np.where(df['side_of_field'] == df['home_team'], 1, 0)
    
    object_cols = ['posteam', 'posteam_type','defteam',
                   'side_of_field','game_date','time','yrdln', 'desc','play_type']
    df = df.drop(columns = object_cols)
    
    games = df.groupby(['game_id','posteam_is_home', 'home_team', 'away_team'], as_index = False ).sum()
    
    games = games[['game_id', 'posteam_is_home', 'home_team', 'away_team','succesful_play',
               'shotgun', 'no_huddle', 'punt_blocked',
                'first_down_rush', 'first_down_pass',
               'first_down_penalty', 'third_down_converted', 'third_down_failed', 'fourth_down_converted',
               'fourth_down_failed', 'interception', 'safety', 'penalty', 'tackled_for_loss', 
              'fumble_lost', 'incomplete_pass', 'qb_hit', 'sack', 'rush_touchdown' , 'pass_touchdown',
               'return_touchdown','field_goal' ,'extra_point', 'two_point_conversion']]
    
    games_home = games.loc[games['posteam_is_home'] == 1]
    games_away = games.loc[games['posteam_is_home'] == 0]
    
    gamedata = games_home.merge(games_away,how = 'inner', on = games_home['game_id'], 
                            suffixes = ('_hometeam', '_awayteam') )
    
    gamedata = gamedata.drop(columns = ['key_0', 'posteam_is_home_hometeam', 'game_id_awayteam',
                                    'posteam_is_home_awayteam','home_team_awayteam', 'away_team_awayteam' ])
    
    dummies_home = pd.get_dummies(gamedata['home_team_hometeam'], prefix= 'hometeam', drop_first = True)
    dummies_away = pd.get_dummies(gamedata['away_team_hometeam'], prefix= 'awayteam', drop_first = True)
    
    gamedata = gamedata.join([dummies_home, dummies_away])
    
    gamedata = gamedata.drop(columns = ['home_team_hometeam' , 'away_team_hometeam'])
    
    gamedata = gamedata.assign( total_points_hometeam = ((gamedata['rush_touchdown_hometeam'] * 6)
                                                     + (gamedata['pass_touchdown_hometeam'] * 6)
                                                     + (gamedata['return_touchdown_hometeam'] * 6)
                                                     + (gamedata['field_goal_hometeam'] * 3) 
                                                     + (gamedata['extra_point_hometeam'])
                                                     + (gamedata['two_point_conversion_hometeam'] * 2)))
    
    gamedata = gamedata.assign( total_points_awayteam = ((gamedata['rush_touchdown_awayteam'] * 6)
                                                     + (gamedata['pass_touchdown_awayteam'] * 6)
                                                     + (gamedata['return_touchdown_awayteam'] * 6)
                                                     + (gamedata['field_goal_awayteam'] * 3) 
                                                     + (gamedata['extra_point_awayteam'])
                                                     + (gamedata['two_point_conversion_awayteam'] * 2)))
    
    gamedata = gamedata.assign(hometeam_is_winner = np.where(gamedata['total_points_hometeam'] > gamedata['total_points_awayteam'],
                                                        1, 0))
    return gamedata

