# Python 3.6.0 |Anaconda 4.3.1 (64-bit)|

import pandas as pd
from datetime import datetime
from os import listdir
from sklearn import preprocessing
from sklearn.decomposition import PCA


def convert_kickoff_to_date(df):
    
    """
    Converts kickoff column from sting to datetime. Supported formats: "%Y-%m-%d", "%m/%d/%Y"
    Returns updated dataframe.
    """
    
    date_value = df.kickoff[0]
    
    if type(date_value) is not pd.tslib.Timestamp:
        
        if '-' in date_value:
            from_str_to_date = lambda date: datetime.strptime(date, "%Y-%m-%d")
            df.kickoff = [from_str_to_date(date) for date in list(df.kickoff)]

        if '/' in date_value:
            from_str_to_date = lambda date: datetime.strptime(date, "%m/%d/%Y")
            df.kickoff = [from_str_to_date(date) for date in list(df.kickoff)]
        
    return df



def scale_stats_per_minute(df):
    
    """
    Scale features in ['CLR','FC','FK','FS','G','GA','GC','GK','PAS','R','S','SAV','TA','Y'] per minute played
    Returns updated df
    """
    
    columns_to_scale = ['CLR', 'FC', 'FK', 'FS', 'G', 'GA', 'GC', 
                    'GK', 'PAS', 'R', 'S', 'SAV', 'TA', 'Y']

    for column in columns_to_scale:
        df[column] = df[column] / df.MINS
        
    return df



def lineup_by_date(df, date, team):
    
    """
    Returns team's lineup for a given date as a dictionary.
    """
    
    
    df = df[df.kickoff == date]
    df = df[df.team == team]
    
    forwards = df[df.position == "forward"].player.tolist()
    defenders = df[df.position == "defender"].player.tolist()
    goalkeepers = df[df.position == "goalkeeper"].player.tolist()
    midfielders = df[df.position == "midfielder"].player.tolist()
    
    field = df.field.tolist()[0]

    lineup = {
        
        "field" : field,
        "team" : team,
        "date" : date,
        "for" : forwards,
        "mid" : midfielders,
        "def" : defenders,
        "goal" : goalkeepers
    }
    
    return lineup



def attack(df, lineup):
    
    """
    Given lineup, assess team's attacking potential.
    Attack is represented with 'G', 'GA', 'S', 'PAS' featues.
    Each statistic is the sum of players' averages appearing in lineup.
    Returns dictionary.
    """

    forwards = {}
    forwards_avg = []

    for player in lineup["for"]:
    
        date = lineup["date"]
        player_df = df[(df.player == player) & (df.kickoff < date)]

        features = ['G', 'GA', 'S', 'PAS']
        player_df = player_df.loc[:, features] 
        player_avg_performance = player_df.mean().to_dict()
    
        forwards_avg.append(player_avg_performance)
        
    forwards_df = pd.DataFrame(forwards_avg).round(4)
    forwards_df.columns = [key + "_f" for key in forwards_df.keys()]
    
    forwards = forwards_df.sum().to_dict()
    
    
    return forwards



def midfield(df, lineup):
    
    """
    Given lineup, assess team's creative potential and ball control.
    Midfield is represented with 'G', 'GA', 'S', 'FC', 'FS', 'PAS' featues.
    Each statistic is the sum of players' averages appearing in lineup.
    Returns dictionary.
    """

    midfielders = {}
    midfielders_avg = []

    for player in lineup["mid"]:
    
        date = lineup["date"]
        player_df = df[(df.player == player) & (df.kickoff < date)]

        main_features = ['G', 'GA', 'S', 'FC', 'FS']
        midfield_features = ['PAS']
        
        player_main_df = player_df.loc[:, main_features]
        player_avg_performance_main = player_main_df.mean().to_dict()
        midfielder_df = player_df[player_df.position == 'midfielder']
        midfielder_df = midfielder_df.loc[:, midfield_features]
        player_avg_performance_midfield = midfielder_df.mean().to_dict()
        midfielder = {**player_avg_performance_main, **player_avg_performance_midfield}
    
        midfielders_avg.append(midfielder)
        
    midfielders_df = pd.DataFrame(midfielders_avg).round(4)
    midfielders_df.columns = [key + "_m" for key in midfielders_df.keys()]
    
    midfielders = midfielders_df.sum().to_dict()
    
    
    return midfielders



def defence(df, lineup):
    
    """
    Given lineup, assess team's defending potential.
    Defence is represented with 'G', 'GA', 'S', 'FC', 'FS', 'TA', 'CLR' featues.
    Each statistic is the sum of players' averages appearing in lineup.
    Returns dictionary.
    """

    defenders = {}
    defenders_avg = []

    for player in lineup["def"]:
    
        date = lineup["date"]
        player_df = df[(df.player == player) & (df.kickoff < date)]

        main_features = ['G', 'GA', 'S', 'FC', 'FS']
        defender_features = ['TA', 'CLR']
        
        player_main_df = player_df.loc[:, main_features]
        player_avg_performance_main = player_main_df.mean().to_dict()
        defender_df = player_df[player_df.position == 'defender']
        defender_df = defender_df.loc[:, defender_features]
        player_avg_performance_defence = defender_df.mean().to_dict()
        defender = {**player_avg_performance_main, **player_avg_performance_defence}
    
        defenders_avg.append(defender)
        
    defenders_df = pd.DataFrame(defenders_avg).round(4)
    defenders_df.columns = [key + "_d" for key in defenders_df.keys()]
    
    defenders = defenders_df.sum().to_dict()
    
    
    return defenders



def goalkeeper(df, lineup):
    
    """
    Given lineup, assess goalkeeper strength.
    Goalkeeper is represented with 'SAV', 'GC', 'GK' featues.
    Returns dictionary.
    """
    player = lineup["goal"][0]
    
    date = lineup["date"]
    player_df = df[(df.player == player) & (df.kickoff < date)]
    
    features = ['SAV', 'GC', 'GK']
    player_df = player_df.loc[:, features] 
    goalkeeper_df = player_df.mean().round(4)
    goalkeeper_df.columns = [key + "_g" for key in goalkeeper_df.keys()]
    
    goalkeeper = goalkeeper_df.to_dict()
    
    return goalkeeper



def team_corners(df, date, team):
    
    """
    Returns a dict with average number of corners a team scored by date.
    Corners are calculated separately as it is considered a team statistic.
    """
    
    team_df = df[(df.team == team) & (df.kickoff < date)]
    corners = team_df["COR"].sum()
    matches = team_df.kickoff.unique().shape[0]
    
    try:
        corners_avg = round(corners/matches ,4)
    except ZeroDivisionError:
        corners_avg = round(corners, 4)
        
    corners_dict = {"COR" : corners_avg}
    
    return corners_dict



def compose_team(df, date, team):
    
    """
    Prepairs statistics for a team for a given date. Retruns DataFrame.
    """
    
    lineup = lineup_by_date(df, date, team)

    team = lineup["team"]
    field = lineup["field"]

    attack_stats = attack(df, lineup)
    midfield_stats = midfield(df, lineup)
    defence_stats = defence(df, lineup)
    goalkeeper_stats = goalkeeper(df, lineup)

    corners = team_corners(df, date, team)

    team_dict = {**attack_stats, **midfield_stats, **defence_stats, **goalkeeper_stats, **corners}
    team_df = pd.DataFrame(team_dict, index = [1])
    team_df.columns = [key + "_" + field for key in team_df.keys()]
    
    return team_df



def compose_fixture(df, date, home, away):
    
    """
    Returns df representing a fixture specified by date, home, and away.
    DataFrame has all 42 statisctis calculated by attack, midfield, defence and goalkeeper functions.
    """
    
    home_df = compose_team(df, date, home)
    away_df = compose_team(df, date, away)
    
    fixture = pd.concat([home_df, away_df], axis = 1)
    fixture["home"] = home
    fixture["away"] = away
    fixture["kickoff"] = date
     
    return fixture



def get_fixtures(df):
    
    """
    Returns df with all fixtures, given raw df. Columns are "home", "away", "kickoff"
    """

    records = df.loc[:, ["team", "field" , "opponent", "kickoff"]]
    
    home_fixs = records[records.field == "home"]
    home_fixs = home_fixs.loc[:, ["team", "opponent", "kickoff"]]
    home_fixs.columns = ["home", "away", "kickoff"]
    
    away_fixs = records[records.field == "away"]
    away_fixs = away_fixs.loc[:, ["opponent", "team", "kickoff"]]
    away_fixs.columns = ["home", "away", "kickoff"]
    
    fixtures = pd.concat([home_fixs, away_fixs], axis = 0)
    fixtures.drop_duplicates(inplace = True)
    fixtures.reset_index(drop = True, inplace = True)

    return fixtures



def transform_data(df):
    
    """
    Get data ready for analysis. Takes raw df and return df with all fixtures, each represented by 42 statistics.
    """
    
    fixtures = get_fixtures(df)

    home = list(fixtures.home)
    away = list(fixtures.away)
    date = list(fixtures.kickoff)

    dataset = pd.DataFrame()

    for home, away, date in zip(home, away, date):
        
        try:
            fixture = compose_fixture(df, date, home, away)
        except IndexError:
            continue
        
        dataset = pd.concat([dataset, fixture])

    dataset = dataset.reset_index(drop = True)
    
    return dataset



def collect_data_from_csvs():
    
    """
    Collects data from all csv files located in data/.
    Retruns dataframe sorted by kickoff date.
    """
    
    path_to_files = "data/players/"
    extension = ".csv"

    files = listdir(path_to_files)
    csv_files = [file for file in files if extension in file]

    data = pd.DataFrame()
    for file in csv_files:

        df = pd.read_csv(path_to_files + file, encoding = "latin1")
        df = convert_kickoff_to_date(df)
        data = pd.concat([data, df])

    data = data.sort_values(by = "kickoff")
    data = data.reset_index(drop = True)
    
    return data



def aggregate_features(data):
    
    """
    Aggregate features across positions, reduce dimensionality to 18 features.
    """

    data['GA_home'] = data['GA_d_home'] + data['GA_f_home'] + data['GA_m_home']
    data['S_home'] =  data['S_d_home'] + data['S_f_home'] + data['S_m_home']
    data['G_home'] =  data['G_d_home'] + data['G_f_home'] + data['G_m_home']
    data['PAS_home'] = data['PAS_f_home'] + data['PAS_m_home']

    data['GA_away'] = data['GA_d_away'] + data['GA_f_away'] + data['GA_m_away']
    data['S_away'] =  data['S_d_away'] + data['S_f_away'] + data['S_m_away']
    data['G_away'] =  data['G_d_away'] + data['G_f_away'] + data['G_m_away']
    data['PAS_away'] = data['PAS_f_away'] + data['PAS_m_away']

    # scale pass statistic per minute
    data["PAS_home"] = [value/90 for value in list(data["PAS_home"])]
    data["PAS_away"] = [value/90 for value in list(data["PAS_away"])]

    columns = [
        'CLR_d_home', 'COR_home', 'SAV_home', 'TA_d_home', 'GC_home', 'GA_home', 'S_home', 'G_home', 'PAS_home',
        'CLR_d_away', 'COR_away', 'SAV_away', 'TA_d_away', 'GC_away', 'GA_away', 'S_away', 'G_away', 'PAS_away',
        'home', 'away', 'kickoff'
    ]

    data = data.loc[:, columns]
    data = data.dropna(axis = 0)
    data = data.sort_values(by = "kickoff")
    data = data.reset_index(drop = True)    

    return data



def add_goals(data):
    
    """
    Adds goals scored for each fixture in data dataframe.
    """

    path_to_files = "data/goals/"
    extension = ".csv"

    files = listdir(path_to_files)
    csv_files = [file for file in files if extension in file]
    
    goals_df = pd.DataFrame()
    
    for file in csv_files:

        goals = pd.read_csv(path_to_files + file, encoding = "latin1")
        goals = goals.loc[:, ["HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
        goals.columns = ["home", "away", "HG", "AG"]
        
        goals_df = pd.concat([goals_df, goals])
    
    data = data.merge(goals_df, how = "left", on = ["home", "away"])
    
    return data



def add_prices(data, price):
    
    """
    Adds prices to dataframe.
    
    Arguments:
    
        data(df): dataframe to which prices are added
        price(str): column name of price column to add
        
    Return:
        
        data(df): updated dataframe
    """
    
    # path to datasets with prices
    path_to_files = "data/goals/"
    extension = ".csv"

    files = listdir(path_to_files)
    csv_files = [file for file in files if extension in file]
    
    df = pd.DataFrame()
    
    if price == "draw":
        price = "BbMxD"
    
    for file in csv_files:
        
        prices = pd.read_csv(path_to_files + file, encoding = "latin1")
        prices = prices.loc[:, ["HomeTeam", "AwayTeam", price]]
        prices.columns = ["home", "away", price]
        
        df = pd.concat([df, prices])
    
    data = data.merge(df, how = "left", on = ["home", "away"])
    
    return data



def scale_and_add_goals_prices(data, price):
    
    """
    Take df after transform_data() is applied and scale data + add goals and prices.
    Arguments:
        data(df) - dataframe to process
        price(str) - name of price column, if draw use just "draw"
    Return:
        data(df) - updated dataframe
    """

    # split dataframe
    fixtures = data.iloc[:, -3:]
    fixtures = fixtures.loc[:, ["home", "away", "kickoff"]]
    data = data.iloc[:, :-3]

    # scale data
    index = data.index
    columns = data.columns
    data = preprocessing.scale(data)
    data = pd.DataFrame(data)
    data.index = index
    data.columns = columns

    # bring fixture info back
    fixtures = fixtures.reset_index()
    data = data.reset_index()
    data = data.merge(fixtures, how = "left", on = ["index"])
    data = data.sort_values(by = "kickoff")
    data = data.reset_index(drop = True)
    data = data.iloc[:, 1:]

    #add goals and prices
    data = data.dropna(axis = 0)
    data = add_goals(data)
    data = add_prices(data, price)
    data = data.sort_values(by = "kickoff")
    data = data.reset_index(drop = True)

    return data


def break_df_by_month(df):
    
    """
    Given df with kickoff column, breaks down fixtures on monthly basis
    Returns dictionary of dataframes with keys being ordinary number of months, i.e. 12 for december, 3 for march.
    """

    dates = pd.date_range("2016-07-01", "2017-06-01", freq = "M")

    dfs = {}

    previous_date = dates[0]

    for date in dates[1:]:

        month = date.month
        month_df = df[(df.kickoff > previous_date) & (df.kickoff < date)]
        
        dfs[month] = month_df.iloc[:, :-1]
        
        previous_date = date
        
    return dfs



def preprocess_data(dfs, target):
    
    """
    For each df in dictionary the number of columns is reduced to number of features.
    Features get scaled using preprocessing.scale()
    
    Arguments:
    dfs(dict) - dictionary with monthly fixtures
    target(str) - market to predict, three markets get handled: "even", "odd", "draw"
    
    Returns updated dictionary.
    """
    
    dfs_ready = {}
    
    for month in dfs:
        
        # copy features
        temp = dfs[month]
        number_of_features = len(temp.columns) - 5
        data = temp.iloc[:, :number_of_features]
        
        # scale data
        index = data.index
        data = preprocessing.scale(data)
        data = pd.DataFrame(data)
        data.index = index
        
        # add target
        if target == "even":
            data["target"] = (1 + temp.HG + temp.AG) % 2
        elif target == "odd":
            data["target"] = (temp.HG + temp.AG) % 2
        elif target == "draw":
            draws = [1 if val else 0 for val in (temp.HG == temp.AG)]
            data["target"] = draws
        else:
            print ("Select either even, odd or draw market.")
        
        dfs_ready[month] = data
    
    return dfs_ready




def pca_4_components(data):
    """
    Prepair dataset for training and prediction.
    Split 42 features set into home and away subsets and apply 2 components PCA to each subset.
    Add draw column and break fixtures by month.

    Arguments:
        data(df) - 42 features dataset after scale_and_add_goals_prices() is applied

    Return:
        dfs(dict) - dictionary with monthly fixtures, ready for classification
    """

    draws = [1 if draw else 0 for draw in data.HG == data.AG]
    draws_df = pd.DataFrame(draws, columns = ["label"])

    home_fts = [feature for feature in data.columns if '_home' in feature ]
    away_fts = [feature for feature in data.columns if '_away' in feature ]

    data_home = data.loc[:, home_fts]
    data_away = data.loc[:, away_fts]

    pca = PCA(n_components = 2)
    pca.fit(data_home)
    home = pca.transform(data_home)
    pca.fit(data_away)
    away = pca.transform(data_away)

    home_df = pd.DataFrame(home, columns = ['Home1', 'Home2'])
    away_df = pd.DataFrame(away, columns = ['Away1', 'Away2'])
    data_pca = pd.concat([home_df, away_df], axis = 1)

    data_pca_kickoff = pd.concat([data_pca, 
                                  draws_df,
                                  pd.DataFrame(data["kickoff"])], axis = 1)

    data_pca_kickoff = convert_kickoff_to_date(data_pca_kickoff)
    dfs = break_df_by_month(data_pca_kickoff)

    return dfs



def apply_pca_to_positions(data, pca_params):

    """
    Transform dataset such that each position subspace gets reduced by pca. 
    
    Arguments:
        data(df): dataset to transform
        pca_params(dict): dictionary with keys being position and values number of pca components
        
        Example: 
            pca_params = {
                "attack" : 1,
                "midfield" : 2, 
                "defence" : 3,
                "goalkeeper" : 2 }
        
    Returns:
        pca_df(df): transformed dataset
        report(dict): dictionary with info about pca transformation
        
        Example:
        report = 
            {'away': {'attack': [4, 1, 0.75],
              'defence': [5, 3, 0.71],
              'goalkeeper': [3, 2, 0.84],
              'midfield': [4, 2, 0.76]},
              
             'home': {'attack': [4, 1, 0.74],
              'defence': [5, 3, 0.71],
              'goalkeeper': [3, 2, 0.82],
              'midfield': [4, 2, 0.77]}}
    """
    
    fields = ["home", "away"]

    tags = {
        "defence" : "_d_",
        "attack" : "_f_",
        "midfield" : "_m_", 
        "goalkeeper" : "_g_"    
    }

    pca_df = pd.DataFrame()
    report = {}

    for field in fields:

        field_dict = {}
        for tag in tags:

                position_tag = tags[tag] + field
                position_components = pca_params[tag]

                pca_columns = [field + "_" + tag + str(component) for component in range(1, position_components + 1)]

                position_cols = [feature for feature in data.columns if position_tag in feature]
                position_df = data[position_cols]

                pca = PCA(n_components = position_components)
                pca.fit(position_df)
                position_pca = pca.transform(position_df)
                position_pca = pd.DataFrame(position_pca, columns = pca_columns)

                pca_df = pd.concat([pca_df, position_pca], axis = 1)

                variance_expl = round(pca.explained_variance_ratio_.sum(), 2)
                n_init_cols = len(position_cols)

                field_dict[tag] = [n_init_cols, position_components, variance_expl]

        report[field] = field_dict

    pca_df = pd.concat([pca_df, data["COR_home"], data["COR_away"] ], axis = 1)
    
    return pca_df, report