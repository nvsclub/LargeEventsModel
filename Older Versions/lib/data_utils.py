import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FIELD_SIZE = {'x':1.05, 'y':0.68}

VAEP_FEATURE_SET = [
    'absolute_sec', 'period', 
    'x', 'y', 'end_x', 'end_y', 
    'goal', 'own_goal', 'assist', 'key_pass', 
    'counter_attack', 'left', 'right', 'head', 'direct', 'indirect', 
    'dangerous_ball_lost', 'blocked', 'high', 'low', 'interception', 
    'clearance', 'opportunity', 'feint', 'missed ball', 'sliding_tackle', 
    'anticipated', 'anticipation', 'red', 'yellow', 'second_yellow', 
    'through', 'lost', 'neutral', 'won', 'accurate', 
    'not_accurate', 'subtype_id', 'type_id', 
    'possession_type_id', 
    'possession_team_id', 'previous_action_type_id_1', 
    'previous_action_is_same_team_1', 'previous_action_is_same_possession_1', 
    'previous_action_is_same_player_1', 'previous_action_x_1', 'previous_action_y_1', 
    'previous_action_time_since_1', 'previous_action_x_displacement_1', 
    'previous_action_type_id_2', 'previous_action_is_same_team_2', 
    'previous_action_is_same_possession_2', 'previous_action_is_same_player_2', 
    'previous_action_x_2', 'previous_action_y_2', 'previous_action_time_since_2', 
    'previous_action_x_displacement_2', 'possession_start_is_same_team', 
    'possession_start_action_x', 'possession_start_action_y', 
    'possession_start_time_since', 'possession_start_x_displacement', 
    'start_distance_to_goal', 'start_angle_to_goal', 'end_distance_to_goal', 
    'end_angle_to_goal', 'intent_progressive', 'shot_assist'
]

XG_FEATURE_SET = [
    'subtype_id', 'x', 'y', 
    'left', 'right', 'head',
    'previous_action_type_id_1', 'previous_action_is_same_team_1',
    'previous_action_is_same_player_1', 'previous_action_x_1', 'previous_action_y_1', 
    'previous_action_time_since_1', 'previous_action_x_displacement_1', 
    'possession_type_id', 
    'possession_start_action_x', 'possession_start_action_y', 
    'possession_start_time_since', 'possession_start_x_displacement', 
    'start_distance_to_goal', 'start_angle_to_goal', 'end_distance_to_goal', 'end_angle_to_goal'
]

SUBEVENT_TYPE_MAP = {
    'air_duel': 1,
    'ground_attacking_duel': 2,
    'ground_defending_duel': 3,
    'ground_loose_ball_duel': 4,
    'foul': 5,
    'hand_foul': 6,
    'late_card_foul': 7,
    'out_of_game_foul': 8,
    'protest': 9,
    'simulation': 10,
    'time_lost_foul': 11,
    'violent_foul': 12,
    'corner': 13,
    'free_kick': 14,
    'free_kick_cross': 15,
    'goal_kick': 16,
    'penalty': 17,
    'throw_in': 18,
    'goalkeeper_leaving_line': 19,
    'acceleration': 20,
    'clearance': 21,
    'touch': 22,
    'cross': 23,
    'hand_pass': 24,
    'head_pass': 25,
    'high_pass': 26,
    'launch': 27,
    'simple_pass': 28,
    'smart_pass': 29,
    'reflexes': 30,
    'save_attempt': 31,
    'free_kick_shot': 32,
    'shot': 33,
}

EVENT_TYPE_MAP = {
    'duel': 1,
    'foul': 2,
    'free_kick': 3,
    'goalkeeper_leaving_line': 4,
    'offside': 5,
    'others_on_the_ball': 6,
    'pass': 7,
    'interruption': 8,
    'save_attempt': 9,
    'shot': 10,
}


def load_data(path):
    df = pd.read_csv(path)
    df = df.fillna(0)
    return df

def compute_features(df):

    df['subtype_id'] = df['subtype_name'].map(SUBEVENT_TYPE_MAP)
    df['type_id'] = df['type_name'].map(EVENT_TYPE_MAP)
    df = df.dropna(subset=['subtype_id']).copy()

    df['player_is_next_1'] = np.where((df.type_name == 'pass') & (df.team_name == df.team_name.shift(-1)), df.player_name.shift(-1), '')
    df['receiving_player_name'] = np.where((df.type_name == 'pass') & (df.team_name == df.team_name.shift(-2)), df.player_name.shift(-2), '')
    df.loc[df['receiving_player_name'] == '', 'receiving_player_name'] = df.loc[df['receiving_player_name'] == '', 'player_is_next_1']

    # A possession starts with a pass and ends when a successful pass from the opponent is made
    # or when the ball goes out of play
    start_new_possession = (((df['type_name'] == 'pass') * df['accurate'] + (df['type_name'] == 'free_kick')) * df.team_id).replace(0, np.NaN).ffill()
    start_new_possession = (start_new_possession != start_new_possession.shift(1)).cumsum()
    start_new_possession = start_new_possession + ((df['type_name'] == 'interruption') | (df['type_name'] == 'foul')).shift(1).fillna(0).cumsum()
    df['possession_id'] = start_new_possession
    df['possession_type_name'] = (df['possession_id'].diff(1).fillna(1) * df['type_name']).replace('', np.NaN).ffill()
    df['possession_type_id'] = df['possession_type_name'].map(EVENT_TYPE_MAP)
    df['possession_team_id'] = (df['possession_id'].diff(1).fillna(1) * df['team_id']).replace(0, np.NaN).ffill()
    df['possession_start_time'] = (df['possession_id'].diff(1).fillna(1) * df['absolute_sec']).replace(0, np.NaN).ffill()

    for i in range(1, 3):
        df[f'previous_action_type_id_{i}'] = df['type_id'].shift(i)
        df[f'previous_action_is_same_team_{i}'] = (df['team_id'] == df['team_id'].shift(i)).astype(int)
        df[f'previous_action_is_same_possession_{i}'] = (df['possession_id'] == df['possession_id'].shift(i)).astype(int)
        df[f'previous_action_is_same_player_{i}'] = (df['player_id'] == df['player_id'].shift(i)).astype(int)
        df[f'previous_action_x_{i}'] = abs((100 * (1-df[f'previous_action_is_same_team_{i}'])) - df['x'].shift(i))
        df[f'previous_action_y_{i}'] = abs((100 * (1-df[f'previous_action_is_same_team_{i}'])) - df['y'].shift(i))
        df[f'previous_action_time_since_{i}'] = df['absolute_sec'] - df['absolute_sec'].shift(i)
        df[f'previous_action_x_displacement_{i}'] = df['x'] - df[f'previous_action_x_{i}']

    df['possession_start_is_same_team'] = (df['possession_team_id'] == df['team_id']).astype(int)
    df['possession_start_action_x'] = (df['possession_id'].diff(1).fillna(1) * df['x']).replace(0, np.NaN).ffill()
    df['possession_start_action_y'] = (df['possession_id'].diff(1).fillna(1) * df['y']).replace(0, np.NaN).ffill()
    df['possession_start_time_since'] = df['absolute_sec'] - df['possession_start_time']
    df['possession_start_x_displacement'] = df['x'] - df['possession_start_action_x']

    df['start_distance_to_goal'] = np.sqrt(((df['x'] - 100) * FIELD_SIZE['x'])**2 + ((df['y'] - 50) * FIELD_SIZE['y'])**2)
    df['start_angle_to_goal'] = abs(np.arctan2((df['y'] - 50) * FIELD_SIZE['y'], (df['x'] - 100) * FIELD_SIZE['x']))
    df['end_distance_to_goal'] = np.sqrt(((df['end_x'] - 100) * FIELD_SIZE['x'])**2 + ((df['end_y'] - 50) * FIELD_SIZE['y'])**2)
    df['end_angle_to_goal'] = abs(np.arctan2((df['end_y'] - 50) * FIELD_SIZE['y'], (df['end_x'] - 100) * FIELD_SIZE['x']))

    df['intent_progressive'] = ((df['type_name'] == 'pass') * (df['end_distance_to_goal'] < df['start_distance_to_goal'])).astype(int)
    
    df['shot_assist'] = ((df['type_name'].isin(['pass', 'free_kick']) & (df['accurate'] == 1)) & (((df['type_name'].shift(1) == 'shot') | (df['type_name'].shift(2) == 'shot')).astype(int).diff() < 0)).shift(-1)
    
    df['home_score'] = (
        ((df.type_name == 'shot') & (df.goal == 1) & (df.team_id == df.home_team_id)) |
        ((df.type_name.isin(['others_on_the_ball', 'pass'])) & (df.own_goal == 1) & (df.team_id == df.away_team_id))
        ).cumsum()
    df['home_score'] = df['home_score'] - df['match_id'].map(df.groupby('match_id')['home_score'].min())
    df['away_score'] = (
        ((df.type_name == 'shot') & (df.goal == 1) & (df.team_id == df.away_team_id)) |
        ((df.type_name.isin(['others_on_the_ball', 'pass'])) & (df.own_goal == 1) & (df.team_id == df.home_team_id))
        ).cumsum()
    df['away_score'] = df['away_score'] - df['match_id'].map(df.groupby('match_id')['away_score'].min())

    df = df.fillna(0)

    return df

def compute_labels(df, k=5):
    df['goal'] = df['goal'].fillna(0)

    actions_before_goal = None
    actions_before_own_goal = None
    for i in range(k):
        if actions_before_goal is None:
            actions_before_goal = df.goal.shift(-(i))
            actions_before_own_goal = -df.own_goal.shift(-(i))
        else:
            actions_before_goal += df.goal.shift(-(i))
            actions_before_own_goal -= df.own_goal.shift(-(i))
    actions_before_goal = actions_before_goal.fillna(0)
    actions_before_own_goal = actions_before_own_goal.fillna(0)

    is_same_period = (df.goal * df.period).replace(to_replace=False, method='bfill') == df.period
    is_same_game = (df.goal * df.match_id).replace(to_replace=False, method='bfill') == df.match_id
    is_team_next_goal = 2 * ((df.goal * df.team_id).replace(to_replace=False, method='bfill') == df.team_id) - 1
    is_team_next_goal *= actions_before_own_goal
    time_before_goal = ((df.goal * df.absolute_sec).replace(to_replace=False, method='bfill') - df.absolute_sec) / 60

    df['vaep_label_0'] = actions_before_goal * is_same_period * is_same_game * is_team_next_goal
    df['vaep_label_0_scoring'] = df['vaep_label_0'].clip(0, 1)
    df['vaep_label_0_conceding'] = abs(df['vaep_label_0'].clip(-1, 0))

    action_importance = np.maximum(1 - time_before_goal, actions_before_goal) * is_same_period * is_same_game
    action_importance[action_importance < 0] = 0
    action_importance *= is_team_next_goal
    df['VAEP_label_regression'] = action_importance

    match_result = df.match_winner != 0
    match_result *= (df.match_winner == df.team_id).astype(int) * 2 - 1
    df['vaep_label_winner'] = match_result

    return df

def encode_targets(df):
    enc = OneHotEncoder()
    df['next_action_type'] = df.groupby('match_id')['subtype_id'].shift(-1).fillna(28).astype(int)
    df['next_action_plus_seconds'] = (df.groupby('match_id')['absolute_sec'].shift(-1) - df['absolute_sec']).clip(0, 30).round().fillna(0)
    df['next_action_x'] = df.groupby('match_id')['x'].shift(-1).fillna(50)
    df['next_action_y'] = df.groupby('match_id')['y'].shift(-1).fillna(50)
    df['next_action_accurate'] = df.groupby('match_id')['accurate'].shift(-1).fillna(1)
    df['next_action_goal'] = df.groupby('match_id')['goal'].shift(-1).fillna(0)
    df['is_home_team'] = df['team_id'] == df['home_team_id']
    df['next_action_team'] = df.groupby('match_id')['is_home_team'].shift(-1).fillna(1)
    targets = ['next_action_plus_seconds', 'next_action_x', 'next_action_y', 'next_action_type', 'next_action_accurate', 'next_action_team', 'next_action_goal']

    enc.fit(df[targets])
    df_enc = pd.DataFrame(enc.transform(df[targets]).toarray(), columns=enc.get_feature_names_out(enc.feature_names_in_))
    df_enc = df_enc.rename(columns={col: col.split('.')[0] for col in df_enc.columns})
    df_enc = df_enc.drop(columns=[col for col in df_enc.columns if 'nan' in col])
    
    df_enc_type = df_enc[df_enc.columns[df_enc.columns.str.contains('next_action_type')]]
    df_enc_acc = df_enc[df_enc.columns[(df_enc.columns.str.contains('next_action_accurate') | df_enc.columns.str.contains('next_action_goal')) & ~df_enc.columns.str.contains('_0')]]
    df_enc_data = df_enc[df_enc.columns[~df_enc.columns.str.contains('next_action_type') & ~df_enc.columns.str.contains('next_action_accurate') & ~df_enc.columns.str.contains('next_action_goal') & ~df_enc.columns.str.contains('team_False')]]

    return df_enc_type, df_enc_acc, df_enc_data

def normalize_and_encode_features(df):
    df['next_action_type'] = df.groupby('match_id')['subtype_id'].shift(-1).fillna(28).astype(int)
    df['next_action_accurate'] = df.groupby('match_id')['accurate'].shift(-1).fillna(1)
    df['next_action_goal'] = df.groupby('match_id')['goal'].shift(-1).fillna(0)

    df['x'] = df.x / 100
    df['y'] = df.y / 100

    df['home_score'] = df.home_score / 10
    df['away_score'] = df.away_score / 10

    df['minute'] = df.minute / 60
    df['period'] = df.period - 1

    df.subtype_id = df.subtype_id.astype(int)
    df = pd.get_dummies(df, columns=['subtype_id'])
    enc_type_vars = [i for i in list(df.columns) if 'subtype_id_' in i]
    df = pd.get_dummies(df, columns=['next_action_type'])
    enc_next_type_vars = [i for i in list(df.columns) if 'next_action_type_' in i]

    features = enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'] + enc_next_type_vars + ['next_action_accurate', 'next_action_goal']

    return df, features

def encode_targets_v2(df):
    enc = OneHotEncoder()
    df['next_action_type'] = df.groupby('match_id')['subtype_id'].shift(-1).fillna(28).astype(int)
    df['next_action_plus_seconds'] = (df.groupby('match_id')['absolute_sec'].shift(-1) - df['absolute_sec']).clip(0, 60).round().fillna(0)
    df['next_action_x'] = df.groupby('match_id')['x'].shift(-1).fillna(50)
    df['next_action_y'] = df.groupby('match_id')['y'].shift(-1).fillna(50)
    df['next_action_accurate'] = df.groupby('match_id')['accurate'].shift(-1).fillna(1)
    df['next_action_goal'] = df.groupby('match_id')['goal'].shift(-1).fillna(0)
    df['is_home_team'] = df['team_id'] == df['home_team_id']
    df['next_action_team'] = df.groupby('match_id')['is_home_team'].shift(-1).fillna(1)
    targets = ['next_action_plus_seconds', 'next_action_x', 'next_action_y', 'next_action_type', 'next_action_accurate', 'next_action_team', 'next_action_goal']

    enc.fit(df[targets])
    df_enc = pd.DataFrame(enc.transform(df[targets]).toarray(), columns=enc.get_feature_names_out(enc.feature_names_in_))
    df_enc = df_enc.rename(columns={col: col.split('.')[0] for col in df_enc.columns})
    df_enc = df_enc.drop(columns=[col for col in df_enc.columns if 'nan' in col])
    
    df_enc_loc = df_enc[df_enc.columns[df_enc.columns.str.contains('next_action_x') | df_enc.columns.str.contains('next_action_y')]]
    df_enc_type = df_enc[df_enc.columns[df_enc.columns.str.contains('next_action_type')]]
    df_enc_acc = df_enc[df_enc.columns[(df_enc.columns.str.contains('next_action_accurate') | df_enc.columns.str.contains('next_action_goal')) & ~df_enc.columns.str.contains('_0')]]
    df_enc_data = df_enc[df_enc.columns[~df_enc.columns.str.contains('next_action_x') & ~df_enc.columns.str.contains('next_action_y') & ~df_enc.columns.str.contains('next_action_type') & ~df_enc.columns.str.contains('next_action_accurate') & ~df_enc.columns.str.contains('next_action_goal') & ~df_enc.columns.str.contains('team_False')]]

    df_y = {
        'LOC': df_enc_loc,
        'TYPE': df_enc_type,
        'ACC': df_enc_acc,
        'DATA': df_enc_data
    }

    return df_y

def normalize_and_encode_features_v2(df):
    df['next_action_type'] = df.groupby('match_id')['subtype_id'].shift(-1).fillna(28).astype(int)
    df['next_action_accurate'] = df.groupby('match_id')['accurate'].shift(-1).fillna(1)
    df['next_action_goal'] = df.groupby('match_id')['goal'].shift(-1).fillna(0)
    df['next_action_x'] = df.groupby('match_id')['x'].shift(-1).fillna(50)
    df['next_action_y'] = df.groupby('match_id')['y'].shift(-1).fillna(50)

    df['x'] = df.x / 100
    df['y'] = df.y / 100
    df['next_action_x'] = df.next_action_x / 100
    df['next_action_y'] = df.next_action_y / 100

    df['home_score'] = df.home_score / 10
    df['away_score'] = df.away_score / 10

    df['minute'] = df.minute / 60
    df['period'] = df.period - 1

    df.subtype_id = df.subtype_id.astype(int)
    df = pd.get_dummies(df, columns=['subtype_id'])
    enc_type_vars = [i for i in list(df.columns) if 'subtype_id_' in i]
    df = pd.get_dummies(df, columns=['next_action_type'])
    enc_next_type_vars = [i for i in list(df.columns) if 'next_action_type_' in i]

    features = enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'] + ['next_action_x', 'next_action_y'] + enc_next_type_vars + ['next_action_accurate', 'next_action_goal']

    features_model = {
        'LOC': enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'],
        'TYPE': enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'] + ['next_action_x', 'next_action_y'],
        'ACC': enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'] + ['next_action_x', 'next_action_y'] + enc_next_type_vars,
        'DATA': enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'] + ['next_action_x', 'next_action_y'] + enc_next_type_vars + ['next_action_accurate', 'next_action_goal']
    }

    return df, features, features_model


def encode_targets_v3(df):
    enc = OneHotEncoder()
    df['next_action_type'] = df.groupby('match_id')['subtype_id'].shift(-1).fillna(28).astype(int)
    df['next_action_plus_seconds'] = (df.groupby('match_id')['absolute_sec'].shift(-1) - df['absolute_sec']).clip(0, 60).round().fillna(0)
    df['next_action_x'] = df.groupby('match_id')['x'].shift(-1).fillna(50)
    df['next_action_y'] = df.groupby('match_id')['y'].shift(-1).fillna(50)
    df['next_action_accurate'] = df.groupby('match_id')['accurate'].shift(-1).fillna(1)
    df['next_action_goal'] = df.groupby('match_id')['goal'].shift(-1).fillna(0)
    df['is_home_team'] = df['team_id'] == df['home_team_id']
    df['next_action_team'] = df.groupby('match_id')['is_home_team'].shift(-1).fillna(True)
    targets = ['next_action_plus_seconds', 'next_action_x', 'next_action_y', 'next_action_type', 'next_action_accurate', 'next_action_team', 'next_action_goal']

    enc.fit(df[targets])
    df_enc = pd.DataFrame(enc.transform(df[targets]).toarray(), columns=enc.get_feature_names_out(enc.feature_names_in_))
    df_enc = df_enc.rename(columns={col: col.split('.')[0] for col in df_enc.columns})
    df_enc = df_enc.drop(columns=[col for col in df_enc.columns if 'nan' in col])
    
    df_enc_type = df_enc[df_enc.columns[df_enc.columns.str.contains('next_action_type')]]
    df_enc_acc = df_enc[df_enc.columns[(df_enc.columns.str.contains('next_action_accurate') | df_enc.columns.str.contains('next_action_goal')) & ~df_enc.columns.str.contains('_0')]]
    df_enc_data = df_enc[df_enc.columns[~df_enc.columns.str.contains('next_action_type') & ~df_enc.columns.str.contains('next_action_accurate') & ~df_enc.columns.str.contains('next_action_goal') & ~df_enc.columns.str.contains('team_False')]]

    df_y = {
        'TYPE': df_enc_type.reset_index(drop=True),
        'ACC': df_enc_acc.reset_index(drop=True),
        'DATA': df_enc_data.reset_index(drop=True)
    }

    return df_y

def normalize_and_encode_features_v3(df):
    df['next_action_type'] = df.groupby('match_id')['subtype_id'].shift(-1).fillna(28).astype(int)
    df['next_action_accurate'] = df.groupby('match_id')['accurate'].shift(-1).fillna(1)
    df['next_action_goal'] = df.groupby('match_id')['goal'].shift(-1).fillna(0)
    df['next_action_x'] = df.groupby('match_id')['x'].shift(-1).fillna(50)
    df['next_action_y'] = df.groupby('match_id')['y'].shift(-1).fillna(50)

    df['x'] = df.x / 100
    df['y'] = df.y / 100
    df['next_action_x'] = df.next_action_x / 100
    df['next_action_y'] = df.next_action_y / 100

    df['home_score'] = df.home_score / 10
    df['away_score'] = df.away_score / 10

    df['minute'] = df.minute / 60
    df['period'] = df.period - 1

    df.subtype_id = df.subtype_id.astype(int)
    df = pd.get_dummies(df, columns=['subtype_id'])
    enc_type_vars = [i for i in list(df.columns) if 'subtype_id_' in i]
    df = pd.get_dummies(df, columns=['next_action_type'])
    enc_next_type_vars = [i for i in list(df.columns) if 'next_action_type_' in i]

    features = enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'] + ['next_action_x', 'next_action_y'] + enc_next_type_vars + ['next_action_accurate', 'next_action_goal']

    features_model = {
        'TYPE': enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'],
        'ACC': enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'] + enc_next_type_vars,
        'DATA': enc_type_vars + ['period', 'minute', 'x', 'y', 'is_home_team', 'accurate', 'goal', 'home_score', 'away_score'] + enc_next_type_vars + ['next_action_accurate', 'next_action_goal']
    }

    return df.reset_index(drop=True), features, features_model

def load_model_training_data_template(train_sets, optimization_sets, test_sets):
    df_train = []
    df_train_y = None
    if train_sets != []:
        for fname in train_sets:
            df_train.append(load_data(fname))
            df_train[-1] = compute_features(df_train[-1])
        df_train = pd.concat(df_train)
        df_train_y = encode_targets_v3(df_train)
        df_train, complete_feature_set, features_model = normalize_and_encode_features_v3(df_train)

    df_optimization = []
    df_optimization_y = None
    if optimization_sets != []:
        for fname in optimization_sets:
            df_optimization.append(load_data(fname))
            df_optimization[-1] = compute_features(df_optimization[-1])
        df_optimization = pd.concat(df_optimization)
        df_optimization_y = encode_targets_v3(df_optimization)
        df_optimization, complete_feature_set, features_model = normalize_and_encode_features_v3(df_optimization)

    df_test = []
    df_test_y = None
    if test_sets != []:
        for fname in test_sets:
            df_test.append(load_data(fname))
            df_test[-1] = compute_features(df_test[-1])
        df_test = pd.concat(df_test)
        df_test_y = encode_targets_v3(df_test)
        df_test, complete_feature_set, features_model = normalize_and_encode_features_v3(df_test)

    return df_train, df_train_y, df_optimization, df_optimization_y, df_test, df_test_y, complete_feature_set, features_model