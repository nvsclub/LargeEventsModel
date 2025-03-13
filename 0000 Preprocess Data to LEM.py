"""
LEM (Large Events Model) Data Preprocessing Script

This script handles the preprocessing of soccer event data into the LEM standard format.
It performs three main tasks:
1. Converts raw data to LEM standard format
2. Preprocesses data for tabular models
3. Preprocesses data for time series models

Built for Wyscout V3 data.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional, Union, Dict
from pathlib import Path

class LEMTokenizer:
    """Tokenizer for event types in soccer data."""
    
    def __init__(self):
        # Initialize vocabulary with numbers 0-100
        self.vocab = {i: i for i in range(0, 101)}
        self.vocab['<UNK>'] = -1

        # List of predefined event types
        self.event_types_list = [
            'pass', 'long_pass', 'cross', 'touch', 'aerial_duel', 'clearance', 'interception',
            'loose_ball_duel', 'defensive_duel', 'offensive_duel', 'dribble', 'carry',
            'game_interruption', 'own_goal', 'throw_in', 'free_kick', 'goal_kick', 'infraction',
            'corner', 'acceleration', 'offside', 'right_foot_shot', 'left_foot_shot', 'head_shot',
            'goalkeeper_exit', 'save', 'shot_against', 'fairplay', 'yellow_card', 'red_card',
            'first_half_end', 'game_end'
        ]

        # Build vocabularies
        for i, event_type in enumerate(self.event_types_list):
            self.vocab[event_type] = i
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.UNK_TOKEN_ID = self.vocab['<UNK>']

    def encode_event_types(self, data: pd.Series) -> pd.Series:
        """Encode event types to their corresponding IDs."""
        return data.map(self.vocab)

    def decode_event_types(self, data: pd.Series) -> pd.Series:
        """Decode IDs back to event type names."""
        return data.map(self.reverse_vocab)

def convert_to_lem_standard(
    competitions_path: str,
    seasons_path: str,
    matches_path: str,
    events_dir: str,
    output_path: str,
    areas: List[str] = None,
    division_levels: List[int] = None,
    seasons: List[str] = None
) -> None:
    """
    Convert raw soccer data to LEM standard format.
    
    Args:
        competitions_path: Path to competitions.csv
        seasons_path: Path to seasons.csv
        matches_path: Path to matches.csv
        events_dir: Directory containing event files
        output_path: Path to save the processed data
        areas: List of areas to include (e.g. ['Germany', 'France'])
        division_levels: List of division levels to include (e.g. [1, 2])
        seasons: List of seasons to include (e.g. ['2022/2023'])
    """
    if areas is None:
        areas = ['Germany', 'France', 'Spain', 'Portugal', 'Belgium', 'Denmark']
    if division_levels is None:
        division_levels = [1, 2]
    
    # Load base data
    competitions = pd.read_csv(competitions_path)
    seasons = pd.read_csv(seasons_path)
    
    # Filter seasons based on criteria
    selected_seasons = seasons[
        seasons.competition_id.isin(
            competitions[
                competitions.area_name.isin(areas) & 
                competitions.division_level.isin(division_levels)
            ].wy_id.tolist()
        )
    ]
    if seasons is not None:
        selected_seasons = selected_seasons[selected_seasons.name.isin(seasons)]

    # Load matches and events
    matches = pd.read_csv(matches_path, low_memory=False)
    events = []
    for season in tqdm(selected_seasons.wy_id.tolist(), desc="Loading events"):
        events.append(pd.read_feather(os.path.join(events_dir, f"{season}.feather")))
    
    # Merge events with match data
    events = pd.concat(events).merge(
        matches[['wy_id', 'home_team_id', 'away_team_id', 'winner']].rename(columns={'wy_id': 'match_id'}),
        on='match_id'
    )
    events['game_result'] = -1 + (events.winner == 0) + (events.winner == events.team_id) * 2

    # Process event types
    events = process_event_types(events)
    
    # Process game state variables
    events = process_game_state(events)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    events.to_feather(output_path)

def process_event_types(events: pd.DataFrame) -> pd.DataFrame:
    """Process and categorize different types of events."""
    
    # Discriminate duels
    events.loc[events.defensive_duel, 'type_primary'] = 'defensive_duel'
    events.loc[events.offensive_duel, 'type_primary'] = 'offensive_duel'
    events.loc[events.aerial_duel, 'type_primary'] = 'aerial_duel'
    events.loc[events.loose_ball_duel, 'type_primary'] = 'loose_ball_duel'
    events.loc[events.dribble, 'type_primary'] = 'dribble'

    # Discriminate crosses & long passes
    events.loc[events.cross, 'type_primary'] = 'cross'
    events.loc[events.long_pass, 'type_primary'] = 'long_pass'

    # Process carries
    events_carries = events[events.carry].copy()
    events_carries['type_primary'] = 'carry'

    # Process shots
    events.loc[events.shot_body_part == 'head_or_other', 'type_primary'] = 'head_shot'
    events.loc[events.shot_body_part == 'right_foot', 'type_primary'] = 'right_foot_shot'
    events.loc[events.shot_body_part == 'left_foot', 'type_primary'] = 'left_foot_shot'
    events.loc[events.type_primary == 'shot', 'type_primary'] = 'right_foot_shot'

    # Process other events
    events.loc[events.save, 'type_primary'] = 'save'
    events.loc[events.yellow_card, 'type_primary'] = 'yellow_card'
    events.loc[events.red_card, 'type_primary'] = 'red_card'

    # Process end events
    events_end = events.groupby(['match_id', 'match_period']).tail(1)
    events_end.loc[events_end.match_period == '1H', 'type_primary'] = 'first_half_end'
    events_end.loc[events_end.match_period == '2H', 'type_primary'] = 'game_end'

    # Combine all events
    events = pd.concat([events, events_carries, events_end])
    return events.sort_values(['match_id', 'match_period', 'minute', 'second'])

def process_game_state(events: pd.DataFrame) -> pd.DataFrame:
    """Process game state variables like goals, cards, etc."""
    
    # Calculate time between events
    events['t'] = (events.minute - events.minute.shift(1).fillna(0)) * 60 + (events.second - events.second.shift(1).fillna(0))
    
    # Calculate cumulative statistics per match
    for prefix, condition in [
        ('hg', events.goal & events.h),
        ('ag', events.goal & ~events.h),
        ('hr', events.red_card & events.h),
        ('ar', events.red_card & ~events.h),
        ('hy', events.yellow_card & events.h),
        ('ay', events.yellow_card & ~events.h)
    ]:
        events[prefix] = condition.groupby(events.match_id).cumsum()
    
    # Process period indicators
    events['p'] = events.match_period.map({'1H': False, '2H': True}).shift(1)
    events.loc[events.match_id != events.match_id.shift(1), 'p'] = 0
    
    # Process minute and second
    for col in ['m', 's']:
        events[col] = events[col[0]].shift(1)
        events.loc[events.match_id != events.match_id.shift(1), col] = 0
    
    return events

def arrange_data_for_tabular(
    input_path: str,
    seq_len: int,
    n_files: int,
    output_dir: str = None
) -> None:
    """
    Arrange data for tabular models.
    
    Args:
        input_path: Path to input feather file
        seq_len: Sequence length to use
        n_files: Number of files to split the data into
        output_dir: Directory to save the processed files
    """
    if output_dir is None:
        output_dir = input_path.replace('/raw_lem/', '/tabular_lem/').rsplit('.', 1)[0]
    
    # Load and tokenize data
    data = pd.read_feather(input_path)
    tokenizer = LEMTokenizer()
    data['e'] = tokenizer.encode_event_types(data['e'])
    
    # Add context for each sequence length
    event_vars = ['h', 'e', 'x', 'y', 't', 'a']
    for i in range(1, seq_len + 1):
        data_context = data.shift(i).fillna(tokenizer.UNK_TOKEN_ID)
        data_context.loc[data_context['match_id'] != data['match_id'], event_vars] = tokenizer.UNK_TOKEN_ID
        data_context = data_context[event_vars].add_prefix(f'c{i}_').astype(np.int8)
        data = pd.concat([data, data_context], axis=1)
    
    # Prepare data for each event variable
    data_lst = []
    for i, var in enumerate(event_vars):
        edit_data = data.copy()
        edit_data['target'] = edit_data[var].clip(0, 100)
        edit_data[event_vars[i:]] = tokenizer.UNK_TOKEN_ID
        edit_data = edit_data.drop(columns=['match_id'])
        data_lst.append(edit_data)
    
    # Combine and process final dataset
    data = pd.concat(data_lst)
    data = data.astype(np.int8)
    data = data.sample(frac=1, random_state=42)
    
    # Save data splits
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_files):
        start_idx = i * len(data) // n_files
        end_idx = (i + 1) * len(data) // n_files
        output_path = f"{output_dir}_sq{seq_len}_rs42_{i}.feather"
        data.iloc[start_idx:end_idx].to_feather(output_path)

def arrange_data_for_time_series(
    input_path: str,
    seq_len: int,
    n_files: int,
    output_dir: str = None
) -> None:
    """
    Arrange data for time series models.
    
    Args:
        input_path: Path to input feather file
        seq_len: Sequence length to use
        n_files: Number of files to split the data into
        output_dir: Directory to save the processed files
    """
    if output_dir is None:
        output_dir = input_path.replace('/raw_lem/', '/time_series_lem/').rsplit('.', 1)[0]
    
    # Load data
    df = pd.read_feather(input_path)
    tokenizer = LEMTokenizer()
    
    # Process events
    event_vars = ['h', 'e', 'x', 'y', 't', 'a']
    context_vars = ['p', 'm', 's', 'hg', 'ag', 'hr', 'ar', 'hy', 'ay']
    
    # Create temporary directory for processing
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process data in chunks
    for match_id in tqdm(df.match_id.unique(), desc="Processing matches"):
        match_data = df[df.match_id == match_id]
        data_events = match_data[event_vars]
        data_contexts = match_data[context_vars]
        
        for i in range(len(data_events)):
            event_no = i // len(event_vars)
            event_var_id = i % len(event_vars)
            
            # Prepare series data
            series = data_events.iloc[:, max(0, i-(len(event_vars) * seq_len)):i].clip(0, 100)
            if series.shape[1] < (len(event_vars) * seq_len):
                padding = pd.DataFrame(
                    [[tokenizer.UNK_TOKEN_ID] * (len(event_vars)*seq_len - series.shape[1])] * data_events.shape[0]
                )
                series = pd.concat([padding, series], axis=1)
            series.columns = [f'i{seq_len*len(event_vars) - j}' for j in range(series.shape[1])]
            
            # Prepare target and context
            target = data_events.iloc[:, i].rename('target').clip(0, 100)
            context = data_contexts.iloc[:, max(0, event_no * len(context_vars) - len(context_vars)):event_no * len(context_vars)].clip(0, 100)
            if context.shape[1] < len(context_vars):
                context = pd.DataFrame([[0]*len(context_vars)] * data_events.shape[0])
            context.columns = [f'c{j}' for j in range(len(context_vars))]
            
            # Combine data
            combined_data = pd.concat([context, series, target], axis=1)
            combined_data['event_var_id'] = event_var_id
            combined_data = combined_data.dropna()
            
            if not combined_data.empty:
                for file_idx in range(n_files):
                    temp_file = os.path.join(temp_dir, f'arrange_data_as_time_series_{file_idx}.csv')
                    combined_data.sample(frac=1/n_files).to_csv(
                        temp_file, 
                        mode='a',
                        header=not os.path.exists(temp_file),
                        index=False
                    )
    
    # Save final files
    os.makedirs(output_dir, exist_ok=True)
    for i in range(n_files):
        temp_file = os.path.join(temp_dir, f'arrange_data_as_time_series_{i}.csv')
        if os.path.exists(temp_file):
            data = pd.read_csv(temp_file)
            data = data.astype(np.int8)
            output_path = f"{output_dir}_sq{seq_len}_rs42_{i}.feather"
            data.to_feather(output_path)
            os.remove(temp_file)

def main():
    """Main function to run the preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess soccer event data for LEM models")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the raw data files")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save processed files")
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[1, 3, 5, 7, 9], help="Sequence lengths to process")
    parser.add_argument('--n_files', type=int, default=10, help="Number of files to split the data into")
    args = parser.parse_args()
    
    # Convert to LEM standard
    print("Converting data to LEM standard...")
    convert_to_lem_standard(
        competitions_path=os.path.join(args.data_dir, "competitions.csv"),
        seasons_path=os.path.join(args.data_dir, "seasons.csv"),
        matches_path=os.path.join(args.data_dir, "matches.csv"),
        events_dir=os.path.join(args.data_dir, "seasons/events"),
        output_path=os.path.join(args.output_dir, "raw_lem/data.feather")
    )
    
    # Process for different model types
    raw_lem_path = os.path.join(args.output_dir, "raw_lem/data.feather")
    for seq_len in args.seq_lengths:
        print(f"Processing for sequence length {seq_len}...")
        
        print("Arranging data for tabular models...")
        arrange_data_for_tabular(
            raw_lem_path,
            seq_len,
            args.n_files,
            os.path.join(args.output_dir, "tabular_lem")
        )
        
        # WARNING: Uncomment this when you want to process time series models as per the annexes of the paper
        # print("Arranging data for time series models...")
        # arrange_data_for_time_series(
        #     raw_lem_path,
        #     seq_len,
        #     args.n_files,
        #     os.path.join(args.output_dir, "time_series_lem")
        # )

if __name__ == "__main__":
    main() 