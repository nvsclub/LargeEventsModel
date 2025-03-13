"""
Benchmark Tabular LEMs (Large Events Models)

This script performs comprehensive benchmarking of trained LEM models, including:
1. Model performance metrics (accuracy, F1-score)
2. Distribution analysis of predictions vs real data
3. Simulation analysis for game outcomes
4. Visualization of results

The script generates various plots and metrics to evaluate model quality
and compare different architectures.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from lib.lem import MLP, LEMTokenizer, simulate_game, DEVICE

class ModelBenchmarker:
    """Class to handle model benchmarking operations."""
    
    def __init__(self, seq_len: int = 3, output_size: int = 101):
        """
        Initialize the benchmarker.
        
        Args:
            seq_len: Length of input sequence
            output_size: Number of output classes
        """
        self.seq_len = seq_len
        self.output_size = output_size
        self.tokenizer = LEMTokenizer()
        
        # Set plotting style
        plt.rcParams['font.family'] = ["Times New Roman"]
        plt.rcParams['figure.figsize'] = [12, 3.5]
    
    def load_models_for_testing(self, base_models: List[List[int]], model_dir: str) -> List[Dict]:
        """
        Load models for testing.
        
        Args:
            base_models: List of hidden layer configurations
            model_dir: Directory containing model files
        
        Returns:
            List of dictionaries containing model information
        """
        models_for_testing = []
        
        for model_architecture in [
            MLP(self.seq_len, hidden_sizes, self.output_size, dropout_rate=0.3)
            for hidden_sizes in base_models
        ]:
            n_params = sum(p.numel() for p in model_architecture.parameters())
            for epoch in range(4):
                models_for_testing.append({
                    'model': copy.deepcopy(model_architecture),
                    'n_params': n_params,
                    'dir': os.path.join(model_dir, f'7112_MLP_{n_params}_{self.seq_len}_e{epoch}.pt')
                })
        
        return models_for_testing

    def load_validation_data(self, data_path: str) -> pd.DataFrame:
        """
        Load validation data for benchmarking.
        
        Args:
            data_path: Path to validation data file
        
        Returns:
            DataFrame containing validation data
        """
        return pd.read_feather(data_path)

    def get_target_distributions(self, raw_data_path: str) -> Dict[str, pd.Series]:
        """
        Calculate target distributions from raw data.
        
        Args:
            raw_data_path: Path to raw data file
        
        Returns:
            Dictionary containing various target distributions
        """
        df = pd.read_feather(raw_data_path)
        
        # Calculate goal-related distributions
        goals_delta = df.groupby('match_id')['hg'].max() - df.groupby('match_id')['ag'].max()
        goals_delta = (goals_delta.clip(-5, 5).value_counts().sort_index() / goals_delta.value_counts().sum())
        
        home_goals = (df.groupby('match_id')['hg'].max().clip(-5, 5).value_counts().sort_index() 
                     / df.groupby('match_id')['hg'].max().value_counts().sum())
        
        away_goals = (df.groupby('match_id')['ag'].max().clip(-5, 5).value_counts().sort_index() 
                     / df.groupby('match_id')['ag'].max().value_counts().sum())
        
        # Calculate event-related distributions
        event_type_dist = df['e'].value_counts() / len(df)
        event_type_dist = event_type_dist.sort_index()
        
        x_dist = df['x'].value_counts() / len(df)
        x_dist = x_dist.sort_index()
        
        y_dist = df['y'].value_counts() / len(df)
        y_dist = y_dist.sort_index()
        
        t_dist = df['t'].value_counts() / len(df)
        t_dist = t_dist.sort_index()
        
        return {
            'goals_delta': goals_delta,
            'home_goals': home_goals,
            'away_goals': away_goals,
            'event_type': event_type_dist,
            'x': x_dist,
            'y': y_dist,
            't': t_dist
        }

    def run_simulations(
        self, 
        models: List[Dict], 
        target_distributions: Dict[str, pd.Series],
        n_sims: int = 10000
    ) -> pd.DataFrame:
        """
        Run game simulations for each model and compare with target distributions.
        
        Args:
            models: List of model dictionaries
            target_distributions: Dictionary of target distributions
            n_sims: Number of simulations to run
        
        Returns:
            DataFrame containing simulation results
        """
        results = []
        base_tensor = torch.Tensor([
            [self.tokenizer.UNK_TOKEN_ID] * 6 + [0] * 9 + [1, 0, 50, 50, 0, 1] + 
            [self.tokenizer.UNK_TOKEN_ID] * 6 * (self.seq_len - 1)
        ])
        context_tensor = base_tensor.repeat(n_sims, 1).to(DEVICE)
        
        for model_data in tqdm(models, desc="Running simulations"):
            if 'e3' not in model_data['dir']:  # Only use final epoch models
                continue
                
            model = model_data['model']
            model.load_state_dict(torch.load(model_data['dir'], weights_only=True))
            model.eval()
            model.to(DEVICE)
            
            # Run simulation
            res_goals_delta, res_goals_home, res_goals_away, n_sims, inspect_e, inspect_x, inspect_y, inspect_t, \
                inspect_uncertainty, inspect_shots, inspect_xg = simulate_game(
                    model, context_tensor, max_sims=2500, return_type='results+inspect'
                )
            
            # Calculate distribution differences
            diffs = self._calculate_distribution_differences(
                res_goals_delta, res_goals_home, res_goals_away,
                inspect_e, inspect_x, inspect_y, inspect_t,
                target_distributions
            )
            
            # Calculate additional metrics
            uncertainties = [np.mean(np.concatenate(u)) for u in inspect_uncertainty]
            shots = [np.mean(s) for s in inspect_shots]
            xg = [np.mean(np.concatenate(x)) for x in inspect_xg]
            
            results.append([
                model_data['dir'],
                *diffs,
                *uncertainties,
                *shots,
                *xg
            ])
        
        columns = [
            'model_size', 
            'gdd', 'hgd', 'agd',
            'eventdistdiff', 'x_dist', 'y_dist', 't_dist',
            'uncertainty_h', 'uncertainty_e', 'uncertainty_x', 'uncertainty_y', 'uncertainty_t', 'uncertainty_a',
            'shots_h', 'shots_a', 'xg_h', 'xg_a'
        ]
        
        return pd.DataFrame(results, columns=columns)

    def _calculate_distribution_differences(
        self,
        goals_delta: np.ndarray,
        goals_home: np.ndarray,
        goals_away: np.ndarray,
        events: List[np.ndarray],
        x_coords: List[np.ndarray],
        y_coords: List[np.ndarray],
        times: List[np.ndarray],
        target_distributions: Dict[str, pd.Series]
    ) -> List[float]:
        """Calculate differences between simulated and target distributions."""
        
        # Process goals distributions
        goal_delta_dist = pd.Series(goals_delta).clip(
            lower=target_distributions['goals_delta'].index.min(),
            upper=target_distributions['goals_delta'].index.max()
        ).value_counts(normalize=True).sort_index()
        
        home_goals_dist = pd.Series(goals_home).clip(
            lower=target_distributions['home_goals'].index.min(),
            upper=target_distributions['home_goals'].index.max()
        ).value_counts(normalize=True).sort_index()
        
        away_goals_dist = pd.Series(goals_away).clip(
            lower=target_distributions['away_goals'].index.min(),
            upper=target_distributions['away_goals'].index.max()
        ).value_counts(normalize=True).sort_index()
        
        # Process event distributions
        event_dist = self.tokenizer.decode_event_types(pd.Series(np.concatenate(events)))
        event_dist = event_dist.value_counts(normalize=True).sort_index()
        
        x_dist = pd.Series(np.concatenate(x_coords)).value_counts(normalize=True).sort_index()
        y_dist = pd.Series(np.concatenate(y_coords)).value_counts(normalize=True).sort_index()
        t_dist = pd.Series(np.concatenate(times)).value_counts(normalize=True).sort_index()
        
        # Calculate absolute differences
        return [
            (target_distributions['goals_delta'] - goal_delta_dist).abs().sum(),
            (target_distributions['home_goals'] - home_goals_dist).abs().sum(),
            (target_distributions['away_goals'] - away_goals_dist).abs().sum(),
            (target_distributions['event_type'] - event_dist).abs().sum(),
            (target_distributions['x'] - x_dist).abs().sum(),
            (target_distributions['y'] - y_dist).abs().sum(),
            (target_distributions['t'] - t_dist).abs().sum()
        ]

    def plot_goal_difference_distribution(
        self,
        results: pd.DataFrame,
        target_distribution: pd.Series,
        output_path: str
    ) -> None:
        """
        Plot goal difference distributions comparison.
        
        Args:
            results: DataFrame containing simulation results
            target_distribution: Target goal difference distribution
            output_path: Path to save the plot
        """
        x_values = range(-5, 6)
        
        plt.figure(figsize=(12, 3.5))
        plt.bar(x_values, target_distribution, label='Real Distribution', zorder=-1, color='#003049')
        
        # Plot model distributions by size
        model_sizes = {
            '10M': '#ff99ac',
            '3M': '#fcbf49',
            '1M': '#eae2b7',
            '300k': '#f77f00',
            '100k': '#d62828'
        }
        
        for size_name, color in model_sizes.items():
            size_results = results[results['model_size'].str.contains(size_name)]
            for i, row in enumerate(size_results.itertuples()):
                plt.bar(
                    np.array(x_values) + 0.4 - (i + 0.5) * 0.8 / len(results),
                    row.goal_dist,
                    alpha=0.7,
                    color=color,
                    width=0.8 / len(results),
                    label=f'{size_name} Model'
                )
        
        plt.ylim(0, 0.5)
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=16)
        plt.ylabel('Probability', fontsize=20)
        plt.xlabel('Goal Difference', fontsize=20)
        plt.xticks(x_values, fontsize=16)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=16)
        
        plt.box(False)
        plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

    def plot_expected_goals_distribution(
        self,
        models: List[Dict],
        validation_data: pd.DataFrame,
        competitions_data: pd.DataFrame,
        output_path: str
    ) -> None:
        """
        Plot expected goals distribution comparison.
        
        Args:
            models: List of model dictionaries
            validation_data: Validation dataset
            competitions_data: Real competition data for comparison
            output_path: Path to save the plot
        """
        plt.figure(figsize=(12, 12))
        
        # Filter shots data
        df_shots = validation_data[
            validation_data['e'].isin([21, 22, 23]) & 
            (validation_data['t'] != -1) & 
            (validation_data['a'] == -1)
        ]
        
        # Get benchmark distribution
        sample_for_hist = competitions_data[
            ~competitions_data.shot_body_part.isna()
        ].sample(df_shots.shape[0]).shot_xg
        
        # Plot for each model size
        for spid in range(1, 6):
            plt.subplot(5, 1, spid)
            plt.hist(
                sample_for_hist,
                bins=[i/100 for i in range(101)],
                histtype='step',
                label='Benchmark xG Distribution',
                linewidth=1,
                ls='--',
                color='black',
                zorder=-1
            )
            
            plt.xlim(0, 1.5)
            plt.yscale('log')
            plt.box(False)
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
            plt.yticks([1e0, 1e1, 1e2, 1e3], fontsize=16)
            
            if spid == 5:
                plt.ylabel('Number of Shots', fontsize=20)
                plt.xlabel('Expected Goals', fontsize=20)
        
        # Calculate benchmark histogram
        sample_hist, _ = np.histogram(
            sample_for_hist,
            bins=[i/100 for i in range(101)],
            density=True
        )
        
        # Plot model predictions
        for model_data in models:
            model = model_data['model']
            model.load_state_dict(torch.load(model_data['dir'], weights_only=True))
            model = model.to(DEVICE)
            model.eval()
            
            pred_proba = model.predict_proba(df_shots.drop(columns=['target']))[:, 1]
            pred_hist, _ = np.histogram(
                pred_proba,
                bins=[i/100 for i in range(101)],
                density=True
            )
            
            distance = round(abs(sample_hist - pred_hist).sum())
            f1 = round(
                f1_score(
                    df_shots['target'],
                    model.predict(df_shots.drop(columns=['target'])),
                    average="weighted"
                ),
                3
            )
            
            # Determine subplot and style based on model size
            if '104961' in model_data['dir']:
                plt.subplot(5, 1, 1)
                plt.ylabel('MLP 100k', fontsize=24)
            elif '310781' in model_data['dir']:
                plt.subplot(5, 1, 2)
                plt.ylabel('MLP 300k', fontsize=24)
            elif '1027875' in model_data['dir']:
                plt.subplot(5, 1, 3)
                plt.ylabel('MLP 1M', fontsize=24)
            elif '3051701' in model_data['dir']:
                plt.subplot(5, 1, 4)
                plt.ylabel('MLP 3M', fontsize=24)
            elif '10174361' in model_data['dir']:
                plt.subplot(5, 1, 5)
                plt.ylabel('MLP 10M', fontsize=24)
            
            # Plot with different styles based on epoch
            if '_e0' in model_data['dir']:
                plt.hist(
                    pred_proba,
                    bins=[i/100 for i in range(101)],
                    alpha=0.3,
                    histtype='step',
                    label=f'Epoch 1 - D: {distance} - F1: {f1}',
                    linewidth=3,
                    color='#fcbf49'
                )
            elif '_e1' in model_data['dir']:
                plt.hist(
                    pred_proba,
                    bins=[i/100 for i in range(101)],
                    alpha=0.4,
                    histtype='step',
                    label=f'Epoch 2 - D: {distance} - F1: {f1}',
                    linewidth=3,
                    color='#f77f00'
                )
            elif '_e2' in model_data['dir']:
                plt.hist(
                    pred_proba,
                    bins=[i/100 for i in range(101)],
                    alpha=0.5,
                    histtype='step',
                    label=f'Epoch 3 - D: {distance} - F1: {f1}',
                    linewidth=3,
                    color='#d62828'
                )
            elif '_e3' in model_data['dir']:
                plt.hist(
                    pred_proba,
                    bins=[i/100 for i in range(101)],
                    alpha=0.7,
                    histtype='step',
                    label=f'Epoch 4 - D: {distance} - F1: {f1}',
                    linewidth=3,
                    color='#003049'
                )
        
        # Add legends
        for spid in range(1, 6):
            plt.subplot(5, 1, spid)
            plt.legend(loc='upper right', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run the benchmarking pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Tabular LEMs")
    parser.add_argument('--data_dir', type=str, required=True,
                      help="Directory containing the data files")
    parser.add_argument('--model_dir', type=str, required=True,
                      help="Directory containing the model files")
    parser.add_argument('--output_dir', type=str, required=True,
                      help="Directory to save benchmark results")
    parser.add_argument('--seq_len', type=int, default=3,
                      help="Sequence length used in the models")
    parser.add_argument('--n_sims', type=int, default=10000,
                      help="Number of simulations to run")
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = ModelBenchmarker(seq_len=args.seq_len)
    
    # Define model architectures to test
    base_models = [
        [196, 196, 196],
        [360, 360, 360],
        [682, 682, 682],
        [1200, 1200, 1200],
        [2220, 2220, 2220]
    ]
    
    # Load models
    models = benchmarker.load_models_for_testing(base_models, args.model_dir)
    
    # Load data
    val_data = benchmarker.load_validation_data(
        os.path.join(args.data_dir, f'tabular_lem/val_extensive_2223_sq{args.seq_len}_rs42_0.feather')
    )
    target_distributions = benchmarker.get_target_distributions(
        os.path.join(args.data_dir, 'raw_lem/val_extensive_2223.feather')
    )
    
    # Run simulations
    results = benchmarker.run_simulations(models, target_distributions, args.n_sims)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results.to_csv(os.path.join(args.output_dir, '7120_sim_inspect.csv'), index=False)
    
    # Generate plots
    benchmarker.plot_goal_difference_distribution(
        results,
        target_distributions['goals_delta'],
        os.path.join(args.output_dir, '7120_sim_inspect_goals_delta.pdf')
    )
    
    # Load competition data for xG plot
    competitions = pd.read_csv(os.path.join(args.data_dir, 'competitions.csv'))
    seasons = pd.read_csv(os.path.join(args.data_dir, 'seasons.csv'))
    selected_seasons = seasons[
        seasons.competition_id.isin(
            competitions[
                competitions.area_name.isin(['Germany', 'France', 'Spain', 'Portugal', 'Belgium', 'Denmark']) & 
                competitions.division_level.isin([1, 2])
            ].wy_id.tolist()
        ) & 
        (seasons.name == '2022/2023')
    ]
    
    competition_events = []
    for season_id in selected_seasons.wy_id:
        competition_events.append(
            pd.read_feather(os.path.join(args.data_dir, f'seasons/events/{season_id}.feather'))
        )
    competition_events = pd.concat(competition_events)
    
    benchmarker.plot_expected_goals_distribution(
        models,
        val_data,
        competition_events,
        os.path.join(args.output_dir, '7120_sim_inspect_expected_goals.pdf')
    )

if __name__ == "__main__":
    main() 