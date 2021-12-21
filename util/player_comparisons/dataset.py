import pandas
import numpy as np
import os
import json
from util.dataset import readTable
from fetch_safely import pullData

def constructDataset(dst: str, columns: list):
	data_dir = "data"
	if not os.path.exists(data_dir):
		pullData()

	total_cols = ['season', 'points0', 'points1', 'team_stats0', 'team_stats1']
	
	all_game_data = []
	for year in range(2000, 2021):
		season_dir = "{dir}/{year}".format(dir=data_dir, year=year)
		game_results = readTable("{dir}/game_results.csv".format(dir=season_dir))
		game_results = game_results[["box_score_text", "visitor_team_name", "visitor_pts", "home_team_name", "home_pts"]]
		player_stats = readTable("{dir}/player_stats.csv".format(dir=season_dir))
		print(year)
		
		for (i, box_score_text, team0, team0_points, team1, team1_points) in game_results.itertuples():
			teams = [team0, team1]
			game_data = [year, team0_points, team1_points]
			for team in teams:
				game_team = "{game}_{team}".format(game=box_score_text, team=team)
				team_table = readTable("{dir}/{csv}.csv".format(dir=season_dir, csv=game_team))
				players = team_table["player"]

				players = pandas.concat([
					player_stats[
						(player_stats['player'].str.match(r'^' + player + r'\*?$')) & (player_stats['team_id'] == team)
					][columns]
					for player in players
				])

				game_data += [players.to_numpy()]
			all_game_data += [game_data]
	df = pandas.DataFrame(all_game_data, columns=total_cols)

	print(df)
	df.to_csv(dst)
	return df

def fixNumpyStrings(df):
	for team_stats in [df['team_stats0'], df['team_stats1']]:
		for i in range(len(team_stats)):
			players_str = team_stats.iloc[i].strip()
			players_str = players_str[1:-1]

			players_strs = players_str.replace(']', '').split('[')[1:]
			players_splits = [player_str.split() for player_str in players_strs]

			player_stats = np.empty((len(players_splits), len(players_splits[0])))
			for player in range(player_stats.shape[0]):
				for col in range(player_stats.shape[1]):
					value = players_splits[player][col]
					if value == 'nan':
						value = 0
					else:
						value = float(value)
					player_stats[player][col] = value
			team_stats.iloc[i] = player_stats
		print(team_stats)
	return df

def standardizeDataset(df, cache_dir: str, columns: list):
	seasons = df['season'].unique()
	season_min = seasons.min()
	season_max = seasons.max()
	standards_file = 'standards_{min}-{max}.json'.format(min=season_min, max=season_max)
	standards_file = os.path.join(cache_dir, standards_file)

	if not os.path.exists(standards_file):
		data_dir = "data"
		mins = []
		maxs = []
		for year in seasons:
			season_dir = "{dir}/{year}".format(dir=data_dir, year=year)
			player_stats = readTable("{dir}/player_stats.csv".format(dir=season_dir))
			player_stats = player_stats[columns]
			mins += [player_stats.min()]
			maxs += [player_stats.max()]

		true_mins = mins[0]
		true_maxs = maxs[0]
		for year in range(1, len(mins)):
			for col in range(1, len(mins[year])):
				if (maxs[year][col] > true_maxs[col]):
					true_maxs[col] = maxs[year][col]
				if (mins[year][col] < true_mins[col]):
					true_mins[col] = mins[year][col]

		data = {
			'min': true_mins.to_numpy().tolist(),
			'max': true_maxs.to_numpy().tolist()
		}
		with open(standards_file, 'w') as jsonfile:
			json.dump(data, jsonfile)

	with open(standards_file, 'r') as jsonfile:
		data = json.load(jsonfile)
	maxs = np.asarray(data['max'])
	mins = np.asarray(data['min'])

	value_spread = maxs - mins
	
	all_team_stats = [df['team_stats0'], df['team_stats1']]
	for team_stats in all_team_stats:
		for players in team_stats:
			for player in players:
				new_player = (player - mins) / value_spread
				for i in range(len(new_player)):
					player[i] = new_player[i]

	return df
