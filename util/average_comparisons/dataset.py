import pandas
import os
from util.dataset import readTable
from fetch_safely import pullData

def constructDataset(dst: str):
	data_dir = "data"
	if not os.path.exists(data_dir):
		pullData()

	data_dir = "data"
	columns = [
		'fg_per_g', 'fga_per_g', 'fg_pct',
		'fg3_per_g', 'fg3a_per_g', 'fg3_pct',
		'fg2_per_g', 'fg2a_per_g', 'fg2_pct',
		'efg_pct',
		'ft_per_g', 'fta_per_g', 'ft_pct',
		'orb_per_g', 'drb_per_g', 'trb_per_g',
		'ast_per_g', 'stl_per_g', 'blk_per_g', 'tov_per_g', 'pf_per_g', 'pts_per_g'
	]

	team_cols = [
		[col + str(i) for col in columns] for i in range(2)
	]
	total_cols = ['season', 'points0', 'points1'] + team_cols[0] + team_cols[1]
	
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

				game_data += list(players.mean())
			all_game_data += [game_data]
	df = pandas.DataFrame(all_game_data, columns=total_cols)

	print(df)
	df.to_csv(dst)
	return df

def standardizeDataset(df):
	# NOTE: this does not standardize common fields across home vs visitor teams features
	# (i.e. fg_per_g0 and fg_per_g1)
	# NOTE: discovered interpolation from 0-1 performs better than a z-score
	return (df - df.min()) / (df.max() - df.min())
	# return (df - df.mean()) / df.std()
