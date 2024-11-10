def model(words, strikes, isOneAway, correctGroups, previousGuesses, error):
	"""
	_______________________________________________________
	Parameters:
	words - 1D Array with 16 shuffled words
	strikes - Integer with number of strikes
	isOneAway - Boolean if your previous guess is one word away from the correct answer
	correctGroups - 2D Array with groups previously guessed correctly
	previousGuesses - 2D Array with previous guesses
	error - String with error message (0 if no error)

	Returns:
	guess - 1D Array with 4 words
	endTurn - Boolean if you want to end the puzzle
	_______________________________________________________
	"""

	# Your Code here
	# Good Luck!

	print(error)

	#region Imports
	import fasttext
	import fasttext.util
	import pandas as pd
	import numpy as np
	from k_means_constrained import KMeansConstrained
	from sklearn.metrics.pairwise import cosine_similarity
	#endregion

	#region Word-embedding function attribute
	if not hasattr(model, "words_and_embeds"):
		print("Loading model...")
		ft = fasttext.load_model('/Users/davidhunt/Documents/fastText/cc.en.300.bin')
		ft.get_dimension()
		fasttext.util.reduce_model(ft, 100)
		ft.get_dimension()
		print("Model loaded.")

		embeddings = [ft.get_word_vector(word) for word in words]
		model.words_and_embeds = list(zip(words, embeddings)) # store words and their embeddings in a list of tuples
		model.focus_cluster = 0
	#endregion

	#region Finding words, embeds, and similarities
	flat_correct = [item for sublist in correctGroups for item in sublist]
	words = [word for word in words if word not in flat_correct] # Remove words that have already been guessed correctly

	embeds = [embed for word, embed in model.words_and_embeds if word in words] # Extract embeddings for remaining words
	groups = len(words) // 4 # Number of groups to form

	similarities = cosine_similarity(embeds) # Calculate cosine similarities between words
	#endregion

	#region Finding clusters and sorting by priority

	clf = KMeansConstrained(n_clusters=groups, size_min=4, size_max=4, random_state=42)
	clf.fit_predict(similarities) # Cluster words into groups using KMeansConstrained

	cluster_similarity_list = []

	for i in range(groups):
		# Get indices of elements in the current cluster
		cluster_indices = [j for j in range(len(words)) if clf.labels_[j] == i]
		# Extract similarities for the current cluster only
		cluster_similarities = similarities[np.ix_(cluster_indices, cluster_indices)]
		# Calculate the average similarity (excluding self-similarity)
		avg_similarity = np.mean(cluster_similarities[np.triu_indices(len(cluster_indices), k=1)])
		# cluster_cohesion[i] = avg_similarity
		# Print the words in each cluster for reference
		cluster_words = [words[j] for j in cluster_indices]
		print(f"Cluster {i+1}: {cluster_words}")
		print(f"Average similarity (cohesion): {avg_similarity:.4f}\n")

		cluster_similarity_list.append((cluster_words, avg_similarity))

	cluster_similarity_list.sort(key=lambda x: x[1], reverse=True)
	sorted_clusters = [x[0] for x in cluster_similarity_list]

	#endregion

	#region Similarity pairs and function defs
	similarity_pairs = {}

	for i in range(len(words)):
		for j in range(i + 1, len(words)):
			similarity_pairs[(words[i], words[j])] = similarities[i, j]

	def get_scores(list1, list2):
		scores = []
		for word in list1:
			score = 0
			for word2 in list2:
				if word != word2:
					score += similarity_pairs.get((word, word2), similarity_pairs.get((word2, word)))
			scores.append(score)
		return scores

	def find_most_dissimilar(list):
		scores = get_scores(list, list)
		print(scores.index(min(scores)))
		return scores.index(min(scores))

	def find_next_similar(list1, list2, priority = 2):
		scores = get_scores(list1, list2)
		print(words[scores.index(max(scores))])
		print(scores.index(max(scores)))
		return words[scores.index(max(scores))]
	#endregion

	#region DETERMINING THE GUESS

	one_off_streak = 0
	if (isOneAway):
		one_off_streak = 1
		for i in range(len(previousGuesses) - 2, -1, -1):
			# Convert each guess to a set and find the symmetric difference
			diff_count = len(set(previousGuesses[i]).symmetric_difference(set(previousGuesses[i + 1])))
			
			# If exactly one word is different, increment the consecutive one-off counter
			if diff_count == 2:  # Two words differ in the symmetric difference (one in each set)
				one_off_streak += 1
			else:
				# Break the loop if a guess is not one-off
				break

	def increment_counter(x):
		return (x + 1) % len(sorted_clusters)

	if not previousGuesses or (previousGuesses[-1] == correctGroups[-1]): # if no previous guesses or the last guess was correct
		model.focus_cluster = 0
		guess = sorted_clusters[model.focus_cluster] # guess the most cohesive cluster remaining

	if (isOneAway and one_off_streak == 1): # if the previous guess was one-off but no previous ones were
		guess = previousGuesses[-1]
		guess[find_most_dissimilar(guess)] = (find_next_similar(words, guess))
	
	if (isOneAway and one_off_streak > 1 or not isOneAway): # if there have been consecutive one-off guesses or if the last guess was more than one off
		model.focus_cluster = increment_counter(model.focus_cluster) # move to the next cluster
		guess = sorted_clusters[model.focus_cluster]

	while (guess in previousGuesses):
		model.focus_cluster = increment_counter(model.focus_cluster)
		guess = sorted_clusters[model.focus_cluster]

	return guess, False

	#endregion

	# Example code where guess is hard-coded
	# guess = ["apples", "bananas", "oranges", "grapes"] # 1D Array with 4 elements containing guess
	# endTurn = False # True if you want to end puzzle and skip to the next one