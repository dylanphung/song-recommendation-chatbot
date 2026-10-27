"""Song recommendations from the Spotify Tracks Dataset."""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_COLUMNS = ["track_name", "artists", "track_genre", "popularity"]

# Audio features used to judge how similar two songs sound
AUDIO_FEATURES = ["danceability", "energy", "valence", "acousticness",
                  "instrumentalness", "speechiness", "loudness", "tempo"]


class SongRecommender:
    def __init__(self, dataset_filename):
        self.data = pd.read_csv(dataset_filename)

        # The dataset lists a song once for each genre it belongs to. Genre searches use every
        # row; everything else uses one row per song so results don't repeat the same track.
        self.tracks = (self.data.sort_values("popularity", ascending=False)
                       .drop_duplicates(subset=["track_name", "artists"])
                       .reset_index(drop=True))

        # Standardise each audio feature (mean 0, standard deviation 1) so that tempo, measured
        # in BPM, doesn't outweigh features measured from 0 to 1. Genre is added as one-hot columns.
        features = self.tracks[AUDIO_FEATURES]
        standardised = (features - features.mean()) / features.std()
        genres = pd.get_dummies(self.tracks["track_genre"], dtype=float)
        self.feature_matrix = np.hstack([standardised.values, genres.values])

    def _top(self, songs, top_n):
        songs = songs.sort_values("popularity", ascending=False).drop_duplicates(subset=["track_name", "artists"])
        return songs.head(top_n)[OUTPUT_COLUMNS].to_dict(orient="records")

    def recommend_by_genre(self, genre, top_n=5):
        # Exact genre first ("pop"), otherwise any genre containing the text ("j-pop", "synth-pop")
        genres = self.data["track_genre"].astype(str).str.lower()
        matches = self.data[genres == genre.lower()]
        if matches.empty:
            matches = self.data[genres.str.contains(genre.lower(), regex=False)]
        return self._top(matches, top_n)

    def recommend_by_artist(self, artist, top_n=5):
        # Artists are stored as "Artist One;Artist Two". Prefer songs where one of the artists
        # is exactly the name given, so "Drake" doesn't also return "Drakeo the Ruler".
        artist_lists = self.tracks["artists"].fillna("").str.lower().str.split(";")
        matches = self.tracks[artist_lists.apply(lambda names: artist.lower() in [n.strip() for n in names])]
        if matches.empty:
            matches = self.tracks[self.tracks["artists"].str.contains(artist, case=False, na=False, regex=False)]
        return self._top(matches, top_n)

    def recommend_popular(self, top_n=5):
        return self._top(self.tracks, top_n)

    def find_song(self, song_title):
        """Row index of the song with this title: an exact match if there is one, otherwise
        the most popular song whose title contains the text. None if nothing matches."""
        titles = self.tracks["track_name"].astype(str)
        exact = self.tracks.index[titles.str.lower() == song_title.lower()]
        if len(exact):
            return exact[0]
        partial = self.tracks.index[titles.str.contains(song_title, case=False, regex=False)]
        return partial[0] if len(partial) else None

    def recommend_similar_songs(self, song_title, top_n=5, candidates=50):
        """Popular songs whose audio features and genre are closest to the given song.

        Takes the `candidates` most similar songs, then returns the `top_n` most popular of
        them, so results aren't dominated by obscure re-releases with near-zero popularity.
        Returns a message string if the song can't be found.
        """
        song_idx = self.find_song(song_title)
        if song_idx is None:
            return f"Sorry, I couldn't find a song titled '{song_title}' in the dataset."

        similarities = cosine_similarity(self.feature_matrix[song_idx].reshape(1, -1), self.feature_matrix).flatten()
        similarities[song_idx] = -np.inf  # don't recommend the song itself
        nearest = np.argsort(similarities)[::-1][:candidates]
        return self._top(self.tracks.iloc[nearest], top_n)
