"""The user's personal list of saved song recommendations."""


class RecommendationListManager:
    def __init__(self):
        self.rec_list = []

    def add_song(self, song):
        if song in self.rec_list:
            return f"'{song['track_name']}' is already in your recommendation list."
        self.rec_list.append(song)
        return f"'{song['track_name']}' by {song['artists']} has been added to your recommendation list. What else can I do for you?"

    def remove_song(self, song):
        if song not in self.rec_list:
            return f"'{song['track_name']}' is not in your recommendation list."
        self.rec_list.remove(song)
        return f"'{song['track_name']}' by {song['artists']} has been removed from your recommendation list."

    def view_list(self):
        if not self.rec_list:
            return "Your recommendation list is currently empty."
        lines = [f"{i}. {song['track_name']} by {song['artists']} (Genre: {song.get('track_genre', 'Unknown')})"
                 for i, song in enumerate(self.rec_list, start=1)]
        return "Here is your recommendation list:\n" + "\n".join(lines)

    def clear_list(self):
        self.rec_list.clear()
        return "Your recommendation list has been cleared."
