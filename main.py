"""A command-line chatbot that recommends songs, keeps a recommendation list and answers questions."""
import os
import re

from identity_manager import IdentityManager
from intent_matching import IntentMatching
from question_answering import QuestionAnsweringSystem
from rec_list_manager import RecommendationListManager
from song_recommender import SongRecommender

SONGS_PATH = "data/spotify-songs.csv"
QA_PATH = "data/COMP3074-CW1-Dataset.csv"
SMALL_TALK_PATH = "data/small-talk.csv"

YES_WORDS = {"yes", "y", "yeah", "yep", "sure", "ok", "okay"}

# Intents that count as a new request if typed when the bot asked a yes/no question
NEW_REQUEST_INTENTS = {"song_recommendation", "rec_list", "question"}


def is_yes(text):
    return any(word in YES_WORDS for word in re.findall(r"[a-z]+", text.lower()))


def format_songs(songs):
    return "\n".join(f"{i}. {s['track_name']} by {s['artists']} ({s['track_genre']}, popularity {s['popularity']})"
                     for i, s in enumerate(songs, start=1))


class ExitChat(Exception):
    """Raised when the user types 'exit' at any prompt."""


class Chatbot:
    def __init__(self):
        if not os.path.exists(SONGS_PATH):
            raise SystemExit(f"Missing {SONGS_PATH}. See data/README.md for where to download it.")
        self.identity = IdentityManager()
        self.intents = IntentMatching()
        self.rec_list = RecommendationListManager()
        self.songs = SongRecommender(SONGS_PATH)
        self.small_talk = QuestionAnsweringSystem(SMALL_TALK_PATH)
        # The Q&A dataset is optional; without it the bot still recommends songs and chats
        self.qa = QuestionAnsweringSystem(QA_PATH) if os.path.exists(QA_PATH) else None
        self.pending_input = None  # a new request typed in reply to a yes/no question

    # ---- input and output ----
    def say(self, text):
        print(f"Bot: {text}")

    def ask(self, prompt=None):
        if prompt:
            self.say(prompt)
        reply = input(f"{self.identity.get_name()}: ").strip()
        if reply.lower() == "exit":  # 'exit' ends the chat from any prompt
            raise ExitChat
        return reply

    def say_okay(self):
        if not self.pending_input:
            self.say("Okay, let me know how else I can help.")

    def ask_yes_no(self, prompt):
        """Ask a yes/no question. If the reply is a new request instead (e.g. another
        recommendation), save it to handle next rather than ignoring it."""
        reply = self.ask(prompt)
        if is_yes(reply):
            return True
        if self.intents.get_intent(reply) in NEW_REQUEST_INTENTS:
            self.pending_input = reply
        return False

    def pick_song(self, songs, prompt, not_found):
        """Let the user choose a song by its number or its title. Returns None if they cancel."""
        while True:
            choice = self.ask(prompt + " (or 'cancel')")
            if choice.lower() in {"cancel", "no", "none", "nothing"}:
                return None
            if choice.isdigit() and 1 <= int(choice) <= len(songs):
                return songs[int(choice) - 1]
            match = next((s for s in songs if s["track_name"].lower() == choice.lower()), None)
            if match:
                return match
            self.say(not_found)

    # ---- conversation ----
    def run(self):
        print("Welcome to the chatbot! Type 'exit' to quit.")
        try:
            self.converse()
        except ExitChat:
            self.say("Goodbye!")

    def converse(self):
        self.say("Hi, what is your name?")
        while not self.identity.name:
            self.identity.set_name(self.ask())
            if not self.identity.name:
                self.say("You can enter your name by saying 'my name is ...' or simply typing it!")
        self.say(f"Nice to meet you, {self.identity.get_name()}! How can I help you today?")

        while True:
            if self.pending_input:
                user_input, self.pending_input = self.pending_input, None
            else:
                user_input = self.ask()

            intent = self.intents.get_intent(user_input)
            if intent == "small_talk":
                self.handle_small_talk(user_input)
            elif intent == "question":
                self.handle_question(user_input)
            elif intent == "rec_list":
                self.handle_rec_list(user_input)
            elif intent == "song_recommendation":
                self.handle_recommendation(user_input)
            else:
                self.say("I'm not quite sure what you're asking. Can you rephrase your question?")

    def handle_small_talk(self, user_input):
        answer = self.small_talk.answer_question(user_input)
        self.say(answer.replace("{name}", self.identity.get_name()))

    def handle_question(self, user_input):
        if self.qa is None:
            self.say("Sorry, question answering isn't set up (the Q&A dataset isn't installed).")
        else:
            self.say(self.qa.answer_question(user_input))

    def handle_rec_list(self, user_input):
        action = self.intents.get_list_intent(user_input)
        if action == "display":
            self.say(self.rec_list.view_list())
        elif action == "clear":
            self.say(self.rec_list.clear_list())
        elif action == "delete":
            if not self.rec_list.rec_list:
                self.say("Your recommendation list is already empty!")
                return
            if not self.ask_yes_no("You want to remove a song from your recommendation list, is that correct?"):
                self.say_okay()
                return
            self.say(self.rec_list.view_list())
            song = self.pick_song(self.rec_list.rec_list,
                                  "Which song would you like to remove? Enter its number or its name",
                                  "Sorry, I couldn't find that song in your recommendation list. Please try again.")
            if song:
                self.say(self.rec_list.remove_song(song))
                if self.ask_yes_no("Would you like to see your recommendation list now?"):
                    self.say(self.rec_list.view_list())
                    return
            self.say_okay()
        else:
            self.say("I can show, clear or remove songs from your recommendation list.")

    def handle_recommendation(self, user_input):
        kind = self.intents.get_rec_intent(user_input)
        if kind == "unknown":
            self.say("I'm sorry, I didn't understand that. I can recommend songs by genre or artist, "
                     "popular songs, or songs similar to one you like.")
            return

        # Take the genre, artist or song title from the text after the keyword,
        # e.g. "recommend songs by genre jazz" -> "jazz"
        pattern = {"genre": r"genre\s+(.+)", "artist": r"artist\s+(.+)", "similar": r"similar to\s+(.+)"}.get(kind)
        detail = None
        if pattern:
            match = re.search(pattern, user_input, re.IGNORECASE)
            detail = match.group(1).strip() if match else self.ask(
                {"genre": "Which genre?", "artist": "Which artist?", "similar": "Similar to which song?"}[kind])

        confirmation = {
            "popular": "You want to be recommended the most popular songs, is that correct?",
            "similar": f"You want to be recommended songs similar to '{detail}', is that correct?",
        }.get(kind, f"You want to be recommended songs by the {kind} '{detail}', is that correct?")
        if not self.ask_yes_no(confirmation):
            self.say_okay()
            return

        if kind == "genre":
            results = self.songs.recommend_by_genre(detail)
        elif kind == "artist":
            results = self.songs.recommend_by_artist(detail)
        elif kind == "popular":
            results = self.songs.recommend_popular()
        else:
            results = self.songs.recommend_similar_songs(detail)

        if isinstance(results, str):  # an error message, e.g. song not found
            self.say(results)
            return
        if not results:
            self.say(f"Sorry, I couldn't find any songs for '{detail}'.")
            return

        self.say("Thank you, here are your recommendations:\n" + format_songs(results))
        if self.ask_yes_no("Would you like to add any of these songs to your recommendation list?"):
            song = self.pick_song(results, "Please enter the number or name of the song you'd like to add",
                                  "Sorry, I couldn't find that song in the recommendations. Could you try again?")
            if song:
                self.say(self.rec_list.add_song(song))
                return
        self.say_okay()


if __name__ == "__main__":
    Chatbot().run()
