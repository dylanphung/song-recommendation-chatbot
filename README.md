# Song Recommendation Chatbot

A command-line chatbot that recommends songs from 114,000 Spotify tracks, keeps a personal recommendation list, answers general knowledge questions and handles small talk. It's built from classic NLP techniques rather than machine learning models or LLMs: every response comes from term-document matrices and cosine similarity.

## What you can say

| Feature | Example |
|---|---|
| Recommend by genre | `recommend songs by genre jazz` |
| Recommend by artist | `recommend songs by artist Taylor Swift` |
| Popular songs | `recommend popular songs` |
| Similar songs | `recommend songs similar to Blinding Lights` |
| Manage your list | `show my list`, `remove a song`, `clear my list` |
| Ask a question | `how are glacier caves formed?` |
| Small talk | `hi`, `tell me a joke`, `what's my name` |

The bot confirms each request before acting on it, numbers its recommendations so you can add one by typing its number or title, and remembers your name for the session.

## How it works

- **Intent matching** (`intent_matching.py`): each intent (recommend, list, question, small talk) has example phrases. These are lemmatised and turned into a term-document matrix, and the user's input is matched to the intent with the highest cosine similarity, provided it's above 0.1. The same method then picks the kind of recommendation (genre, artist, popular, similar) and the list action (show, clear, remove).
- **Question answering** (`question_answering.py`): the same matching against a dataset of question-answer pairs returns the answer to the most similar stored question. Small talk reuses this class with its own small dataset.
- **Recommendations** (`song_recommender.py`): genre and artist searches return the most popular matching songs. "Similar songs" compares eight audio features (such as danceability, energy, valence and tempo, each standardised so none dominates) plus genre, using cosine similarity. It then recommends the most popular of the 50 closest matches, so results aren't dominated by obscure re-releases.
- **Recommendation list and identity** (`rec_list_manager.py`, `identity_manager.py`): a session-long list of saved songs, and the user's name, taken from phrases like "my name is ...".

## Evaluation

I evaluated usability with a version of the Chatbot Usability Questionnaire (CUQ; Holmes et al., 2019), adapted to this bot, with 6 participants. The mean score was **68/100** (range 62–84). Participants found the bot friendly and easy to navigate. Their main criticisms were that responses felt rigid, because they're fixed text, and that complex inputs were sometimes misunderstood, which is a limit of keyword-based matching.

This evaluation was run on the original coursework version, before the fixes below.

## What I fixed when revisiting the project

- **"Similar songs" was effectively based on tempo alone.** Tempo (60–200 BPM) wasn't scaled, so it swamped energy (0–1), and every song scored above 0.99 similarity to every other. Audio features are now standardised, and five more are included.
- **Error messages were never shown.** When a song couldn't be found, the bot skipped the message and went straight to asking which song to add, which then picked a single *letter* of the hidden error text.
- **Recommendations weren't numbered,** even though the bot asked users to choose one by number. Numbers that were out of range crashed the program, and 0 silently picked the last song.
- **Searches matched too broadly.** "pop" also returned j-pop and k-pop, and an artist search returned anyone whose name contained the text. Exact matches now come first.
- **The same song appeared several times,** because the dataset lists a track once per genre it belongs to.
- **Smaller fixes:**
  - The debug text "(SMALL TALK)" no longer appears in replies.
  - "Hi" and "Hello" now get greetings.
  - "Yes" is recognised regardless of capitals.
  - Users can cancel when choosing a song.
  - A new request typed in reply to a yes/no question is now handled instead of ignored.
  - The three copies of the intent-matching logic are now one function.

## Limitations

- **Fixed phrasing.** Requests work best in the phrasings shown above. The bot matches keywords, so it can't understand paraphrases that share no words with its examples.
- **No memory between sessions.** The name and recommendation list reset each time.
- **Question answering only finds questions phrased similarly** to ones in its dataset. It retrieves answers; it doesn't generate them.

## Running it

Requirements: Python 3.9 or newer.

```bash
pip install -r requirements.txt
python main.py
```

Before the first run, download the song dataset into `data/` (see [`data/README.md`](data/README.md)). The question-answering dataset is optional: without it, the bot still recommends songs and chats.

## Project structure

```
main.py                 Conversation flow
intent_matching.py      Works out what the user is asking for
question_answering.py   Answers questions and small talk
song_recommender.py     Song search and similarity
rec_list_manager.py     The user's recommendation list
identity_manager.py     The user's name
utils.py                Text preprocessing and cosine-similarity matching
data/small-talk.csv     Small-talk questions and answers
```

Built for the Human-AI Interaction module of my BSc Computer Science with Artificial Intelligence at the University of Nottingham.
