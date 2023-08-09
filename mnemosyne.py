import json
import numpy as np
import pickle
from gpt4all import GPT4All
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

model = GPT4All("C://AI_MODELS/orca-mini-3b.ggmlv3.q4_0.bin")

EMBEDDINGS_FILE = "embeddings.npy"
DATA_FILE = "data.npy"
VECTORIZER_FILE = 'vectorizer.pkl'
#HISTORY_FILE = "response_history.json"
#MAX_HISTORY_LEN = 3
SYSTEM_PROMPT = "You are a assistant that provides clear and direct responses."
HISTORY_FILE = "conversation_history.json"
MAX_HISTORY_EXCHANGES = 3  # Number of user-bot exchanges to consider for context

# --- HISTORY HANDLING FUNCTIONS --- #

def load_history():
    with open(HISTORY_FILE, 'r') as file:
        return json.load(file)['exchanges']

def save_to_history(user_input, response):
    history = load_history()
    exchange = {"user_input": user_input, "bot_response": response}
    history.append(exchange)
    # Keep only the last few exchanges for context
    history = history[-MAX_HISTORY_EXCHANGES:]
    with open(HISTORY_FILE, 'w') as file:
        json.dump({"exchanges": history}, file)

def is_in_recent_history(response):
    history = load_history()
    return response in history
# Load or create a new TfidfVectorizer

if os.path.exists(VECTORIZER_FILE):
    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)
else:
    vectorizer = TfidfVectorizer()

data = []
embeddings = []

if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(DATA_FILE):
    data = list(np.load(DATA_FILE, allow_pickle=True))
    texts = [item['input'] + ' ' + item['output'] for item in data]
    vectorizer.fit(texts)
    embeddings = vectorizer.transform(texts).toarray()

else:
    data = [
        {"input": "I am trying my best.", "output": "That's all we can do, I'll try my best.", "timestamp": datetime.now()}
    ]
    texts = [item['input'] + ' ' + item['output'] for item in data]
    vectorizer.fit(texts)
    embeddings = vectorizer.transform(texts).toarray()

    with open(VECTORIZER_FILE, 'wb') as file:
        pickle.dump(vectorizer, file)

# --- EMBEDDING AND DATA HANDLING FUNCTIONS --- #
def decayed_similarity(original_similarity, timestamp, decay_factor=0.05):
    time_elapsed = np.abs(np.datetime64('now') - np.datetime64(timestamp))
    days_elapsed = time_elapsed.astype('timedelta64[D]').astype(int)
    return original_similarity * np.exp(-decay_factor * days_elapsed)

def aggregate_embeddings(embeddings, user_input_embedding, top_N=5):
    kernel_values = np.dot(embeddings, user_input_embedding.T).ravel()
    top_indices = kernel_values.argsort()[-top_N:][::-1]
    return np.mean(embeddings[top_indices], axis=0)

def update_and_save_data(user_input, response):
    global embeddings, vectorizer
    data_entry = {
        "input": user_input,
        "output": response,
        "input_sentiment": compute_sentiment(user_input),
        "output_sentiment": compute_sentiment(response),
        "timestamp": datetime.now()
    }
    data.append(data_entry)
    new_embedding = vectorizer.transform([user_input + ' ' + response]).toarray()
    embeddings = np.vstack([embeddings, new_embedding])
    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(DATA_FILE, data)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

# --- CONVERSATIONAL CONTEXT AND RESPONSE GENERATION --- #
def find_aggregated_embedding_context(user_input):
    user_input_embedding = vectorizer.transform([user_input]).toarray()
    aggregated_embedding = aggregate_embeddings(embeddings, user_input_embedding)
    closest_idx = np.argmax(cosine_similarity(embeddings, aggregated_embedding.reshape(1, -1)))
    return [data[closest_idx]['input'], data[closest_idx]['output']]


def find_most_relevant_contexts(user_input, top_N=2):
    user_input_embedding = vectorizer.transform([user_input])
    similarities = cosine_similarity(embeddings, user_input_embedding)
    flattened_similarities = similarities.ravel()

    for idx, item in enumerate(data):
        timestamp = item.get("timestamp", datetime.now())
        flattened_similarities[idx] = decayed_similarity(flattened_similarities[idx], timestamp)

    # Exclude the most recent interaction
    most_recent_idx = len(data) - 1
    flattened_similarities[most_recent_idx] = 0

    most_relevant_indices = np.argsort(flattened_similarities)[-top_N:]
    contexts = [(data[idx]['input'], data[idx]['output']) for idx in most_relevant_indices]
    return contexts


def choose_approach_based_on_sentiment(user_input):
    sentiment_value = compute_sentiment(user_input)
    
    # Define sentiment thresholds (these can be adjusted as needed)
    positive_threshold = 0.5
    negative_threshold = -0.5

    # If sentiment is strongly positive or negative
    if sentiment_value > positive_threshold or sentiment_value < negative_threshold:
        return find_most_relevant_contexts(user_input)
    else:
        # Ensure that the aggregated context is returned as a list containing one tuple
        return [find_aggregated_embedding_context(user_input)]

    
def ask_ai(user_input):
    # Retrieve the last few exchanges from history
    history = load_history()
    
    # Construct the dialogue history
    dialogue_history = ""
    for exchange in history:
        dialogue_history += f"USER: {exchange['user_input']}\nASSISTANT: {exchange['bot_response']}\n"

    # Using the GPT4All model's expected format:
    system_template = SYSTEM_PROMPT
    prompt_template = 'USER: {0}\nASSISTANT: '

    # Construct the prompt using the system_template, dialogue history, and user input
    complete_prompt = f"{system_template}\n\n{dialogue_history}{prompt_template.format(user_input)}"

    # Print the generated prompt for debugging:
    print("Generated Prompt:", complete_prompt)
    
    # Use the model to generate the response
    with model.chat_session():
        response = model.generate(complete_prompt, temp=0.2, max_tokens=800)

    return response



# --- UTILITY FUNCTIONS --- #

def compute_sentiment(text):
    blob = TextBlob(text)
    # The polarity is a float within the range [-1.0, 1.0]
    # where -1 means negative sentiment, 1 means positive sentiment, and 0 is neutral.
    return blob.sentiment.polarity

# --- MAIN CONVERSATION LOOP --- #


def main_conversation_loop():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        response = ask_ai(user_input)
        
        # Check against recent history to prevent repetitiveness
        while is_in_recent_history(response):
            response = ask_ai(user_input)

        print("Bot:", response)

        # Update and save data
        update_and_save_data(user_input, response)
        
        # Save the response and user input to the recent history
        save_to_history(user_input, response)



if __name__ == "__main__":
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as file:
            json.dump({"exchanges": []}, file)
    main_conversation_loop()
