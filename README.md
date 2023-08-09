# mnemosyneBOT
==============================

![Mnemosyne](https://github.com/EveryOneIsGross/mnemosyneBOT/assets/23621140/3859e884-143e-46a1-9281-a474dbd98add)

mnemosyneBOT is a conversation agent that generates responses based on user input and considers the historical context via the sentiment of the input. The agent is built on top of the GPT4ALL and uses several novel features to enhance the conversational experience. Drawing inspiration from Mnemosyne, the ancient Greek goddess of memory, this bot dynamically manages its conversational context using sentiment analysis and a unique memory decay mechanism.

## Features
--------

![mnemosyneFLOW py](https://github.com/EveryOneIsGross/mnemosyneBOT/assets/23621140/b26d4284-94b3-441b-8764-ac4c579786dc)


**Memory Decay Mechanism:**

 The agent uses a decay function to give more weight to recent interactions and reduce the influence of older interactions. This simulates the concept of "forgetting" in human memory, ensuring that the agent's responses are more aligned with recent context.

**Sentiment-Based Context Selection:**

 Depending on the sentiment of the user's input, the agent chooses between different methods to retrieve relevant conversational context. This ensures that the agent's responses are more emotionally attuned to the user's sentiments.

**Embedding & Data Handling:**

 The agent uses TF-IDF Vectorization to convert text data into embeddings. These embeddings are then used to calculate similarities and retrieve relevant conversational contexts.

**History Handling:**

 The agent maintains a history of recent interactions to avoid repetitive responses and to provide contextually relevant answers.

**Data Persistence:**

 Embeddings, vectorizers, and interaction data are persistently stored in files, allowing the agent to maintain its state across sessions.

**GPT4ALL Integration:**
 The agent is built on top of the GPT4ALL python api, using a local model.

