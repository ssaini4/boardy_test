# FastAPI Chatbot with Semantic Caching

A FastAPI server that functions as a chatbot with text embedding capabilities. Users can interact via terminal, and the server processes text through a semantic cache for similar queries or LLM APIs to generate responses.

## Features

- 🤖 **FastAPI Server**: RESTful API with automatic documentation
- 🧠 **Text Embedding**: Uses sentence-transformers for text embedding
- 💬 **LLM Integration**: OpenAI GPT integration for chat responses
- 🖥️ **Terminal Client**: Rich terminal interface for easy interaction
- 🏥 **Health Monitoring**: Built-in health checks and logging
- 🐳 **Docker Ready**: Includes Docker configuration


## DEMO Video: https://youtu.be/t1e848-j1Wo

## Getting Started

### Prerequisites


Before running the application, you'll need to set up several services and environment variables:

1. **Setup Local Services with Dev Container and Docker**

   - Make sure you have [Docker](https://docs.docker.com/get-docker/), [NodeJS](https://nodejs.org/en/download/), and [npm](https://www.npmjs.com/get-npm) installed.
   - Open codebase as a container in [VSCode](https://code.visualstudio.com/) or your favorite VSCode fork using `Cmd + Shift + P`

   <b>NOTE: This will start a dev container and a redis server using https://github.com/ssaini4/boardy_test/docker-compose.yml</b>


## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the environment template and add your OpenAI API key:

```bash
cp env.template .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Start the Server

You can start the server in multiple ways:

**Option 1: Using the start script**
```bash
cd src
python start_server.py
```

**Option 2: Direct uvicorn**
```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 3000
```

**Option 3: Using Python directly**
```bash
cd src
python main.py
```

The server will be available at:
- API: http://localhost:3000
- Documentation: http://localhost:3000/docs
- Health Check: http://localhost:3000/health

### 4. Use the Terminal Client

Open a new terminal and run the interactive client:

```bash
cd src
python client.py
```

Or send a single message:

```bash
cd src
python client.py --message "Hello, how are you?"
```

## API Endpoints

### `/chat` (POST)
Full chat endpoint with embedding and LLM response
- Input: `{"query": "your query", "forceRefresh": false \\ Optional parameter to bypass cache}`
- Output: `{"response": "llm response", "metadata": {"source": "cache"} // 'cache' or 'llm'}`


### `/health` (GET)
Health check endpoint
- Output: Server status and model availability

### `/metrics` (GET)
Redis info endpoint
- Output: Redis metrics and metadata

### `/print_metrics` (GET)
Redis info endpoint
- Output: Prints Redis metrics and metadata in server logs

## Terminal Client Commands

- **Normal chat**: Just type your message
- **`quit`/`exit`**: Exit the client
- **`help`**: Show available commands
- **`health`**: Check server health

## Example Usage

1. **Start the server**:
```bash
cd src && python start_server.py
```

2. **In another terminal, start the client**:
```bash
cd src && python client.py
```

3. **Chat with the bot**:
```
🤖 FastAPI Chatbot Terminal Client

You: Hello! Can you help me understand machine learning?

🤔 Thinking...
```

## Architecture Overview

### Overall Architecture
The application relies on a FastAPI server with a Redis cache that server as a Least Recently Used (LRU) cache to store the incoming queries and data generated to process queries.

### FastAPI server
FastAPI is a widely used backend framework with flexibility to horizontally scale the endpoints and applications as required. FastAPI also provides Swagger APIs out of the box so developers get the benefit of documentation.
We are using uvicorn servers with worker configuration which makes it easy to scale the number of workers based on our infrastructure. 

### LRU Cache

Redis is the Swiss army knife of backend development. For this application, we are using Redis in two-configurations - text cache and semantic cache.

For this application, we are using the LRU configuration and store the most recent 1000 items. This can be easily increase and scaled to accomodate for higher traffic and data.

#### Text Cache

Simple query message <> response cache to store most recent N queries. If there is a duplicate query received, the server simply returns the cached response.


#### Semantic Cache

Float32 cache which stores the query message, response, embedding and timestamp. 

### Request Flow

1. User sends query message to `POST /api/query`
2. If `force_refresh` is True, send response from LLM API
3. If `forch_refresh` is False, check if query is [`time_sensitive`](https://github.com/ssaini4/boardy_test/blob/a93863cc8029d5a5f67796ddeea58b699ccd852c/src/helper.py#L1), e.g. `What is the weather like in NYC today?`
4. If query is time, i.e. it includes time-sensitive words, send response from LLM API
5. If query is NOT time sensitive, check if there is a query hit in `text_cache`. If there is a hit, return cached response.
6. If there is no hit in `text_cache`, check `semantic_cache` 

### Semantic Cache Flow

1. Convert the text query into embeddings using OpenAI API. We use `text-embedding-3-small` because it provides a good performance vs cost balance. It also the fastest embedding model from OpenAI.
2. We use the query embeddings and run a KNN search to get the 50(configurable) nearest neighbors for the embeddings using cosine distance. Cosine Distance is robust in semantic search and RAG applications. This is because the magnitudes of the vectors does not matter, only the direction does. Therefore making is the better choice for language similarity tasks
3. Next, [we will extract entities from the query](https://github.com/ssaini4/boardy_test/blob/a93863cc8029d5a5f67796ddeea58b699ccd852c/src/lru.py#L312). We determine entities to add more weight in our calculation of the final cached response. Take for instance queries: "Who was the first president of USA?" and "First president of Mexico. Tell me". Here, the semantic meaning of the queries is the same but the entities are different - USA & Mexico. Therefore, more weight needs to be given in the final computation to the entities.
4. We extract entities in the categories of - persons, geo-political entries, locations and organizations. We use [spaCy](https://spacy.io/) to run local categorization for its performance and offline capabilities - <b>this saves money and time.</b>.
5. We iterate through all kNN neighbors and find the exact cosine distance between the user query and the neighbor embeddings. This will allow us to rank the neighbors. We also extract the entities from each of the cached neighbors and do a fuzzy match comparison with the entities from the user query and give a score accordingly.
6. Now we compute the combined score using:
```python
# RERANK_ALPHA: weight for exact cosine
# cosine_distance_embeddings: exact cosine distance between user query and neighbor embeddings
# ENTITY_WEIGHT: 1.0 - RERANK_ALPHA  # weight for entity match
# entity_score: 1 if both embeddings have entities, else 0
score = (RERANK_ALPHA * cosine_distance_embeddings) + (ENTITY_WEIGHT * entity_score) 
```
7. Once we have the scores for all neighbors, we check if the highest score passes our `SIMILARITY_THRESHOLD` and return the cached response if it does. Otherwise, we return the LLM response.
8. Cache values are updated both both LLM and cached responses to ensure LRU cache is up to date and older values expire.



## Future Improvements
1. The application is built so that you can add additional application servers with a load balancer for the FastAPI server
2. For Redis server, we can add multiple read replicas and use bigger cluster based on our traffic and data.
3. After running some load tests with [`test.py`](https://github.com/ssaini4/boardy_test/blob/a93863cc8029d5a5f67796ddeea58b699ccd852c/src/test.py) , I found four bottlenecks:

   1. Embedding generation: Currently we use OpenAI API to compute our embeddings. This can take a long time depending on our network, OpenAI's systems and network hops. To overcome this, we can use [Sentence Transformers](https://huggingface.co/sentence-transformers) with a suitable model to create our embeddings locally.
   2. KNN embeddings: This primarily happens when our cache gets too big with a lot of sparse data. To avoid this, we can maintain separate redis cache indexes based on topics (e.g. politics, sports, food, etc.). We can use a local BERT model to classify the user query into these topics and hit the corresponding redis cache index to narrow our search space.
   3. Static Output: Currently when the LLM response is returned, we are returning a static response which takes longer. We can return a streaming output to improve response time and improve customer experience.
   4. Cosine similarity in a linear loop: We currently iterate over the KNN neighbors in a for loop to calculate cosine similarity. Instead we can optimize this matrix multiplication by vectorizing all the neighbors in a big normalized matrix and calculating a dot product with the user query embedding to get the cosine similarities in a matrix.