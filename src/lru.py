#!/usr/bin/env python3
from enum import Enum
import re
from typing import Any, Optional, Set, Dict
from redis import Redis
import pickle
import numpy as np
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import time
import logging
import threading
from dataclasses import dataclass, asdict

SIMILARITY_THRESHOLD = 0.55
MIN_RESULTS = 10
# Rerank / entity-check hyperparams
RERANK_TOP_K = 50  # number of neighbors to fetch for rerank (<= index K)
RERANK_ALPHA = 0.85  # weight for exact cosine
ENTITY_WEIGHT = 1.0 - RERANK_ALPHA  # weight for entity match
ENTITY_MATCH_THRESHOLD = 0.0  # if any overlap -> entity_score = 1.0, else 0.0


logger = logging.getLogger(__name__)


class CacheType(Enum):
    TEXT = "text"
    SEMANTIC = "semantic"


@dataclass
class CacheMetrics:
    """Comprehensive cache performance metrics"""

    # Hit/Miss counts
    hits: int = 0
    misses: int = 0

    # Latency tracking (in seconds)
    total_latency: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0

    # Operation counts
    get_operations: int = 0
    set_operations: int = 0
    has_operations: int = 0
    semantic_operations: int = 0

    # Semantic-specific metrics
    semantic_hits: int = 0
    semantic_misses: int = 0
    semantic_total_latency: float = 0.0

    # Start time for calculating rates
    start_time: float = time.time()

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as percentage"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    @property
    def semantic_hit_rate(self) -> float:
        """Calculate semantic hit rate as percentage"""
        total = self.semantic_hits + self.semantic_misses
        return (self.semantic_hits / total * 100) if total > 0 else 0.0

    @property
    def avg_latency(self) -> float:
        """Calculate average latency"""
        total_ops = self.get_operations + self.has_operations + self.set_operations
        return (self.total_latency / total_ops) if total_ops > 0 else 0.0

    @property
    def semantic_avg_latency(self) -> float:
        """Calculate average semantic operation latency"""
        return (
            (self.semantic_total_latency / self.semantic_operations)
            if self.semantic_operations > 0
            else 0.0
        )

    @property
    def operations_per_second(self) -> float:
        """Calculate operations per second"""
        elapsed = time.time() - self.start_time
        total_ops = self.get_operations + self.has_operations + self.set_operations
        return total_ops / elapsed if elapsed > 0 else 0.0

    @property
    def uptime_minutes(self) -> float:
        """Get uptime in minutes"""
        return (time.time() - self.start_time) / 60

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for easy serialization"""
        base_dict = asdict(self)
        # Add computed properties
        base_dict.update(
            {
                "hit_rate": self.hit_rate,
                "semantic_hit_rate": self.semantic_hit_rate,
                "avg_latency": self.avg_latency,
                "semantic_avg_latency": self.semantic_avg_latency,
                "operations_per_second": self.operations_per_second,
                "uptime_minutes": self.uptime_minutes,
            }
        )
        return base_dict


class LRUCache:
    """
    A Least Recently Used (LRU) cache keeps items in the cache until it reaches its size
    and/or item limit (only item in our case). In which case, it removes an item that was accessed
    least recently.
    An item is considered accessed whenever `has`, `get`, or `set` is called with its key.

    Redis-based implementation of LRU cache with comprehensive metrics tracking.
    """

    def __init__(
        self,
        item_limit: int,
        redis_host: str = "redis_db",
        redis_port: int = 6379,
        redis_db: int = 0,
        cache_prefix: str = "lru_cache",
        cache_type: CacheType = CacheType.TEXT,
    ):
        self.item_limit = item_limit
        self.redis = Redis(host=redis_host, port=redis_port, db=redis_db, decode_responses=False)
        self.cache_prefix = cache_prefix
        self.lru_list_key = f"{cache_prefix}:lru_order"
        self.cache_type = cache_type

        # Initialize metrics tracking
        self.metrics = CacheMetrics()
        self._metrics_lock = threading.Lock()

        if cache_type == CacheType.SEMANTIC:
            self.similarity_threshold = SIMILARITY_THRESHOLD
            self._create_cache_index()
            # Import OpenAI for embeddings (add to requirements.txt if not present)
            try:
                import openai

                self.openai_client = openai.OpenAI()
            except ImportError:
                raise ImportError(
                    "OpenAI package required for semantic cache. Install with: pip install openai"
                )

            # Try to import spaCy for better NER; optional fallback available
            try:
                import spacy

                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except Exception:
                    # In case the model itself isn't installed, try to load blank English model
                    self._nlp = spacy.blank("en")
            except Exception:
                self._nlp = None  # fallback to heuristic

        else:
            self.similarity_threshold = None
            self.openai_client = None

        # Test Redis connection
        try:
            self.redis.ping()
        except Exception as e:
            raise ConnectionError(f"Could not connect to Redis: {e}")

    def _record_operation(self, operation_type: str, latency: float, hit: bool = False):
        """Thread-safe metrics recording"""
        with self._metrics_lock:
            # Update latency stats
            self.metrics.total_latency += latency
            if latency < self.metrics.min_latency:
                self.metrics.min_latency = latency
            if latency > self.metrics.max_latency:
                self.metrics.max_latency = latency

            # Update operation counts
            if operation_type == "get":
                self.metrics.get_operations += 1
            elif operation_type == "has":
                self.metrics.has_operations += 1
            elif operation_type == "set":
                self.metrics.set_operations += 1
            elif operation_type == "semantic":
                self.metrics.semantic_operations += 1
                self.metrics.semantic_total_latency += latency
                if hit:
                    self.metrics.semantic_hits += 1
                else:
                    self.metrics.semantic_misses += 1

            # Update hit/miss counts (for non-semantic operations)
            if operation_type in ["get", "has"]:
                if hit:
                    self.metrics.hits += 1
                else:
                    self.metrics.misses += 1

    def _create_cache_index(self, vector_dim=1536):
        """Create Redis search index for semantic similarity search."""
        if self.cache_type != CacheType.SEMANTIC:
            return

        schema = (
            TextField("query"),
            TextField("response"),
            TextField("timestamp"),
            VectorField(
                "embedding",
                "HNSW",
                {"TYPE": "FLOAT32", "DIM": vector_dim, "DISTANCE_METRIC": "COSINE"},
            ),
        )

        try:
            self.redis.ft(self.cache_prefix).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[f"{self.cache_prefix}:semantic:"], index_type=IndexType.HASH
                ),
            )
        except Exception:
            pass  # Index might already exist

    def _get_embedding(self, text: str) -> list:
        """Generate embedding for text using OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized for semantic cache")

        response = self.openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        return response.data[0].embedding

    def _cache_key(self, key: str) -> str:
        """Generate the Redis key for cache storage."""
        return f"{self.cache_prefix}:data:{key}"

    def _semantic_key(self, key: str) -> str:
        """Generate the Redis key for semantic cache storage."""
        return f"{self.cache_prefix}:semantic:{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from Redis storage."""
        return pickle.loads(data)

    def _move_to_end(self, key: str):
        """Move key to the end of the LRU list (most recently used)."""
        # Remove the key from wherever it is in the list
        self.redis.lrem(self.lru_list_key, 0, key)
        # Add it to the end (most recently used)
        self.redis.rpush(self.lru_list_key, key)

    def has(self, key: str) -> bool:
        """Check if key exists in cache and update LRU order."""
        start_time = time.time()

        cache_key = self._cache_key(key)
        exists = self.redis.exists(cache_key)

        if exists:
            self._move_to_end(key)

        latency = time.time() - start_time
        self._record_operation("has", latency, hit=exists)

        return bool(exists)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache and update LRU order."""
        start_time = time.time()

        cache_key = self._cache_key(key)
        data = self.redis.get(cache_key)

        hit = data is not None
        result = None

        if hit:
            self._move_to_end(key)
            result = self._deserialize(data)

        latency = time.time() - start_time
        self._record_operation("get", latency, hit=hit)

        return result

    @staticmethod
    def _exact_cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Compute exact cosine similarity between two 1D numpy arrays."""
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        na = a / (np.linalg.norm(a) + 1e-12)
        nb = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(na, nb))

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract simple named entities. Prefer spaCy if available, else simple heuristic."""
        if not text:
            return set()

        # Try spaCy NER if we have it
        if self._nlp:
            try:
                doc = self._nlp(text)
                ents = {
                    ent.text.strip().lower()
                    for ent in doc.ents
                    if ent.label_ in ("PERSON", "GPE", "LOC", "ORG")
                }
                if ents:
                    return ents
            except Exception:
                # fallthrough to heuristic
                pass

        # Simple heuristic: capture capitalized words (not at sentence start only)
        words = re.findall(r"\b[A-Z][a-zA-Z0-9\-']+\b", text)
        # filter common starting words like "Who" "What" etc.
        stop = {"Who", "What", "When", "Where", "Why", "How", "First", "Last", "The", "A", "An"}
        ents = {w.strip().lower() for w in words if w not in stop}
        return ents

    # -----------------------------
    # Enhanced get_similar with rerank + entity-check + metrics
    # -----------------------------
    def get_similar(self, query: str) -> Optional[Any]:
        """Get similar query response from semantic cache using KNN -> exact rerank -> entity-check."""
        start_time = time.time()

        if self.cache_type != CacheType.SEMANTIC or not self.openai_client:
            return None

        try:
            # 1) Compute query embedding
            query_embedding = np.array(self._get_embedding(query), dtype=np.float32)
            query_vector = query_embedding.tobytes()

            # 2) KNN search (fetch candidates)
            knn_k = max(MIN_RESULTS, RERANK_TOP_K)
            search_query = (
                Query(f"*=>[KNN {knn_k} @embedding $vector AS score]")
                # we only need 'query' and 'response' fields returned for later entity checks
                .return_fields("query", "response", "timestamp", "score").dialect(2)
            )

            results = self.redis.ft(self.cache_prefix).search(
                search_query, query_params={"vector": query_vector}
            )

            # If no candidates, return None
            if not results.docs:
                latency = time.time() - start_time
                self._record_operation("semantic", latency, hit=False)
                return None

            # 3) Exact rerank:
            best_candidate = None
            best_combined_score = -1.0

            # Extract query entities once
            query_entities = self._extract_entities(query)

            for doc in results.docs:
                # doc.id is the Redis key for the hash (e.g. "lru_cache:semantic:<key>")
                doc_id = getattr(doc, "id", None)
                # fetch stored embedding for this candidate
                try:
                    emb_bytes = self.redis.hget(doc_id, "embedding")
                    if emb_bytes is None:
                        # maybe the doc.fields already include embedding (unlikely) or key mismatch; skip
                        continue
                    cand_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                except Exception:
                    # fallback: try reading by constructing semantic key from returned field 'query'
                    cand_key_field = getattr(doc, "query", None)
                    if isinstance(cand_key_field, bytes):
                        cand_key_field = cand_key_field.decode("utf-8")
                    if cand_key_field:
                        emb_bytes = self.redis.hget(self._semantic_key(cand_key_field), "embedding")
                        if emb_bytes is None:
                            continue
                        cand_emb = np.frombuffer(emb_bytes, dtype=np.float32)
                    else:
                        continue

                # exact cosine
                try:
                    sim = self._exact_cosine(query_embedding, cand_emb)
                except Exception:
                    continue

                # entity extraction and match scoring
                cand_query_field = getattr(doc, "query", None)
                if isinstance(cand_query_field, bytes):
                    cand_query_field = cand_query_field.decode("utf-8")
                cand_entities = self._extract_entities(cand_query_field or "")

                entity_score = 0.0
                if query_entities and cand_entities and (len(query_entities & cand_entities) > 0):
                    entity_score = 1.0
                elif (not query_entities) and (not cand_entities):
                    # if neither has entities, neutral (0); leave entity_score = 0.0
                    entity_score = 0.0
                else:
                    entity_score = 0.0

                # combine similarity + entity signal
                combined = (RERANK_ALPHA * sim) + (ENTITY_WEIGHT * entity_score)

                # keep track of best
                if combined > best_combined_score:
                    best_combined_score = combined
                    # prepare response
                    cand_response = getattr(doc, "response", None)
                    if isinstance(cand_response, bytes):
                        try:
                            cand_response = cand_response.decode("utf-8")
                        except Exception:
                            # stored as pickled maybe; try to read directly from hash
                            raw = self.redis.hget(doc_id, "response")
                            if raw:
                                try:
                                    cand_response = raw.decode("utf-8")
                                except Exception:
                                    cand_response = raw
                    best_candidate = {
                        "query": cand_query_field,
                        "response": cand_response,
                        "similarity": sim,
                        "entity_score": entity_score,
                        "combined": combined,
                        "redis_id": doc_id,
                    }

            # 4) Decide whether to use best candidate
            hit = False
            result = None

            if best_candidate and best_candidate["combined"] >= (
                self.similarity_threshold or SIMILARITY_THRESHOLD
            ):
                # update LRU order using the original cache key (candidate['query'])
                try:
                    self._move_to_end(best_candidate["query"])
                except Exception:
                    # maybe keys stored as bytes in LRU; attempt decode
                    try:
                        self._move_to_end(best_candidate["query"].decode("utf-8"))
                    except Exception:
                        pass

                # return the candidate response
                hit = True
                result = best_candidate["response"]

            latency = time.time() - start_time
            self._record_operation("semantic", latency, hit=hit)
            return result

        except Exception as e:
            logger.exception("Error in semantic search rerank: %s", e)
            latency = time.time() - start_time
            self._record_operation("semantic", latency, hit=False)

        return None

    # -----------------------------
    # Existing set / clear / size methods with metrics
    # -----------------------------
    def set(self, key: str, value: Any):
        """Set key-value pair in cache, maintaining LRU order and size limit."""
        start_time = time.time()

        cache_key = self._cache_key(key)

        # Check if key already exists
        key_exists = self.redis.exists(cache_key)

        if key_exists:
            # Key exists, just update value and move to end
            self.redis.set(cache_key, self._serialize(value))
            self._move_to_end(key)
        else:
            # New key, check if we need to evict
            current_size = self.redis.llen(self.lru_list_key)

            if current_size >= self.item_limit:
                # Need to evict the least recently used item
                lru_key = self.redis.lpop(self.lru_list_key)
                if lru_key:
                    # lru_key comes as bytes if decode_responses=False
                    if isinstance(lru_key, bytes):
                        lru_key = lru_key.decode("utf-8")
                    self.redis.delete(self._cache_key(lru_key))
                    # Also delete semantic cache entry if it exists
                    if self.cache_type == CacheType.SEMANTIC:
                        self.redis.delete(self._semantic_key(lru_key))

            # Add the new key-value pair
            self.redis.set(cache_key, self._serialize(value))
            self.redis.rpush(self.lru_list_key, key)

            # For semantic cache, also store embedding
            if self.cache_type == CacheType.SEMANTIC and self.openai_client:
                try:
                    embedding = self._get_embedding(key)
                    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

                    semantic_data = {
                        "query": key,
                        "response": value if isinstance(value, str) else str(value),
                        "timestamp": str(int(time.time())),
                        "embedding": embedding_bytes,
                    }

                    # store in a Redis hash at the semantic key
                    self.redis.hset(self._semantic_key(key), mapping=semantic_data)
                except Exception as e:
                    logger.exception("Error storing semantic data: %s", e)

        latency = time.time() - start_time
        self._record_operation("set", latency)

    def clear(self):
        """Clear all cache data."""
        # Get all keys in the LRU list
        keys = self.redis.lrange(self.lru_list_key, 0, -1)

        # Delete all cache data
        if keys:
            cache_keys = []
            for k in keys:
                if isinstance(k, bytes):
                    k = k.decode("utf-8")
                cache_keys.append(self._cache_key(k))
            if cache_keys:
                self.redis.delete(*cache_keys)

            # Also delete semantic cache entries
            if self.cache_type == CacheType.SEMANTIC:
                semantic_keys = [
                    self._semantic_key(k.decode("utf-8") if isinstance(k, bytes) else k)
                    for k in keys
                ]
                if semantic_keys:
                    self.redis.delete(*semantic_keys)

        # Clear the LRU list
        self.redis.delete(self.lru_list_key)

    def size(self) -> int:
        """Get current cache size."""
        return self.redis.llen(self.lru_list_key)

    # -----------------------------
    # New methods for metrics access
    # -----------------------------
    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics"""
        with self._metrics_lock:
            return CacheMetrics(**asdict(self.metrics))

    def get_metrics_dict(self) -> Dict:
        """Get current cache metrics as dictionary"""
        return self.get_metrics().to_dict()

    def print_metrics(self):
        """Print formatted metrics to console"""
        metrics = self.get_metrics_dict()

        print(f"\n{'='*60}")
        print(f"CACHE PERFORMANCE METRICS - {self.cache_type.value.upper()}")
        print(f"{'='*60}")
        print(f"Uptime: {metrics['uptime_minutes']:.1f} minutes")
        print(f"")
        print(f"OPERATIONS:")
        print(f"  GET operations: {metrics['get_operations']}")
        print(f"  HAS operations: {metrics['has_operations']}")
        print(f"  SET operations: {metrics['set_operations']}")
        if self.cache_type == CacheType.SEMANTIC:
            print(f"  SEMANTIC operations: {metrics['semantic_operations']}")
        print(f"  Operations/sec: {metrics['operations_per_second']:.2f}")
        print(f"")
        print(f"HIT/MISS STATS:")
        print(f"  Hits: {metrics['hits']}")
        print(f"  Misses: {metrics['misses']}")
        print(f"  Hit Rate: {metrics['hit_rate']:.1f}%")
        if self.cache_type == CacheType.SEMANTIC:
            print(f"  Semantic Hits: {metrics['semantic_hits']}")
            print(f"  Semantic Misses: {metrics['semantic_misses']}")
            print(f"  Semantic Hit Rate: {metrics['semantic_hit_rate']:.1f}%")
        print(f"")
        print(f"LATENCY STATS:")
        print(f"  Average: {metrics['avg_latency']:.4f}s")
        print(f"  Minimum: {metrics['min_latency']:.4f}s")
        print(f"  Maximum: {metrics['max_latency']:.4f}s")
        if self.cache_type == CacheType.SEMANTIC:
            print(f"  Semantic Avg: {metrics['semantic_avg_latency']:.4f}s")
        print(f"{'='*60}")

    def reset_metrics(self):
        """Reset all metrics counters"""
        with self._metrics_lock:
            self.metrics = CacheMetrics()
            logger.info(f"Cache metrics reset for {self.cache_type.value} cache")

    def log_metrics_periodically(self, interval_seconds: int = 300):
        """Log metrics every N seconds (for background monitoring)"""
        import threading

        def log_metrics():
            while True:
                time.sleep(interval_seconds)
                metrics = self.get_metrics_dict()
                logger.info(
                    f"Cache metrics: hit_rate={metrics['hit_rate']:.1f}%, "
                    f"ops_per_sec={metrics['operations_per_second']:.2f}, "
                    f"avg_latency={metrics['avg_latency']:.4f}s"
                )

        thread = threading.Thread(target=log_metrics, daemon=True)
        thread.start()
        logger.info(f"Started metrics logging every {interval_seconds} seconds")
