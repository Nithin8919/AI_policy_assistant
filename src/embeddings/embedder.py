"""Vector embedding generation with batch processing, checkpointing, and bridge table integration"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
import logging
import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils.logger import get_logger
from src.knowledge_graph.relation_extractor import RelationExtractor, ChunkAnalysis
from src.knowledge_graph.bridge_builder import BridgeTableBuilder
from src.knowledge_graph.supersession_tracker import SupersessionTracker

logger = get_logger(__name__)

@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    chunk_id: str
    doc_id: str
    embedding: List[float]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    # Bridge table integration fields
    extracted_entities: List[Dict[str, Any]] = None
    extracted_relations: List[Dict[str, Any]] = None
    bridge_topic_matches: List[str] = None

@dataclass
class EmbeddingStats:
    """Statistics for embedding generation"""
    total_chunks: int = 0
    successful_embeddings: int = 0
    failed_embeddings: int = 0
    total_time: float = 0.0
    avg_time_per_chunk: float = 0.0
    avg_embedding_magnitude: float = 0.0
    # Bridge table integration stats
    total_entities_extracted: int = 0
    total_relations_extracted: int = 0
    total_bridge_matches: int = 0
    bridge_topics_updated: int = 0

class Embedder:
    """Generate embeddings using sentence transformers with batch processing, checkpointing, and bridge table integration"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 100,
        enable_bridge_integration: bool = False,
        education_terms_path: Optional[str] = None,
        seed_bridge_topics_path: Optional[str] = None
    ):
        """
        Initialize the embedder
        
        Args:
            model_name: Name of the sentence transformer model
            batch_size: Batch size for processing
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Save checkpoint every N chunks
            enable_bridge_integration: Enable bridge table and relation extraction
            education_terms_path: Path to education terms dictionary
            seed_bridge_topics_path: Path to seed bridge topics file
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("data/embeddings")
        self.checkpoint_interval = checkpoint_interval
        self.enable_bridge_integration = enable_bridge_integration
        
        logger.info(f"Initializing embedder with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize bridge table components if enabled
        self.relation_extractor = None
        self.bridge_builder = None
        self.supersession_tracker = None
        
        if enable_bridge_integration:
            logger.info("Initializing bridge table integration components...")
            
            # Initialize relation extractor
            self.relation_extractor = RelationExtractor(education_terms_path)
            
            # Initialize bridge table builder
            self.bridge_builder = BridgeTableBuilder()
            
            # Load seed bridge topics if provided
            if seed_bridge_topics_path:
                self.bridge_builder.load_seed_topics(seed_bridge_topics_path)
            
            # Initialize supersession tracker
            self.supersession_tracker = SupersessionTracker()
            
            logger.info("Bridge table integration enabled")
        
        # Initialize stats
        self.stats = EmbeddingStats()
    
    def embed_single(self, text: str, chunk_id: str = "", doc_id: str = "") -> EmbeddingResult:
        """Generate embedding for a single text"""
        start_time = time.time()
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            embedding_list = embedding.tolist()
            
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                embedding=embedding_list,
                processing_time=processing_time,
                success=True,
                extracted_entities=[],
                extracted_relations=[],
                bridge_topic_matches=[]
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to generate embedding for chunk {chunk_id}: {e}")
            
            return EmbeddingResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                embedding=[],
                processing_time=processing_time,
                success=False,
                error_message=str(e),
                extracted_entities=[],
                extracted_relations=[],
                bridge_topic_matches=[]
            )
    
    def embed_batch(self, texts: List[str], chunk_ids: List[str] = None, doc_ids: List[str] = None) -> List[EmbeddingResult]:
        """Generate embeddings for a batch of texts"""
        if chunk_ids is None:
            chunk_ids = [f"chunk_{i}" for i in range(len(texts))]
        if doc_ids is None:
            doc_ids = ["" for _ in range(len(texts))]
        
        start_time = time.time()
        results = []
        
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
            processing_time = time.time() - start_time
            
            # Create results
            for i, embedding in enumerate(embeddings):
                results.append(EmbeddingResult(
                    chunk_id=chunk_ids[i],
                    doc_id=doc_ids[i],
                    embedding=embedding.tolist(),
                    processing_time=processing_time / len(texts),  # Average time per chunk
                    success=True
                ))
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to generate batch embeddings: {e}")
            
            # Create error results for all texts
            for i in range(len(texts)):
                results.append(EmbeddingResult(
                    chunk_id=chunk_ids[i],
                    doc_id=doc_ids[i],
                    embedding=[],
                    processing_time=processing_time / len(texts),
                    success=False,
                    error_message=str(e)
                ))
        
        return results
    
    def load_chunks_from_jsonl(self, jsonl_path: str) -> Iterator[Dict[str, Any]]:
        """Load chunks from JSONL file"""
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line.strip())
                    yield chunk
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse line {line_num} in {jsonl_path}: {e}")
                    continue
    
    def generate_embeddings_from_jsonl(
        self,
        jsonl_path: str,
        output_path: Optional[str] = None,
        resume: bool = True
    ) -> Tuple[List[EmbeddingResult], EmbeddingStats]:
        """
        Generate embeddings from JSONL file with checkpointing
        
        Args:
            jsonl_path: Path to the JSONL file containing chunks
            output_path: Path to save embeddings (optional)
            resume: Whether to resume from checkpoint
            
        Returns:
            Tuple of (embedding_results, stats)
        """
        logger.info(f"Starting embedding generation from {jsonl_path}")
        
        # Setup checkpoint file
        checkpoint_file = self.checkpoint_dir / "embedding_checkpoint.pkl"
        processed_chunks = set()
        
        # Load checkpoint if resuming
        if resume and checkpoint_file.exists():
            logger.info("Loading checkpoint...")
            try:
                with open(checkpoint_file, 'rb') as f:
                    processed_chunks = pickle.load(f)
                logger.info(f"Resumed from checkpoint: {len(processed_chunks)} chunks already processed")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                processed_chunks = set()
        
        # Load all chunks first to count total
        all_chunks = list(self.load_chunks_from_jsonl(jsonl_path))
        total_chunks = len(all_chunks)
        logger.info(f"Total chunks to process: {total_chunks}")
        
        # Filter out already processed chunks
        chunks_to_process = [chunk for chunk in all_chunks if chunk.get('chunk_id', '') not in processed_chunks]
        logger.info(f"Chunks remaining to process: {len(chunks_to_process)}")
        
        # Initialize stats
        self.stats = EmbeddingStats(total_chunks=total_chunks)
        all_results = []
        
        # Process in batches
        start_time = time.time()
        
        with tqdm(total=len(chunks_to_process), desc="Generating embeddings") as pbar:
            for i in range(0, len(chunks_to_process), self.batch_size):
                batch_chunks = chunks_to_process[i:i + self.batch_size]
                
                # Extract texts and IDs
                batch_texts = []
                batch_chunk_ids = []
                batch_doc_ids = []
                
                for chunk in batch_chunks:
                    batch_texts.append(chunk.get('content', ''))
                    batch_chunk_ids.append(chunk.get('chunk_id', f'chunk_{i}'))
                    batch_doc_ids.append(chunk.get('doc_id', ''))
                
                # Generate embeddings for batch
                batch_results = self.embed_batch(batch_texts, batch_chunk_ids, batch_doc_ids)
                all_results.extend(batch_results)
                
                # Update stats
                for result in batch_results:
                    if result.success:
                        self.stats.successful_embeddings += 1
                        processed_chunks.add(result.chunk_id)
                    else:
                        self.stats.failed_embeddings += 1
                
                # Save checkpoint
                if (i // self.batch_size + 1) % self.checkpoint_interval == 0:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump(processed_chunks, f)
                    logger.info(f"Checkpoint saved: {len(processed_chunks)} chunks processed")
                
                pbar.update(len(batch_chunks))
        
        # Final stats calculation
        self.stats.total_time = time.time() - start_time
        if self.stats.successful_embeddings > 0:
            self.stats.avg_time_per_chunk = self.stats.total_time / self.stats.successful_embeddings
            
            # Calculate average embedding magnitude
            successful_embeddings = [r.embedding for r in all_results if r.success and r.embedding]
            if successful_embeddings:
                magnitudes = [np.linalg.norm(emb) for emb in successful_embeddings]
                self.stats.avg_embedding_magnitude = float(np.mean(magnitudes))
        
        # Save final results if output path specified
        if output_path:
            self.save_embeddings(all_results, output_path)
        
        # Clean up checkpoint
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        logger.info(f"Embedding generation complete. Success: {self.stats.successful_embeddings}, Failed: {self.stats.failed_embeddings}")
        
        return all_results, self.stats
    
    def save_embeddings(self, results: List[EmbeddingResult], output_path: str):
        """Save embedding results to file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSONL for easy loading
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                if result.success:
                    record = {
                        'chunk_id': result.chunk_id,
                        'doc_id': result.doc_id,
                        'embedding': result.embedding,
                        'processing_time': result.processing_time
                    }
                    f.write(json.dumps(record) + '\n')
        
        logger.info(f"Saved {sum(1 for r in results if r.success)} embeddings to {output_path}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.model.get_sentence_embedding_dimension()
    
    def get_stats(self) -> EmbeddingStats:
        """Get current embedding statistics"""
        return self.stats