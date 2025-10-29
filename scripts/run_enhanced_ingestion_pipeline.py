#!/usr/bin/env python3
"""
Run Enhanced Ingestion Pipeline for Education Policy Documents

This script runs the comprehensive 7-stage ingestion pipeline that extracts
entities, relations, and builds bridge table connections during processing.

Usage:
    python scripts/run_enhanced_ingestion_pipeline.py [options]

Options:
    --docs-dir: Directory containing documents to process
    --output-dir: Directory to save processed outputs
    --max-docs: Maximum number of documents to process (for testing)
    --test-mode: Run with a small subset for testing
    --quality-threshold: Minimum quality score to accept documents
    --log-level: Logging level (DEBUG, INFO, WARNING, ERROR)
"""

import argparse
import logging
import sys
from pathlib import Path
import time
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion import EnhancedIngestionPipeline
from src.utils.logger import get_logger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion_pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Enhanced Ingestion Pipeline for Education Policy Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all documents in default directory
    python scripts/run_enhanced_ingestion_pipeline.py
    
    # Process specific directory with custom output
    python scripts/run_enhanced_ingestion_pipeline.py --docs-dir /path/to/docs --output-dir /path/to/output
    
    # Test mode with 5 documents
    python scripts/run_enhanced_ingestion_pipeline.py --test-mode --max-docs 5
    
    # Process with quality filtering
    python scripts/run_enhanced_ingestion_pipeline.py --quality-threshold 60
        """
    )
    
    # Input/output arguments
    parser.add_argument(
        '--docs-dir',
        type=str,
        default=str(project_root / "data" / "raw" / "Documents"),
        help='Directory containing documents to process (default: data/raw/Documents)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(project_root / "data" / "processed"),
        help='Directory to save processed outputs (default: data/processed)'
    )
    
    # Processing control arguments
    parser.add_argument(
        '--max-docs',
        type=int,
        help='Maximum number of documents to process (useful for testing)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run in test mode with limited documents and detailed logging'
    )
    
    parser.add_argument(
        '--quality-threshold',
        type=int,
        default=40,
        help='Minimum quality score to accept documents (default: 40)'
    )
    
    # Logging and debugging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate processing results for debugging'
    )
    
    # Pipeline configuration
    parser.add_argument(
        '--skip-deduplication',
        action='store_true',
        help='Skip duplicate detection phase'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=700,
        help='Chunk size in characters (default: 700)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=100,
        help='Chunk overlap in characters (default: 100)'
    )
    
    return parser.parse_args()


def setup_directories(output_dir: str):
    """Setup required output directories."""
    output_path = Path(output_dir)
    
    # Create main output directories
    directories = [
        output_path,
        output_path / "chunks",
        output_path / "entities",
        output_path / "relations",
        output_path / "metadata",
        output_path / "quality_reports",
        Path(project_root / "logs" / "ingestion")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def validate_inputs(docs_dir: str, output_dir: str):
    """Validate input parameters."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.error(f"Documents directory does not exist: {docs_dir}")
        sys.exit(1)
    
    # Check if there are any PDF files
    pdf_files = list(docs_path.rglob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in: {docs_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_files)} PDF files in {docs_dir}")
    
    # Validate output directory
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Cannot create output directory {output_dir}: {e}")
        sys.exit(1)


def print_pipeline_summary(results: dict):
    """Print summary of pipeline execution."""
    stats = results.get('pipeline_stats', {})
    corpus_stats = results.get('statistics', {})
    
    print("\n" + "="*80)
    print("ENHANCED INGESTION PIPELINE SUMMARY")
    print("="*80)
    
    # Processing statistics
    print(f"Documents Processed: {stats.get('documents_processed', 0)}")
    print(f"Successful: {stats.get('documents_successful', 0)}")
    print(f"Failed: {stats.get('documents_failed', 0)}")
    
    success_rate = stats.get('documents_successful', 0) / max(stats.get('documents_processed', 1), 1)
    print(f"Success Rate: {success_rate:.1%}")
    
    # Content statistics
    print(f"\nContent Generated:")
    print(f"Total Chunks: {stats.get('total_chunks', 0):,}")
    print(f"Total Entities: {stats.get('total_entities', 0):,}")
    print(f"Total Relations: {stats.get('total_relations', 0):,}")
    print(f"Topics Matched: {stats.get('total_topics_matched', 0):,}")
    
    # Averages
    successful_docs = max(stats.get('documents_successful', 1), 1)
    print(f"\nAverages per Document:")
    print(f"Chunks: {stats.get('total_chunks', 0) / successful_docs:.1f}")
    print(f"Entities: {stats.get('total_entities', 0) / successful_docs:.1f}")
    print(f"Relations: {stats.get('total_relations', 0) / successful_docs:.1f}")
    
    # Quality distribution
    quality_dist = stats.get('quality_distribution', {})
    print(f"\nQuality Distribution:")
    for level, count in quality_dist.items():
        print(f"  {level.title()}: {count}")
    
    # Errors
    errors = stats.get('processing_errors', [])
    if errors:
        print(f"\nErrors: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error.get('doc_id', 'Unknown')}: {error.get('error', 'Unknown error')}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    
    print("="*80)


def save_execution_report(results: dict, args, execution_time: float, output_dir: str):
    """Save detailed execution report."""
    report = {
        "execution_info": {
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": execution_time,
            "execution_time_formatted": f"{execution_time/60:.1f} minutes",
            "arguments": vars(args),
            "pipeline_version": "2.0.0"
        },
        "results": results
    }
    
    report_file = Path(output_dir) / "execution_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Execution report saved to: {report_file}")


def main():
    """Main execution function."""
    start_time = time.time()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Print startup banner
    logger.info("="*80)
    logger.info("ENHANCED INGESTION PIPELINE STARTING")
    logger.info("="*80)
    logger.info(f"Documents Directory: {args.docs_dir}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Max Documents: {args.max_docs or 'All'}")
    logger.info(f"Test Mode: {args.test_mode}")
    logger.info(f"Quality Threshold: {args.quality_threshold}")
    logger.info("="*80)
    
    try:
        # Validate inputs
        validate_inputs(args.docs_dir, args.output_dir)
        
        # Setup directories
        setup_directories(args.output_dir)
        
        # Adjust parameters for test mode
        if args.test_mode:
            if not args.max_docs:
                args.max_docs = 5
            args.log_level = 'DEBUG'
            args.save_intermediate = True
            logger.info("Test mode activated - processing limited documents with detailed logging")
        
        # Initialize pipeline
        logger.info("Initializing Enhanced Ingestion Pipeline...")
        pipeline = EnhancedIngestionPipeline(
            data_dir=str(project_root / "data"),
            output_dir=args.output_dir
        )
        
        # Run pipeline
        logger.info("Starting document processing...")
        results = pipeline.process_corpus(
            documents_dir=args.docs_dir,
            max_documents=args.max_docs
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Print summary
        print_pipeline_summary(results)
        
        # Save execution report
        save_execution_report(results, args, execution_time, args.output_dir)
        
        # Check for critical issues
        stats = results.get('pipeline_stats', {})
        success_rate = stats.get('documents_successful', 0) / max(stats.get('documents_processed', 1), 1)
        
        if success_rate < 0.5:
            logger.warning("Low success rate detected - check logs for issues")
        
        critical_quality = stats.get('quality_distribution', {}).get('critical', 0)
        if critical_quality > 0:
            logger.warning(f"{critical_quality} documents have critical quality issues")
        
        logger.info(f"Pipeline completed successfully in {execution_time/60:.1f} minutes")
        logger.info(f"Results saved to: {args.output_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.error("Check logs for detailed error information")
        return 1


if __name__ == "__main__":
    sys.exit(main())