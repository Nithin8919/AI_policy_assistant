#!/usr/bin/env python3
"""
Test the Enhanced Ingestion Pipeline + Vertical Builders with Sample Data

Tests the complete pipeline on the Critical Priority documents:
1. Run enhanced ingestion pipeline 
2. Build vertical databases
3. Validate results and generate report

Sample data includes:
- National Education Policy 2020.pdf
- Multiple GOs (Nadu-Nedu, Amma Vodi, etc.)
- GO metadata CSV
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion import EnhancedIngestionPipeline
from src.vertical_builders import LegalDatabaseBuilder, GODatabaseBuilder
from src.utils.logger import get_logger

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = get_logger(__name__)


def setup_test_directories():
    """Setup test output directories."""
    test_output = project_root / "data" / "test_output"
    test_output.mkdir(exist_ok=True)
    
    # Clear previous test results
    import shutil
    if test_output.exists():
        shutil.rmtree(test_output)
    test_output.mkdir(parents=True, exist_ok=True)
    
    return test_output


def test_enhanced_ingestion():
    """Test the enhanced ingestion pipeline on sample data."""
    logger.info("="*60)
    logger.info("TESTING ENHANCED INGESTION PIPELINE")
    logger.info("="*60)
    
    # Test documents directory
    test_docs_dir = project_root / "data" / "raw" / "Documents" / "Critical Priority"
    test_output = setup_test_directories()
    
    logger.info(f"Input directory: {test_docs_dir}")
    logger.info(f"Output directory: {test_output}")
    
    # Initialize pipeline
    pipeline = EnhancedIngestionPipeline(
        data_dir=str(project_root / "data"),
        output_dir=str(test_output)
    )
    
    # Run pipeline on sample documents
    start_time = time.time()
    
    try:
        # Process with max 5 documents for testing
        results = pipeline.process_corpus(
            documents_dir=str(test_docs_dir),
            max_documents=5
        )
        
        processing_time = time.time() - start_time
        
        # Print results
        stats = results.get('pipeline_stats', {})
        
        logger.info(f"✅ Ingestion completed in {processing_time:.1f} seconds")
        logger.info(f"📄 Documents processed: {stats.get('documents_processed', 0)}")
        logger.info(f"✅ Successful: {stats.get('documents_successful', 0)}")
        logger.info(f"❌ Failed: {stats.get('documents_failed', 0)}")
        logger.info(f"📝 Total chunks: {stats.get('total_chunks', 0)}")
        logger.info(f"🏷️  Total entities: {stats.get('total_entities', 0)}")
        logger.info(f"🔗 Total relations: {stats.get('total_relations', 0)}")
        
        return results, str(test_output)
        
    except Exception as e:
        logger.error(f"❌ Ingestion pipeline failed: {e}")
        return None, str(test_output)


def test_vertical_builders(processed_data_dir):
    """Test vertical database builders."""
    logger.info("="*60)
    logger.info("TESTING VERTICAL DATABASE BUILDERS")
    logger.info("="*60)
    
    results = {}
    
    # Test Legal Database Builder
    logger.info("Testing Legal Database Builder...")
    try:
        legal_builder = LegalDatabaseBuilder(
            data_dir=processed_data_dir,
            output_dir=str(Path(processed_data_dir) / "verticals")
        )
        
        legal_results = legal_builder.process_and_save()
        results["legal"] = legal_results
        
        if legal_results.get("status") == "success":
            logger.info(f"✅ Legal DB: {legal_results.get('database_entries', 0)} entries created")
        else:
            logger.error(f"❌ Legal DB failed: {legal_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ Legal builder error: {e}")
        results["legal"] = {"status": "error", "error": str(e)}
    
    # Test GO Database Builder
    logger.info("Testing GO Database Builder...")
    try:
        go_builder = GODatabaseBuilder(
            data_dir=processed_data_dir,
            output_dir=str(Path(processed_data_dir) / "verticals")
        )
        
        go_results = go_builder.process_and_save()
        results["go"] = go_results
        
        if go_results.get("status") == "success":
            logger.info(f"✅ GO DB: {go_results.get('database_entries', 0)} entries created")
        else:
            logger.error(f"❌ GO DB failed: {go_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"❌ GO builder error: {e}")
        results["go"] = {"status": "error", "error": str(e)}
    
    return results


def analyze_results(ingestion_results, vertical_results, output_dir):
    """Analyze and report on test results."""
    logger.info("="*60)
    logger.info("ANALYZING TEST RESULTS")
    logger.info("="*60)
    
    analysis = {
        "test_timestamp": datetime.now().isoformat(),
        "ingestion_success": ingestion_results is not None,
        "vertical_builders": {},
        "sample_data_analysis": {},
        "recommendations": []
    }
    
    # Analyze ingestion results
    if ingestion_results:
        stats = ingestion_results.get('pipeline_stats', {})
        success_rate = stats.get('documents_successful', 0) / max(stats.get('documents_processed', 1), 1)
        
        analysis["ingestion"] = {
            "success_rate": success_rate,
            "documents_processed": stats.get('documents_processed', 0),
            "chunks_created": stats.get('total_chunks', 0),
            "entities_extracted": stats.get('total_entities', 0),
            "relations_found": stats.get('total_relations', 0),
            "quality_distribution": stats.get('quality_distribution', {})
        }
        
        logger.info(f"📊 Ingestion success rate: {success_rate:.1%}")
        logger.info(f"📝 Average chunks per doc: {stats.get('total_chunks', 0) / max(stats.get('documents_successful', 1), 1):.1f}")
        logger.info(f"🏷️  Average entities per doc: {stats.get('total_entities', 0) / max(stats.get('documents_successful', 1), 1):.1f}")
    
    # Analyze vertical builder results
    for vertical_name, result in vertical_results.items():
        analysis["vertical_builders"][vertical_name] = {
            "success": result.get("status") == "success",
            "entries_created": result.get("database_entries", 0),
            "error": result.get("error")
        }
        
        if result.get("status") == "success":
            logger.info(f"✅ {vertical_name.title()} vertical: {result.get('database_entries', 0)} entries")
        else:
            logger.info(f"❌ {vertical_name.title()} vertical: Failed")
    
    # Check specific sample data
    output_path = Path(output_dir)
    
    # Check chunks file
    chunks_file = output_path / "chunks" / "all_chunks.jsonl"
    if chunks_file.exists():
        chunk_count = sum(1 for line in open(chunks_file) if line.strip())
        analysis["sample_data_analysis"]["chunks_in_file"] = chunk_count
        logger.info(f"📁 Chunks file contains {chunk_count} chunks")
        
        # Sample a few chunks to check entity extraction
        sample_chunks = []
        with open(chunks_file, 'r') as f:
            for i, line in enumerate(f):
                if i < 3:  # First 3 chunks
                    try:
                        chunk = json.loads(line)
                        sample_chunks.append({
                            "chunk_id": chunk.get("chunk_id"),
                            "doc_type": chunk.get("metadata", {}).get("doc_type"),
                            "entity_count": sum(len(v) if isinstance(v, list) else 0 
                                              for v in chunk.get("entities", {}).values()),
                            "has_bridge_topics": len(chunk.get("bridge_topics", [])) > 0
                        })
                    except json.JSONDecodeError:
                        continue
        
        analysis["sample_data_analysis"]["sample_chunks"] = sample_chunks
        
        for chunk in sample_chunks:
            logger.info(f"📄 Chunk {chunk['chunk_id']}: {chunk['entity_count']} entities, "
                       f"topics: {chunk['has_bridge_topics']}")
    
    # Check vertical database files
    verticals_dir = output_path / "verticals"
    if verticals_dir.exists():
        for vertical_dir in verticals_dir.iterdir():
            if vertical_dir.is_dir():
                db_file = vertical_dir / f"{vertical_dir.name}_database.json"
                if db_file.exists():
                    try:
                        with open(db_file, 'r') as f:
                            db_data = json.load(f)
                        
                        # Analyze database content
                        if vertical_dir.name == "legal":
                            acts_count = len(db_data.get("acts", {}))
                            rules_count = len(db_data.get("rules", {}))
                            logger.info(f"⚖️  Legal DB: {acts_count} acts, {rules_count} rules")
                            
                        elif vertical_dir.name == "government_orders":
                            total_gos = len(db_data.get("all_gos", {}))
                            active_gos = len(db_data.get("active_gos", {}))
                            chains = len(db_data.get("supersession_chains", {}))
                            logger.info(f"📋 GO DB: {total_gos} total GOs, {active_gos} active, {chains} chains")
                            
                    except Exception as e:
                        logger.warning(f"⚠️  Could not analyze {vertical_dir.name} database: {e}")
    
    # Generate recommendations
    recommendations = []
    
    if analysis.get("ingestion", {}).get("success_rate", 0) < 0.8:
        recommendations.append("Consider improving PDF extraction settings for better text quality")
    
    if analysis.get("ingestion", {}).get("entities_extracted", 0) == 0:
        recommendations.append("No entities extracted - check entity extraction patterns and dictionaries")
    
    if not any(vb.get("success") for vb in analysis["vertical_builders"].values()):
        recommendations.append("All vertical builders failed - check chunk format and data processing")
    
    if not recommendations:
        recommendations.append("Test completed successfully! Ready for full corpus processing.")
    
    analysis["recommendations"] = recommendations
    
    # Save analysis report
    analysis_file = output_path / "test_analysis_report.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"📊 Analysis report saved to: {analysis_file}")
    
    return analysis


def main():
    """Main test execution."""
    logger.info("🚀 Starting Enhanced Pipeline + Vertical Builders Test")
    logger.info(f"📁 Project root: {project_root}")
    
    try:
        # Step 1: Test Enhanced Ingestion Pipeline
        ingestion_results, output_dir = test_enhanced_ingestion()
        
        if not ingestion_results:
            logger.error("❌ Cannot proceed - ingestion failed")
            return 1
        
        # Step 2: Test Vertical Database Builders
        vertical_results = test_vertical_builders(output_dir)
        
        # Step 3: Analyze Results
        analysis = analyze_results(ingestion_results, vertical_results, output_dir)
        
        # Print final summary
        logger.info("="*60)
        logger.info("🎯 TEST SUMMARY")
        logger.info("="*60)
        
        success_count = sum(1 for vb in analysis["vertical_builders"].values() if vb.get("success"))
        total_builders = len(analysis["vertical_builders"])
        
        logger.info(f"✅ Ingestion: {'Success' if analysis['ingestion_success'] else 'Failed'}")
        logger.info(f"🏗️  Vertical Builders: {success_count}/{total_builders} successful")
        
        logger.info("📋 Recommendations:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            logger.info(f"   {i}. {rec}")
        
        logger.info(f"📊 Full report: {output_dir}/test_analysis_report.json")
        
        return 0 if analysis["ingestion_success"] and success_count > 0 else 1
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())