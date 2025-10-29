#!/usr/bin/env python3
"""
Generate All 5 Specialized Databases for AP Policy Assistant.

This script:
1. Processes the full document corpus
2. Generates all 5 vertical databases (Legal, GO, Judicial, Data, Scheme)
3. Creates comprehensive analytics and reports
"""

import sys
import os
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion import EnhancedIngestionPipeline
from src.vertical_builders import (
    LegalDatabaseBuilder, 
    GODatabaseBuilder,
    JudicialDatabaseBuilder,
    DataDatabaseBuilder,
    SchemeDatabaseBuilder
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Generate all specialized databases."""
    start_time = time.time()
    
    print("🚀 AI Policy Assistant - Full Database Generation")
    print("=" * 60)
    
    # Paths
    data_dir = project_root / "data" / "raw" / "Documents"
    output_dir = project_root / "data" / "processed_full"
    
    print(f"📁 Source: {data_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Step 1: Process Full Corpus
    print("\n" + "=" * 60)
    print("STEP 1: PROCESSING FULL DOCUMENT CORPUS")
    print("=" * 60)
    
    pipeline = EnhancedIngestionPipeline(str(output_dir))
    
    # Process different document types from different folders
    folders_to_process = [
        # Critical Priority (Core documents)
        ("Critical Priority/Executive/GO", "Critical Government Orders"),
        ("Critical Priority/Statutory", "Critical Acts and Rules"),
        ("Critical Priority/Judicial/NCPR", "NCPR Guidelines"),
        ("Critical Priority/Policy", "Policy Documents"),
        
        # Dev 1st message (Comprehensive collection)
        ("Dev 1st message/Government Orders - Flagship Schemes", "Flagship Scheme GOs"),
        ("Dev 1st message/Acts and Core Legislation", "Core Acts & Legislation"),
        ("Dev 1st message/Budget and Financial Reports", "Budget & Financial Reports"),
        ("Dev 1st message/NCTE Norms and Teacher Education", "Teacher Education Norms"),
        ("Dev 1st message/Service and Recruitment Rules", "Service & Recruitment"),
        ("Dev 1st message/Research Papers and Case Studies", "Research & Case Studies"),
        ("Dev 1st message/Academic Calendars and Guidelines", "Academic Guidelines"),
        
        # High Priority (Data and assessments)
        ("High Priority/Achievement & Assessment Data", "Assessment Data"),
        ("High Priority/Financial", "Financial Data"),
        ("High Priority/Student & Teacher Data", "Student & Teacher Data"),
        
        # Judiciary (Case law)
        ("Judiciary/Indian Kanoon", "Indian Kanoon Cases"),
        ("Judiciary/Mission 1", "Judicial Mission 1"),
        ("Judiciary/Mission 2", "Judicial Mission 2"),
        
        # Neeraj 2nd message (Statistical data)
        ("Neeraj 2nd message/ASER", "ASER Data"),
        ("Neeraj 2nd message/UDISE+", "UDISE+ Data"),
        ("Neeraj 2nd message/NATIONAL", "National Data"),
        
        # Pranav 3rd Message (National frameworks)
        ("Pranav 3rd Message/National", "National Frameworks"),
        ("Pranav 3rd Message/State/Budget Book/24-25", "State Budget 24-25"),
        ("Pranav 3rd Message/State", "State Documents"),
        
        # Medium Priority (Institutional knowledge)
        ("Medium/Institutional Knowledge", "Institutional Knowledge"),
        ("Medium/Policy Evolution", "Policy Evolution"),
        ("Medium/Research & Studies", "Research Studies"),
    ]
    
    total_processed = 0
    
    for folder_path, folder_desc in folders_to_process:
        full_path = data_dir / folder_path
        if full_path.exists():
            print(f"\n📂 Processing: {folder_desc}")
            print(f"   Path: {full_path}")
            
            try:
                result = pipeline.process_corpus(str(full_path))
                docs_processed = result.get('successful_documents', 0)
                total_processed += docs_processed
                print(f"   ✅ Processed: {docs_processed} documents")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ⚠️  Folder not found: {full_path}")
    
    processing_time = time.time() - start_time
    print(f"\n✅ Corpus processing completed in {processing_time:.1f} seconds")
    print(f"📄 Total documents processed: {total_processed}")
    
    # Step 2: Generate All Vertical Databases
    print("\n" + "=" * 60)
    print("STEP 2: GENERATING SPECIALIZED DATABASES")
    print("=" * 60)
    
    builders = [
        ("Legal", LegalDatabaseBuilder(str(output_dir), str(output_dir / "verticals"))),
        ("Government Orders", GODatabaseBuilder(str(output_dir), str(output_dir / "verticals"))),
        ("Judicial", JudicialDatabaseBuilder(str(output_dir), str(output_dir / "verticals"))),
        ("Data & Metrics", DataDatabaseBuilder(str(output_dir), str(output_dir / "verticals"))),
        ("Schemes", SchemeDatabaseBuilder(str(output_dir), str(output_dir / "verticals")))
    ]
    
    database_results = {}
    
    for db_name, builder in builders:
        print(f"\n🏗️  Building {db_name} Database...")
        
        try:
            start_db_time = time.time()
            database = builder.build_database()
            db_time = time.time() - start_db_time
            
            if database and len(database) > 0:
                # Count entries
                if 'acts' in database:
                    entry_count = len(database['acts']) + len(database.get('rules', {}))
                elif 'government_orders' in database:
                    entry_count = len(database['government_orders'])
                elif 'cases' in database:
                    entry_count = len(database['cases'])
                elif 'metrics_catalog' in database:
                    entry_count = len(database['metrics_catalog'])
                elif 'schemes' in database:
                    entry_count = len(database['schemes'])
                else:
                    entry_count = len(database.get('entries', {}))
                
                # Save database
                builder.save_database(database)
                
                database_results[db_name] = {
                    "success": True,
                    "entries": entry_count,
                    "build_time": f"{db_time:.1f}s",
                    "file": f"verticals/{builder.get_vertical_name()}/{builder.get_vertical_name()}_database.json"
                }
                
                print(f"   ✅ Success: {entry_count} entries created in {db_time:.1f}s")
                
            else:
                database_results[db_name] = {
                    "success": False,
                    "error": "No data generated"
                }
                print(f"   ❌ Failed: No data generated")
                
        except Exception as e:
            database_results[db_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ Failed: {e}")
    
    # Step 3: Generate Summary Report
    print("\n" + "=" * 60)
    print("STEP 3: GENERATING SUMMARY REPORT")
    print("=" * 60)
    
    total_time = time.time() - start_time
    
    # Count total entities across all databases
    total_entries = sum(
        result.get("entries", 0) 
        for result in database_results.values() 
        if result.get("success")
    )
    
    successful_dbs = sum(1 for result in database_results.values() if result.get("success"))
    
    summary_report = {
        "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_processing_time": f"{total_time:.1f} seconds",
        "corpus_processing": {
            "documents_processed": total_processed,
            "folders_processed": len([f for f, _ in folders_to_process if (data_dir / f).exists()])
        },
        "database_generation": {
            "databases_created": successful_dbs,
            "total_databases": len(builders),
            "total_entries": total_entries,
            "results": database_results
        },
        "output_files": {
            "processed_data": str(output_dir),
            "databases": str(output_dir / "verticals"),
            "logs": str(output_dir / "logs") if (output_dir / "logs").exists() else None
        }
    }
    
    # Save summary report
    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    # Display final results
    print(f"\n📊 FINAL RESULTS:")
    print(f"   ⏱️  Total time: {total_time:.1f} seconds")
    print(f"   📄 Documents processed: {total_processed}")
    print(f"   🏗️  Databases created: {successful_dbs}/{len(builders)}")
    print(f"   📝 Total entries: {total_entries}")
    
    print(f"\n📋 DATABASE BREAKDOWN:")
    for db_name, result in database_results.items():
        if result.get("success"):
            print(f"   ✅ {db_name}: {result['entries']} entries ({result['build_time']})")
        else:
            print(f"   ❌ {db_name}: {result.get('error', 'Failed')}")
    
    print(f"\n📁 Output saved to: {output_dir}")
    print(f"📊 Summary report: {summary_file}")
    
    if successful_dbs == len(builders):
        print("\n🎉 SUCCESS: All databases generated successfully!")
        return 0
    else:
        print(f"\n⚠️  WARNING: Only {successful_dbs}/{len(builders)} databases succeeded")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)