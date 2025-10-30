#!/usr/bin/env python3
"""
Script to reorganize processed data into vertical-specific folders matching
the raw Documents_Organized structure.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

class ProcessedDataOrganizer:
    def __init__(self, source_dir: str, target_base_dir: str):
        self.source_dir = Path(source_dir)
        self.target_base_dir = Path(target_base_dir)
        
        # Define verticals (matching raw organization)
        self.verticals = {
            'legal': 'Legal (Statutory Framework)',
            'gos': 'Government Orders (GOs)',
            'judicial': 'Judicial (Case Law)',
            'data_reports': 'Data Reports (Metrics & Statistics)',
            'schemes': 'Schemes (Implementation Tracking)'
        }
        
        # Classification patterns for processed files
        self.classification_patterns = {
            'legal': [
                'act', 'rule', 'statute', 'legislation', 'regulation', 'amendment',
                'constitution', 'rte act', 'education act', 'ap education act',
                'rte rules', 'service rules', 'recruitment rules', 'transfer rules',
                'statutory', 'legal framework', 'section', 'subsection',
                'ncte', 'norms', 'statutory_', 'acts_and_core_legislation',
                'service_and_recruitment', 'dsc', 'trt', 'transfer',
            ],
            'gos': [
                'go', 'government order', 'g.o.', 'g.o.ms', 'g-o-ms',
                'order', 'circular', 'notification', 'proceedings',
                'executive', 'government_orders', 'flagship', 'go_ms',
                'nadu-nedu', 'nadunedu', 'amma vodi', 'ammavodi',
            ],
            'judicial': [
                'judgment', 'judgement', 'court', 'judicial', 'case', 'legal case',
                'supreme court', 'high court', 'petition', 'writ', 'appeal',
                'ruling', 'verdict', 'precedent', 'case law', 'ncpr',
                'guidelines', 'scpcr',
            ],
            'data_reports': [
                'udise', 'udise+', 'enrollment', 'statistics', 'statistical',
                'data', 'report', 'metrics', 'survey', 'assessment',
                'nas', 'ses', 'aser', 'ptr', 'dropout', 'retention',
                'budget', 'financial', 'outcome budget', 'budget analysis',
                'achievement', 'assessment data', 'student data', 'teacher data',
                'glance report', 'progress report', 'audit report',
                'budget_and_financial', 'achievement_&_assessment',
                'volume_', 'backward_classes', 'minorites', 'child_budget',
            ],
            'schemes': [
                'scheme', 'programme', 'program', 'nadu-nedu', 'nadunedu',
                'amma vodi', 'ammavodi', 'jagananna', 'vidya deevena',
                'gorumudda', 'implementation', 'flagship scheme',
                'takeover policy', 'private aided schools',
            ]
        }
        
        # Processed data subdirectories
        self.processed_subdirs = [
            'text_extraction',
            'chunks',
            'metadata',
            'entities',
            'relations'
        ]
        
        self.stats = defaultdict(lambda: {'count': 0, 'files': []})
        self.unclassified = []
        
    def _classify_file(self, file_path: Path) -> Tuple[str, str]:
        """Classify a processed file into one of the verticals."""
        file_name_lower = file_path.name.lower()
        
        # Score each vertical
        scores = {}
        
        for vertical_key, patterns in self.classification_patterns.items():
            score = 0
            
            # Check filename
            for pattern in patterns:
                if pattern in file_name_lower:
                    score += 2
            
            scores[vertical_key] = score
        
        # Get the highest scoring vertical
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                best_vertical = max(scores, key=scores.get)
                return best_vertical, 'high' if max_score >= 2 else 'medium'
        
        return None, 'low'
    
    def _get_subcategory(self, file_path: Path, vertical_key: str) -> str:
        """Determine subcategory within a vertical (simplified for processed data)."""
        file_name_lower = file_path.name.lower()
        
        if vertical_key == 'legal':
            if 'rte' in file_name_lower:
                return 'RTE_Act_and_Rules'
            elif 'education act' in file_name_lower or '1982' in file_name_lower:
                return 'AP_Education_Act_and_Amendments'
            elif 'recruitment' in file_name_lower or 'dsc' in file_name_lower or 'trt' in file_name_lower:
                return 'Service_and_Recruitment_Rules'
            elif 'transfer' in file_name_lower:
                return 'Transfer_Rules'
            elif 'constitution' in file_name_lower:
                return 'Constitution'
            elif 'ncte' in file_name_lower:
                return 'NCTE_Norms'
            elif 'regulatory' in file_name_lower or 'commission' in file_name_lower:
                return 'Regulatory_Framework'
            else:
                return 'Other_Acts_and_Rules'
        
        elif vertical_key == 'gos':
            if 'nadu-nedu' in file_name_lower or 'nadunedu' in file_name_lower:
                return 'Nadu_Nedu'
            elif 'amma vodi' in file_name_lower or 'ammavodi' in file_name_lower:
                return 'Amma_Vodi'
            elif 'vidya deevena' in file_name_lower:
                return 'Vidya_Deevena'
            elif 'transfer' in file_name_lower:
                return 'Teacher_Transfer_Orders'
            elif 'circular' in file_name_lower:
                return 'Circulars'
            elif 'mdm' in file_name_lower or 'mid-day meal' in file_name_lower:
                return 'Mid_Day_Meal'
            else:
                return 'Other_GOs'
        
        elif vertical_key == 'judicial':
            if 'supreme court' in file_name_lower or 'sc' in file_name_lower:
                return 'Supreme_Court_Judgments'
            elif 'high court' in file_name_lower or 'hc' in file_name_lower:
                return 'High_Court_Judgments'
            elif 'ap' in file_name_lower and 'judgement' in file_name_lower:
                return 'AP_Judgments'
            elif 'case' in file_name_lower:
                return 'Case_Law'
            elif 'ncpcr' in file_name_lower or 'ncpr' in file_name_lower:
                return 'NCPCR_Guidelines'
            else:
                return 'Other_Judicial_Documents'
        
        elif vertical_key == 'data_reports':
            if 'udise' in file_name_lower:
                return 'UDISE_Reports'
            elif 'budget' in file_name_lower:
                return 'Budget_Reports'
            elif 'nas' in file_name_lower or 'national achievement survey' in file_name_lower:
                return 'NAS_Reports'
            elif 'ses' in file_name_lower or 'school education survey' in file_name_lower:
                return 'SES_Reports'
            elif 'aser' in file_name_lower:
                return 'ASER_Reports'
            elif 'financial' in file_name_lower or 'pab' in file_name_lower:
                return 'Financial_Reports'
            elif 'achievement' in file_name_lower:
                return 'Achievement_Reports'
            elif 'teacher' in file_name_lower or 'transfer' in file_name_lower:
                return 'Teacher_Data'
            elif 'student' in file_name_lower:
                return 'Student_Data'
            elif 'scert' in file_name_lower:
                return 'SCERT_Reports'
            else:
                return 'Other_Data_Reports'
        
        elif vertical_key == 'schemes':
            if 'nadu-nedu' in file_name_lower or 'nadunedu' in file_name_lower:
                return 'Nadu_Nedu_Scheme'
            elif 'amma vodi' in file_name_lower or 'ammavodi' in file_name_lower:
                return 'Amma_Vodi_Scheme'
            elif 'vidya deevena' in file_name_lower:
                return 'Vidya_Deevena_Scheme'
            elif 'gorumudda' in file_name_lower:
                return 'Gorumudda_Scheme'
            elif 'takeover' in file_name_lower:
                return 'School_Takeover_Policy'
            else:
                return 'Other_Schemes'
        
        return 'General'
    
    def organize_processed_data(self, copy_mode: bool = True):
        """Organize processed data files into vertical structure."""
        print(f"\n{'='*80}")
        print(f"Organizing processed data from: {self.source_dir}")
        print(f"Target directory: {self.target_base_dir}")
        print(f"Mode: {'COPY' if copy_mode else 'MOVE'}")
        print(f"{'='*80}\n")
        
        # Create vertical structure for each processed subdirectory
        for subdir in self.processed_subdirs:
            source_subdir = self.source_dir / subdir
            if not source_subdir.exists():
                print(f"Skipping {subdir} - directory not found")
                continue
            
            print(f"\nProcessing {subdir}...")
            
            # Find all files in this subdirectory
            all_files = []
            for ext in ['.json', '.jsonl', '.txt', '.csv']:
                all_files.extend(source_subdir.rglob(f'*{ext}'))
            
            print(f"Found {len(all_files)} files in {subdir}")
            
            for file_path in all_files:
                # Skip if already in target directory
                if self.target_base_dir in file_path.parents:
                    continue
                
                # Classify file
                vertical_key, confidence = self._classify_file(file_path)
                
                if vertical_key:
                    # Get subcategory
                    subcategory = self._get_subcategory(file_path, vertical_key)
                    
                    # Get target directory
                    vertical_name = self.verticals[vertical_key]
                    target_dir = self.target_base_dir / vertical_name / subcategory / subdir
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Determine target file path
                    target_file = target_dir / file_path.name
                    
                    # Handle name conflicts
                    counter = 1
                    while target_file.exists():
                        stem = file_path.stem
                        suffix = file_path.suffix
                        target_file = target_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    # Copy or move file
                    try:
                        if copy_mode:
                            shutil.copy2(file_path, target_file)
                            action = "Copied"
                        else:
                            shutil.move(str(file_path), str(target_file))
                            action = "Moved"
                        
                        # Update statistics
                        self.stats[vertical_key]['count'] += 1
                        self.stats[vertical_key]['files'].append({
                            'source': str(file_path),
                            'target': str(target_file),
                            'subcategory': subcategory,
                            'subdir': subdir,
                            'confidence': confidence
                        })
                        
                        print(f"  [{confidence.upper()}] {action}: {file_path.name}")
                        print(f"    → {vertical_name} / {subcategory} / {subdir}")
                        
                    except Exception as e:
                        print(f"  ERROR: Failed to {action.lower()} {file_path}: {e}")
                        self.unclassified.append({
                            'file': str(file_path),
                            'error': str(e)
                        })
                else:
                    print(f"  [UNCLASSIFIED] {file_path.name}")
                    self.unclassified.append({
                        'file': str(file_path),
                        'reason': 'No matching pattern found'
                    })
        
        # Handle root-level files (corpus_statistics.json, document_index.json, etc.)
        print(f"\nProcessing root-level files...")
        root_files = []
        for ext in ['.json', '.jsonl']:
            root_files.extend(list(self.source_dir.glob(f'*{ext}')))
        
        for file_path in root_files:
            target_file = self.target_base_dir / file_path.name
            try:
                if copy_mode:
                    shutil.copy2(file_path, target_file)
                else:
                    shutil.move(str(file_path), str(target_file))
                print(f"  Copied root file: {file_path.name}")
            except Exception as e:
                print(f"  ERROR: Failed to copy {file_path.name}: {e}")
        
        # Print summary
        self._print_summary()
        
        # Save classification report
        self._save_report()
    
    def _print_summary(self):
        """Print organization summary"""
        print(f"\n{'='*80}")
        print("PROCESSED DATA ORGANIZATION SUMMARY")
        print(f"{'='*80}\n")
        
        total_classified = 0
        for vertical_key, vertical_name in self.verticals.items():
            count = self.stats[vertical_key]['count']
            total_classified += count
            print(f"{vertical_name:50s}: {count:4d} files")
        
        print(f"\n{'Total Classified':50s}: {total_classified:4d} files")
        print(f"{'Unclassified':50s}: {len(self.unclassified):4d} files")
        print(f"{'='*80}\n")
        
        if self.unclassified:
            print("UNCLASSIFIED FILES:")
            for item in self.unclassified[:20]:
                print(f"  - {item['file']}")
            if len(self.unclassified) > 20:
                print(f"  ... and {len(self.unclassified) - 20} more")
    
    def _save_report(self):
        """Save detailed classification report"""
        report_path = self.target_base_dir / 'processed_classification_report.json'
        
        report = {
            'summary': {
                vertical_key: {
                    'count': self.stats[vertical_key]['count'],
                    'name': vertical_name
                }
                for vertical_key, vertical_name in self.verticals.items()
            },
            'details': {
                vertical_key: self.stats[vertical_key]['files']
                for vertical_key in self.verticals.keys()
            },
            'unclassified': self.unclassified
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Classification report saved to: {report_path}")


def main():
    # Configuration
    source_dir = "/Users/nitin/Documents/AI policy Assistant/data/processed"
    target_dir = "/Users/nitin/Documents/AI policy Assistant/data/processed_organized"
    
    # Create organizer
    organizer = ProcessedDataOrganizer(source_dir, target_dir)
    
    # Ask user for confirmation
    print("\n" + "="*80)
    print("PROCESSED DATA ORGANIZATION TOOL")
    print("="*80)
    print(f"\nSource: {source_dir}")
    print(f"Target: {target_dir}")
    print("\nThis will organize processed files into 5 verticals:")
    for key, name in organizer.verticals.items():
        print(f"  - {name}")
    
    response = input("\nProceed with COPY mode? (yes/no): ").strip().lower()
    if response in ['yes', 'y']:
        organizer.organize_processed_data(copy_mode=True)
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()

