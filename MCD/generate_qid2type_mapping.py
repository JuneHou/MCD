#!/usr/bin/env python3
"""
VQA-CP v2 Question ID to Question Type Mapping Generator

This script generates the qid2type mapping file required by the MCD (Margin-based 
Counterfactual Debiasing) model for VQA-CP v2 dataset evaluation.

The qid2type mapping is used during model evaluation to categorize questions by their
types for detailed performance analysis across different question categories.

Author: Generated for MCD VQA-CP v2 setup
Date: July 15, 2025

Dependencies:
    - json (standard library)
    - VQA-CP v2 annotation files

Input Files:
    - vqacp_v2_train_annotations.json: Training set annotations
    - vqacp_v2_test_annotations.json: Test set annotations

Output File:
    - qid2type_cpv2.json: Mapping from question IDs to question types

Usage:
    python generate_qid2type_mapping.py

File Structure Expected:
    /data/wang/junh/datasets/vqa-cp-v2/
    ├── vqacp_v2_train_annotations.json
    ├── vqacp_v2_test_annotations.json
    └── ...

Output Location:
    /data/wang/junh/githubs/MCD_new/MCD/util/qid2type_cpv2.json
"""

import json
import os
from collections import Counter
from typing import Dict, List, Any


class QID2TypeGenerator:
    """
    Generator class for creating question ID to question type mappings from VQA-CP v2 annotations.
    
    This class processes VQA-CP v2 annotation files to create a comprehensive mapping
    that enables the MCD model to evaluate performance across different question types.
    """
    
    def __init__(self, 
                 vqa_cp_data_path: str = "/data/wang/junh/datasets/vqa-cp-v2/",
                 output_path: str = "/data/wang/junh/githubs/MCD_new/MCD/util/qid2type_cpv2.json"):
        """
        Initialize the QID2Type generator.
        
        Args:
            vqa_cp_data_path (str): Path to the VQA-CP v2 dataset directory
            output_path (str): Path where the qid2type JSON file will be saved
        """
        self.vqa_cp_data_path = vqa_cp_data_path
        self.output_path = output_path
        
        # Input file paths
        self.train_annotations_path = os.path.join(vqa_cp_data_path, "vqacp_v2_train_annotations.json")
        self.test_annotations_path = os.path.join(vqa_cp_data_path, "vqacp_v2_test_annotations.json")
        
        # Statistics tracking
        self.question_type_stats = Counter()
        self.total_questions = 0
        
    def load_annotations(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load VQA-CP v2 annotations from a JSON file.
        
        Args:
            file_path (str): Path to the annotation file
            
        Returns:
            List[Dict]: List of annotation dictionaries
            
        Raises:
            FileNotFoundError: If the annotation file doesn't exist
            json.JSONDecodeError: If the JSON file is malformed
        """
        print(f"Loading annotations from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
            
        try:
            with open(file_path, 'r') as f:
                annotations = json.load(f)
            print(f"Loaded {len(annotations)} annotations")
            return annotations
            
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error parsing JSON file {file_path}: {e}")
    
    def process_annotations(self, annotations: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Process annotations to extract question ID to question type mappings.
        
        Each annotation contains:
        - question_id: Unique identifier for the question
        - question_type: Category/type of the question (e.g., "what", "how many", "is the")
        - answer_type: Type of answer expected (yes/no, number, other)
        - Other metadata...
        
        Args:
            annotations (List[Dict]): List of annotation dictionaries
            
        Returns:
            Dict[str, str]: Mapping from question ID (as string) to question type
        """
        qid_to_type = {}
        
        print(f"Processing {len(annotations)} annotations...")
        
        for annotation in annotations:
            # Extract question ID and type
            question_id = annotation['question_id']
            question_type = annotation['question_type']
            
            # Convert question ID to string for JSON compatibility
            qid_str = str(question_id)
            
            # Store the mapping
            qid_to_type[qid_str] = question_type
            
            # Update statistics
            self.question_type_stats[question_type] += 1
            self.total_questions += 1
            
        return qid_to_type
    
    def print_statistics(self):
        """Print statistics about the question types found in the dataset."""
        print("\n" + "="*60)
        print("QUESTION TYPE STATISTICS")
        print("="*60)
        print(f"Total questions processed: {self.total_questions:,}")
        print(f"Unique question types: {len(self.question_type_stats)}")
        print("\nTop 15 most common question types:")
        print("-" * 40)
        
        for question_type, count in self.question_type_stats.most_common(15):
            percentage = (count / self.total_questions) * 100
            print(f"{question_type:<25} {count:>8,} ({percentage:>5.1f}%)")
        
        print("-" * 40)
        print(f"{'TOTAL':<25} {self.total_questions:>8,} (100.0%)")
    
    def save_mapping(self, qid2type: Dict[str, str]):
        """
        Save the question ID to type mapping as a JSON file.
        
        Args:
            qid2type (Dict[str, str]): The complete mapping to save
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving qid2type mapping to: {self.output_path}")
        
        with open(self.output_path, 'w') as f:
            json.dump(qid2type, f, indent=2)
        
        print(f"Successfully saved {len(qid2type):,} question ID mappings")
    
    def verify_mapping(self, qid2type: Dict[str, str]) -> bool:
        """
        Verify the completeness and correctness of the generated mapping.
        
        Args:
            qid2type (Dict[str, str]): The mapping to verify
            
        Returns:
            bool: True if verification passes, False otherwise
        """
        print("\n" + "="*60)
        print("VERIFICATION")
        print("="*60)
        
        # Reload original annotations to verify
        train_annotations = self.load_annotations(self.train_annotations_path)
        test_annotations = self.load_annotations(self.test_annotations_path)
        
        # Collect all original question IDs
        train_qids = {str(ann['question_id']) for ann in train_annotations}
        test_qids = {str(ann['question_id']) for ann in test_annotations}
        all_original_qids = train_qids | test_qids
        
        # Get mapped question IDs
        mapped_qids = set(qid2type.keys())
        
        # Check for missing or extra IDs
        missing_qids = all_original_qids - mapped_qids
        extra_qids = mapped_qids - all_original_qids
        
        print(f"Original train questions: {len(train_qids):,}")
        print(f"Original test questions: {len(test_qids):,}")
        print(f"Total unique questions: {len(all_original_qids):,}")
        print(f"Mapped questions: {len(mapped_qids):,}")
        print(f"Missing questions: {len(missing_qids)}")
        print(f"Extra questions: {len(extra_qids)}")
        
        if len(missing_qids) == 0 and len(extra_qids) == 0:
            print("✅ VERIFICATION PASSED: All question IDs correctly mapped!")
            return True
        else:
            print("❌ VERIFICATION FAILED:")
            if missing_qids:
                print(f"   Missing QIDs (first 5): {list(missing_qids)[:5]}")
            if extra_qids:
                print(f"   Extra QIDs (first 5): {list(extra_qids)[:5]}")
            return False
    
    def generate(self) -> Dict[str, str]:
        """
        Main method to generate the complete qid2type mapping.
        
        Returns:
            Dict[str, str]: Complete mapping from question IDs to question types
        """
        print("VQA-CP v2 QID2Type Mapping Generator")
        print("="*60)
        
        # Load annotations from both training and test sets
        train_annotations = self.load_annotations(self.train_annotations_path)
        test_annotations = self.load_annotations(self.test_annotations_path)
        
        # Process annotations to create mappings
        print("\nProcessing training annotations...")
        train_mapping = self.process_annotations(train_annotations)
        
        print("Processing test annotations...")
        test_mapping = self.process_annotations(test_annotations)
        
        # Combine mappings
        complete_mapping = {**train_mapping, **test_mapping}
        
        # Print statistics
        self.print_statistics()
        
        # Save the mapping
        self.save_mapping(complete_mapping)
        
        # Verify the mapping
        verification_passed = self.verify_mapping(complete_mapping)
        
        if verification_passed:
            print(f"\n🎉 Successfully generated qid2type mapping!")
            print(f"   File location: {self.output_path}")
            print(f"   Total mappings: {len(complete_mapping):,}")
        else:
            print(f"\n⚠️  Generated mapping with issues - please review!")
        
        return complete_mapping


def main():
    """
    Main function to run the QID2Type mapping generation.
    
    This function demonstrates how to use the QID2TypeGenerator class
    to create the required mapping file for MCD model evaluation.
    """
    # Initialize the generator
    generator = QID2TypeGenerator()
    
    try:
        # Generate the mapping
        qid2type_mapping = generator.generate()
        
        # Show some example mappings
        print("\nExample mappings (first 5):")
        print("-" * 30)
        for i, (qid, qtype) in enumerate(list(qid2type_mapping.items())[:5]):
            print(f"{qid}: {qtype}")
        
        print("\nMapping generation completed successfully!")
        
    except Exception as e:
        print(f"Error during mapping generation: {e}")
        raise


if __name__ == "__main__":
    main()
