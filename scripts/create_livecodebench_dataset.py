from datasets import load_dataset
import json
import random
import os
from collections import defaultdict
import pickle
import zlib
import base64

def create_public_test_file(input_files, output_file="data/livecodebench_public_test.jsonl"):
    """
    Create livecodebench_public_test.jsonl with public test cases.
    Format similar to mbpp_public_test.jsonl: {"entry_point": "func", "test": ["assert ..."]}
    
    For LiveCodeBench, we convert the structured input/output to executable assert statements.
    """
    public_tests_map = {}  # question_id -> test cases
    
    for input_file in input_files:
        print(f"\nProcessing {input_file} for public tests...")
        with open(input_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                question_id = item['question_id']
                
                # Get metadata to extract func_name
                metadata = json.loads(item.get('metadata', '{}'))
                func_name = metadata.get('func_name', 'wrapped_function')
                
                # Parse public test cases
                try:
                    public_tests = json.loads(item['public_test_cases'])
                except:
                    public_tests = json.loads(
                        pickle.loads(
                            zlib.decompress(
                                base64.b64decode(item['public_test_cases'].encode('utf-8'))
                            )
                        )
                    )
                
                # Store raw input/output pairs directly
                # The Test operator will use these directly without parsing
                test_cases = []
                for i, test_case in enumerate(public_tests):
                    test_cases.append({
                        "input": test_case["input"],
                        "output": test_case["output"]
                    })
                
                if test_cases:
                    public_tests_map[question_id] = {
                        "entry_point": func_name,
                        "question_id": question_id,
                        "test": test_cases
                    }
    
    # Write to output file
    print(f"\nWriting {len(public_tests_map)} entries to {output_file}...")
    with open(output_file, 'w') as f:
        for question_id in sorted(public_tests_map.keys()):
            f.write(json.dumps(public_tests_map[question_id]) + '\n')
    
    print(f"Created {output_file} with {len(public_tests_map)} entries")
    
    # Show a sample
    if public_tests_map:
        sample_id = list(public_tests_map.keys())[0]
        print(f"\nSample entry (question_id={sample_id}):")
        print(json.dumps(public_tests_map[sample_id], indent=2))

if __name__ == "__main__":
    # Load the LiveCodeBench code generation dataset
    print("Loading LiveCodeBench dataset...")
    lcb_codegen = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5")
    
    # Get test split data
    test_split = lcb_codegen["test"]
    print(f"Total examples: {len(test_split)}")
    
    # Group by difficulty to maintain ratio
    difficulty_groups = defaultdict(list)
    for idx, item in enumerate(test_split):
        difficulty = item.get('difficulty', 'unknown')
        difficulty_groups[difficulty].append(item)
    
    # Print difficulty distribution
    print("\nDifficulty distribution:")
    for diff, items in sorted(difficulty_groups.items()):
        print(f"  {diff}: {len(items)}")
    
    # Split each difficulty group into validation (1/5) and test (4/5)
    # This maintains the difficulty ratio in both splits
    random.seed(42)
    val_data = []
    test_data = []
    
    for difficulty, items in difficulty_groups.items():
        # if difficulty != "easy":
        #     continue
        # Shuffle items within difficulty group
        shuffled = items.copy()
        random.shuffle(shuffled)
        
        # Split: 1/5 for validation, 4/5 for test
        val_size = len(shuffled) // 5
        val_data.extend(shuffled[:val_size])
        test_data.extend(shuffled[val_size:])
    
    # Shuffle the final splits
    random.shuffle(val_data)
    random.shuffle(test_data)

    # val_data = val_data[10:11]
    # test_data = test_data[10:11]
    
    print(f"\nValidation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")
    
    # Verify difficulty ratio is maintained
    print("\nValidation difficulty distribution:")
    val_diff_counts = defaultdict(int)
    for item in val_data:
        val_diff_counts[item.get('difficulty', 'unknown')] += 1
    for diff, count in sorted(val_diff_counts.items()):
        print(f"  {diff}: {count}")
    
    print("\nTest difficulty distribution:")
    test_diff_counts = defaultdict(int)
    for item in test_data:
        test_diff_counts[item.get('difficulty', 'unknown')] += 1
    for diff, count in sorted(test_diff_counts.items()):
        print(f"  {diff}: {count}")
    
    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Write directly to jsonl files
    # Keep the original HF dataset format - the benchmark's load_data will process it
    validate_file = "data/livecodebench_validate.jsonl"
    test_file = "data/livecodebench_test.jsonl"
    
    with open(validate_file, "w") as f:
        for item in val_data:
            # Convert dataset item to dict and write as JSON line
            f.write(json.dumps(dict(item)) + "\n")
    
    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(dict(item)) + "\n")
    
    print("\nDatasets created successfully!")
    print(f"Saved to {validate_file} and {test_file}")
    
    # Create public test file from the generated datasets
    print("\n" + "="*80)
    print("Creating public test file...")
    print("="*80)
    create_public_test_file([validate_file, test_file])
    
    print("\n" + "="*80)
    print("All files created successfully!")
    print("="*80)