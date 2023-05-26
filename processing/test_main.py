import json
import pytest
from subprocess import check_output
import time
from pathlib import Path


@pytest.fixture
def run_main():
    def run_main_with_args(main_path, img_path, output_file):
        command = ["python3", main_path, img_path, output_file]
        result = check_output(command)
        return result.decode("utf-8")

    return run_main_with_args


def calculate_correct_chars(expected_output, actual_output):
    correct_chars = 0
    total_chars = 0

    for key, expected_value in expected_output.items():
        actual_value = actual_output.get(key, "")
        for expected_char, actual_char in zip(expected_value, actual_value):
            if expected_char == actual_char:
                correct_chars += 1
            total_chars += 1

    percentage_correct = (correct_chars / total_chars) * 100
    return percentage_correct


def test_main(run_main):
    file_path = Path(__file__).resolve().parent
    img_path = file_path.parent / "test_data"
    output_file = file_path.parent / "test_output.txt"
    solution_file = file_path / "test_solution.txt"
    main_path = file_path.parent / "Pawlowski_Adam.py"

    with open(solution_file, "r") as f:
        expected_output = json.load(f)

    start_time = time.time()  # Start time measurement

    # Run main with the arguments
    _ = run_main(main_path, img_path, output_file)

    end_time = time.time()  # End time measurement
    elapsed_time = end_time - start_time

    with open(output_file, "r") as f:
        actual_output = json.load(f)

    # Calculate percentage of correct characters in the values
    percentage_correct = calculate_correct_chars(expected_output, actual_output)

    print("\nElapsed Time:", elapsed_time, "seconds")
    print("Percentage of Correct Characters:", percentage_correct)

    assert percentage_correct >= 80
