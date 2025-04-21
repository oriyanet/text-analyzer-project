import tempfile
import json
from advanced_text_analysis import SentenceGrouping

# ===========================
# Tests for Task 9: SentenceGrouping
# ===========================


def create_temp_processed_sentences(data):
    """ Helper function to simulate processed sentences. """
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='')
    temp_file.write("\n".join(" ".join(sentence) for sentence in data))
    temp_file.close()
    return temp_file.name


def test_empty_sentences():
    """ Test behavior with no sentences """
    processed_sentences = []
    threshold = 1
    grouper = SentenceGrouping(processed_sentences, threshold)
    grouped_sentences = grouper.group_sentences()
    assert grouped_sentences == []


def test_single_sentence():
    """ Test a single sentence in the input """
    processed_sentences = [["the", "only", "sentence"]]
    threshold = 1
    grouper = SentenceGrouping(processed_sentences, threshold)
    grouped_sentences = grouper.group_sentences()
    expected_groups = [["Group 1", [["the", "only", "sentence"]]]]
    assert grouped_sentences == expected_groups


def test_no_shared_words_above_threshold():
    """ Test when no sentences share enough words to form a group """
    processed_sentences = [["red", "balloon"], ["green", "parrot"], ["blue", "ocean"]]
    threshold = 2  # Sentences must share at least 2 words to be grouped together.

    grouper = SentenceGrouping(processed_sentences, threshold)
    grouped_sentences = grouper.group_sentences()
    expected_groups = [
        ["Group 1", [["blue", "ocean"]]],
        ["Group 2", [["green", "parrot"]]],
        ["Group 3", [["red", "balloon"]]]
    ]
    assert grouped_sentences == expected_groups


def test_generate_output(capsys):
    """ Test output format and correctness """
    processed_sentences = [["the", "quick", "brown", "fox"], ["the", "quick", "red", "fox"],
                           ["the", "blue", "fox"], ["the", "lazy", "dog"]]
    threshold = 2

    grouper = SentenceGrouping(processed_sentences, threshold)
    grouped_sentences = grouper.group_sentences()
    grouper.generate_output(grouped_sentences)  # Ensure output is printed in the expected JSON format.

    captured = capsys.readouterr()
    output_json = json.loads(captured.out.strip())  # Convert printed output to JSON for validationץ

    assert "Question 9" in output_json
    assert "group Matches" in output_json["Question 9"]
    assert len(output_json["Question 9"]["group Matches"]) == len(grouped_sentences)  # Verify group count consistency.


def test_create_temp_processed_sentences():
    """ Test that create_temp_processed_sentences correctly writes sentences to a temporary file. """
    sample_data = [["the", "quick", "brown", "fox"], ["jumps", "over", "the", "lazy", "dog"]]
    temp_file_path = create_temp_processed_sentences(sample_data)
    try:
        # Read the file and check contents
        with open(temp_file_path, "r") as f:
            content = f.read().splitlines()
        expected_content = ["the quick brown fox", "jumps over the lazy dog"]
        assert content == expected_content
    finally:
        # Ensure the file is deleted after the test
        tempfile.NamedTemporaryFile(delete=True).close()
