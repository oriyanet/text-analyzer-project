import pytest
import tempfile
import pandas as pd
import json
from text_tasks import TextProcessor, KSeqAnalyzer, PersonMentionCounter

# ===========================
# Tests for Task 1: TextProcessor
# ===========================


def create_temp_csv(data, header=True):
    """ Helper function to create temporary CSV file """
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='')
    df = pd.DataFrame(data)
    df.to_csv(temp_file.name, index=False, header=header)
    return temp_file.name


def test_text_processor_basic():
    """ Test basic functionality of TextProcessor """
    sentences_data = {"sentence": ["Harry rode on a silver chariot.", "It shined! like a star of platinum.",
                                   "He was known as the red magician."]}
    people_data = {"Name": ["Harry Potter", "Ron Weasley"], "Other Names": ["Boy Who Lived,Dear boy", "Weasel King"]}
    remove_words_data = {"words": ["on", "a", "he", "as", "of", "it", "was", "the"]}

    sentences_path = create_temp_csv(sentences_data)
    people_path = create_temp_csv(people_data)
    remove_words_path = create_temp_csv(remove_words_data)

    processor = TextProcessor(sentences_path, remove_words_path, people_path)
    processor.load_data()
    processed_sentences = processor.process_sentences()
    processed_names = processor.process_names()

    expected_sentences = [["harry", "rode", "silver", "chariot"], ["shined", "like", "star", "platinum"],
                          ["known", "red", "magician"]]
    expected_names = [[["harry", "potter"], [["boy", "who", "lived"], ["dear", "boy"]]],
                      [["ron", "weasley"], [["weasel", "king"]]]]

    assert processed_sentences == expected_sentences
    assert processed_names == expected_names


def test_empty_files():
    """Test behavior with empty input files"""
    empty_sentences_path = create_temp_csv({"sentence": []})
    empty_people_path = create_temp_csv({"Name": [], "Other Names": []})
    empty_remove_words_path = create_temp_csv({"words": []})

    processor = TextProcessor(empty_sentences_path, empty_remove_words_path, empty_people_path)

    with pytest.raises(SystemExit) as e:
        processor.load_data()
    assert str(e.value) == "invalid input"


def test_duplicate_names():
    """ Test handling of duplicate names in input """
    valid_sentences_path = create_temp_csv({"sentence": ["John met JD at the park."]})
    valid_remove_words_path = create_temp_csv({"words": [""]})  # Ensuring column exists

    duplicate_people_data = {"Name": ["John Doe", "John Doe"], "Other Names": ["JD,Johnny", "JD,Johnny"]}
    duplicate_people_path = create_temp_csv(duplicate_people_data)

    processor = TextProcessor(valid_sentences_path, valid_remove_words_path, duplicate_people_path)
    processor.load_data()
    processed_names = processor.process_names()

    assert len(processed_names) == 1  # Should remove duplicate


def test_invalid_input():
    """Test invalid input handling"""
    processor = TextProcessor("non_existent.csv", "non_existent.csv", "non_existent.csv")

    with pytest.raises(SystemExit) as e:
        processor.load_data()
    assert str(e.value) == "invalid input"


def test_generate_output(capsys):
    """ Test generate_output function """
    question_number = 1
    processed_sentences = [["harry", "rode", "silver", "chariot"]]
    processed_names = [["harry", "potter"], [["boy", "who", "lived"], ["dear", "boy"]]]

    TextProcessor.generate_output(question_number, processed_sentences, processed_names)

    captured = capsys.readouterr()
    output_json = json.loads(captured.out.strip())

    assert f"Question {question_number}" in output_json
    assert "Processed Sentences" in output_json[f"Question {question_number}"]
    assert "Processed Names" in output_json[f"Question {question_number}"]
    assert output_json[f"Question {question_number}"]["Processed Sentences"] == processed_sentences
    assert output_json[f"Question {question_number}"]["Processed Names"] == processed_names


def test_missing_words_column():
    """Test missing 'words' column in remove words file"""
    valid_sentences_path = create_temp_csv({"sentence": ["John met JD at the park."]})
    missing_words_column_path = create_temp_csv({"wrong_column": ["the", "a", "an"]})
    valid_names_path = create_temp_csv({"Name": ["John Doe"], "Other Names": ["JD,Johnny"]})

    processor = TextProcessor(valid_sentences_path, missing_words_column_path, valid_names_path)

    with pytest.raises(SystemExit) as e:
        processor.load_data()
    assert str(e.value) == "invalid input"


def test_missing_names_columns():
    """Test missing 'Name' and 'Other Names' columns in names file"""
    valid_sentences_path = create_temp_csv({"sentence": ["John met JD at the park."]})
    valid_remove_words_path = create_temp_csv({"words": ["the", "a", "an"]})
    missing_names_columns_path = create_temp_csv({"wrong_column": ["John Doe"]})

    processor = TextProcessor(valid_sentences_path, valid_remove_words_path, missing_names_columns_path)

    with pytest.raises(SystemExit) as e:
        processor.load_data()
    assert str(e.value) == "invalid input"


# ===========================
# Tests for Task 2: KSeqAnalyzer
# ===========================


def create_temp_json(data):
    """ Helper function to create temporary JSON file """
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, newline='')
    with open(temp_file.name, 'w') as f:
        json.dump(data, f)
    return temp_file.name


def test_kseq_basic():
    """ Test basic k-seq generation """
    sentences = [["harry", "caught", "snitch"], ["snitch", "flew", "away"]]
    analyzer = KSeqAnalyzer(sentences, max_k=2)
    output = analyzer.analyze_sequences()

    expected_output = [
        ["1_seq", [["away", 1], ["caught", 1], ["flew", 1], ["harry", 1], ["snitch", 2]]],
        ["2_seq", [["caught snitch", 1], ["flew away", 1], ["harry caught", 1], ["snitch flew", 1]]]
    ]

    for seq in output:
        seq[1].sort(key=lambda x: x[0])

    assert output == expected_output


def test_kseq_maxk_1():
    """ Test k-seq generation with max_k=1 (only word frequency) """
    sentences = [["word", "word", "test"], ["test", "word"]]
    analyzer = KSeqAnalyzer(sentences, max_k=1)
    output = analyzer.analyze_sequences()

    assert output == [["1_seq", [["test", 2], ["word", 3]]]]


def test_kseq_large_k():
    """ Test k-seq generation when max_k is larger than any sentence """
    sentences = [["short", "sentence"], ["another", "example"]]
    analyzer = KSeqAnalyzer(sentences, max_k=5)
    output = analyzer.analyze_sequences()

    # If no valid k-seq can be formed because max_k is too large, the output should be empty for all k's.
    expected_output = [["1_seq", [["another", 1], ["example", 1], ["sentence", 1], ["short", 1]]],
                       ["2_seq", [["another example", 1], ["short sentence", 1]]]]

    # If the length of output is 0, check that it matches the expected empty output.
    if len(sentences[0]) < 5:  # max_k > sentence length
        assert output == []  # Ensure empty output as expected
    else:
        assert output == expected_output


def test_kseq_empty_sentences():
    """ Test behavior when given empty sentences """
    sentences = []
    analyzer = KSeqAnalyzer(sentences, max_k=3)
    output = analyzer.analyze_sequences()
    assert output == []


def test_kseq_generate_output(capsys):
    """ Test output generation in JSON format for KSeqAnalyzer """
    sentences = [["harry", "caught", "snitch"], ["snitch", "flew", "away"]]
    analyzer = KSeqAnalyzer(sentences, max_k=2)
    # Generate the output
    analyzer.generate_output(2)
    # Capture the output printed to stdout
    captured = capsys.readouterr()
    # Convert the captured output to JSON for validation
    output_json = json.loads(captured.out.strip())
    # Sort the results lexicographically by the sequence
    for seq_type in output_json["Question 2"]["2-Seq Counts"]:
        seq_type[1].sort(key=lambda x: x[0])

    expected_output = {
        "Question 2": {
            "2-Seq Counts": [
                ["1_seq", [["away", 1], ["caught", 1], ["flew", 1], ["harry", 1], ["snitch", 2]]],
                ["2_seq", [["caught snitch", 1], ["flew away", 1], ["harry caught", 1], ["snitch flew", 1]]]
            ]
        }
    }

    # Compare the sorted actual output to the expected output
    assert output_json == expected_output


def test_generate_k_seq_counts_too_short():
    """Test case where sentences are too short to generate a k-seq."""
    sentences = [["short"]]
    analyzer = KSeqAnalyzer(sentences, max_k=2)
    result = analyzer._generate_k_seq_counts(k=2)
    # Expecting an empty dictionary since no k-seq can be formed
    assert result == {}

# ===========================
# Tests for Task 3: PersonMentionCounter
# ===========================


def test_person_mentions_basic():
    """Test basic counting of names and nicknames"""
    sentences = [["harry", "caught", "snitch"], ["snitch", "flew", "away"], ["hermione", "snitch"]]
    names = [
        [["harry", "potter"], []],  # No nicknames
        [["hermione", "granger"], []]  # No nicknames
    ]

    counter = PersonMentionCounter(sentences, names)
    result = counter.count_mentions()

    expected = {
        "harry potter": 1,  # Only "harry" appears
        "hermione granger": 1  # "hermione" appears once
    }

    assert result == expected


def test_person_mentions_with_nicknames():
    """Test counting when only nicknames appear in the text"""
    sentences = [["boy", "who", "lived", "is", "here"], ["dear", "boy", "saw", "snitch"]]
    names = [
        [["harry", "potter"], [["boy", "who", "lived"], ["dear", "boy"]]]
    ]

    counter = PersonMentionCounter(sentences, names)
    result = counter.count_mentions()

    expected = {
        "harry potter": 2  # "boy who lived" + "dear boy"
    }

    assert result == expected


def test_person_mentions_partial_names():
    """Test counting partial occurrences of a name"""
    sentences = [["harry", "potter", "is", "here"], ["potter", "won", "the", "match"]]
    names = [
        [["harry", "potter"], [["the", "chosen", "one"]]]
    ]

    counter = PersonMentionCounter(sentences, names)
    result = counter.count_mentions()

    expected = {
        "harry potter": 3  # "harry" (1) + "potter" (2) (full name is not counted separately)
    }

    assert result == expected


def test_person_mentions_no_appearance():
    """Test when a name exists in the list but doesn't appear in the text"""
    sentences = [["ron", "weasley", "is", "brave"]]
    names = [
        [["harry", "potter"], [["boy", "who", "lived"], ["dear", "boy"]]]
    ]

    counter = PersonMentionCounter(sentences, names)
    result = counter.count_mentions()

    assert result == {}  # "harry potter" and nicknames don't appear


def test_person_mentions_duplicate_count():
    """Test when a nickname could be a substring of another name"""
    sentences = [["potter", "is", "here"], ["harry", "potter", "is", "famous"]]
    names = [
        [["harry", "potter"], [["potter"]]],
        [["james", "potter"], [["prongs"]]]
    ]

    counter = PersonMentionCounter(sentences, names)
    result = counter.count_mentions()

    expected = {
        "harry potter": 5,  # "harry" (1) + "potter" (3) + "potter" as a nickname (1)
        "james potter": 2   # "potter" from "james potter" appears twice
    }

    assert result == expected


def test_person_mentions_empty_sentences():
    """Test behavior when no sentences are provided"""
    sentences = []
    names = [
        [["harry", "potter"], [["boy", "who", "lived"], ["dear", "boy"]]]
    ]

    counter = PersonMentionCounter(sentences, names)
    result = counter.count_mentions()

    assert result == {}  # No names should be counted


def test_generate_output_q3(capsys):
    """Test JSON output generation"""
    sentences = [["harry", "caught", "snitch"], ["hermione", "granger"]]
    names = [
        [["harry", "potter"], [["boy", "who", "lived"], ["dear", "boy"]]],
        [["hermione", "granger"], []]
    ]

    counter = PersonMentionCounter(sentences, names)
    counter.generate_output(3)

    captured = capsys.readouterr()
    output_json = json.loads(captured.out.strip())

    expected_output = {
        "Question 3": {
            "Name Mentions": [
                ["harry potter", 1],  # "harry" appears once in the text
                ["hermione granger", 2]  # "hermione" + "granger" (full name is not counted separately)
            ]
        }
    }

    assert output_json == expected_output


def test_person_mentions_nickname_not_continuous():
    """Test that a nickname is counted only if it appears in full and continuously"""
    sentences = [["chosen", "was", "a", "wizard"], ["chosen", "wizard", "is", "here"]]
    names = [
        [["harry", "potter"], [["chosen", "wizard"]]]
    ]
    counter = PersonMentionCounter(sentences, names)
    result = counter.count_mentions()
    expected = {
        "harry potter": 1  # Only the second sentence should be counted (since "chosen wizard" appears fully)
    }
    assert result == expected


def test_person_mentions_repeated_name():
    """Test that each part of the main name is counted separately"""
    sentences = [["harry", "potter", "harry"]]
    names = [
        [["harry", "potter"], []]  # No nicknames
    ]
    counter = PersonMentionCounter(sentences, names)
    result = counter.count_mentions()
    expected = {
        "harry potter": 3  # "harry" (2 times) + "potter" (1 time)
    }
    assert result == expected


def test_person_mentions_multiple_parts():
    """Test that each part of a full name is counted separately"""
    sentences = [["james", "went", "to", "school"], ["he", "was", "called", "brando", "by", "his", "friend"]]
    names = [
        [["james", "potter", "brando"], []]  # No nicknames
    ]
    counter = PersonMentionCounter(sentences, names)
    result = counter.count_mentions()
    expected = {
        "james potter brando": 2  # "james" (1) + "brando" (1)
    }
    assert result == expected
