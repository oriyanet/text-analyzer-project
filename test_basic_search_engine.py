import pytest
import tempfile
import json
from basic_search_engine import BasicSearchEngine, PersonContextsWithKSeqs, GraphAnalyzer, IndirectConnections
from text_tasks import TextProcessor

# ===========================
# Tests for Task 4: BasicSearchEngine
# ===========================

# Note: Some parameters in the tests are intentionally set to None.
# This is because the program does not require certain inputs for specific tasks.
# For example, the "Names" file is not needed, so None is passed instead.


@pytest.fixture
def create_temp_file():
    def _create_temp_file(content, add_header=False, header_name=""):
        temp = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='')
        if add_header:
            temp.write(f"{header_name}\n")
        temp.write(content)
        temp.close()
        return temp.name
    return _create_temp_file


def test_build_index(create_temp_file):
    """Tests if the index is correctly built for given sentences and K-seq list."""
    sentences_content = "Harry caught the Snitch!\nThe Snitch flew away\nHermione saw the Snitch."
    remove_words_content = "the\n"
    kseq_content = json.dumps({"keys": [["snitch"], ["harry caught"]]})
    # Create temporary files for input data.
    sentences_path = create_temp_file(sentences_content, add_header=True, header_name="sentence")
    remove_words_path = create_temp_file(remove_words_content, add_header=True, header_name="words")
    kseq_path = create_temp_file(kseq_content)
    # Initialize processor and search engine.
    processor = TextProcessor(sentences_path, remove_words_path, None)  # "None" since names file is not needed.
    processor.load_data()

    engine = BasicSearchEngine(sentences_path, remove_words_path, kseq_path)
    engine.processor = processor
    engine.load_data()
    index = engine.build_index()

    assert "snitch" in index
    assert "harry caught" in index
    # Validate that the indexed sentences match the expected output
    assert sorted(index["snitch"], key=lambda s: " ".join(s)) == sorted([
        ["harry", "caught", "snitch"],
        ["snitch", "flew", "away"],
        ["hermione", "saw", "snitch"]
    ], key=lambda s: " ".join(s))

    assert index["harry caught"] == [["harry", "caught", "snitch"]]


def test_generate_output(create_temp_file, capsys):
    """Tests if the search engine correctly generates formatted output."""
    sentences_content = "Harry caught the Snitch!\nThe Snitch flew away\nHermione saw the Snitch."
    remove_words_content = "the\n"
    kseq_content = json.dumps({"keys": [["snitch"], ["harry caught"]]})
    # Create temporary files for input
    sentences_path = create_temp_file(sentences_content, add_header=True, header_name="sentence")
    remove_words_path = create_temp_file(remove_words_content, add_header=True, header_name="words")
    kseq_path = create_temp_file(kseq_content)
    # Initialize processor and engine
    processor = TextProcessor(sentences_path, remove_words_path, None)  # "None" since names file is not needed.
    processor.load_data()
    # Capture the printed output.
    engine = BasicSearchEngine(sentences_path, remove_words_path, kseq_path)
    engine.processor = processor
    engine.load_data()
    engine.generate_output(question_number=4)

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    assert "Question 4" in output
    assert "K-Seq Matches" in output["Question 4"]
    # Compare actual and expected matches.
    actual_matches = sorted(
        [[kseq, sorted(sentences, key=lambda s: " ".join(s))]
         for kseq, sentences in output["Question 4"]["K-Seq Matches"]],
        key=lambda x: x[0]
    )

    expected_matches = sorted([
        ["harry caught", [["harry", "caught", "snitch"]]],
        ["snitch", sorted([
            ["harry", "caught", "snitch"],
            ["snitch", "flew", "away"],
            ["hermione", "saw", "snitch"]
        ], key=lambda s: " ".join(s))]
    ], key=lambda x: x[0])

    assert actual_matches == expected_matches


def test_load_data_with_preprocessed(create_temp_file):
    """Test loading preprocessed data instead of processing raw input."""
    preprocessed_content = json.dumps({
        "Question 1": {
            "Processed Sentences": [["harry", "caught", "snitch"], ["snitch", "flew", "away"]]
        }
    })
    kseq_content = json.dumps({"keys": [["snitch"]]})
    preprocessed_path = create_temp_file(preprocessed_content)
    kseq_path = create_temp_file(kseq_content)

    engine = BasicSearchEngine(None, None, kseq_path, preprocessed_path)
    engine.load_data()

    assert engine.preprocessed_data["Question 1"]["Processed Sentences"] == [
        ["harry", "caught", "snitch"], ["snitch", "flew", "away"]]


def test_load_data_missing_paths():
    """Checks if 'invalid input' triggers SystemExit when file paths are missing."""
    engine = BasicSearchEngine(None, None, "some_kseq_path.json")
    with pytest.raises(SystemExit):
        engine.load_data()


def test_load_data_invalid_kseq_format(create_temp_file):
    """Tests if 'invalid input' triggers SystemExit for an invalid or missing K-seq file."""
    invalid_kseq_content = json.dumps({"invalid_key": ["snitch"]})
    invalid_kseq_path = create_temp_file(invalid_kseq_content)
    sentences_content = "Harry caught the Snitch!\nThe Snitch flew away\nHermione saw the Snitch."
    remove_words_content = "the\n"

    sentences_path = create_temp_file(sentences_content, add_header=True, header_name="sentence")
    remove_words_path = create_temp_file(remove_words_content, add_header=True, header_name="words")

    # Initialize the search engine with the invalid K-seq file
    engine = BasicSearchEngine(sentences_path, remove_words_path, invalid_kseq_path)

    # Expect the program to exit due to invalid input
    with pytest.raises(SystemExit):
        engine.load_data()


def test_build_index_with_preprocessed(create_temp_file):
    """Tests if the index is correctly built when using preprocessed data."""
    preprocessed_content = json.dumps({
        "Question 1": {
            "Processed Sentences": [["harry", "caught", "snitch"], ["snitch", "flew", "away"]]
        }
    })
    kseq_content = json.dumps({"keys": [["snitch"]]})

    preprocessed_path = create_temp_file(preprocessed_content)
    kseq_path = create_temp_file(kseq_content)

    engine = BasicSearchEngine(None, None, kseq_path, preprocessed_path)
    engine.load_data()
    index = engine.build_index()

    assert "snitch" in index
    # Check if the sentences are correctly indexed and sorted.
    assert sorted(index["snitch"]) == sorted([["harry", "caught", "snitch"], ["snitch", "flew", "away"]])


def test_load_data_invalid_json_kseq(create_temp_file):
    """Tests if 'invalid input' triggers SystemExit for an invalid JSON K-seq file."""
    invalid_kseq_path = create_temp_file("{invalid json}")
    engine = BasicSearchEngine("sentences.csv", "remove_words.csv", invalid_kseq_path)
    with pytest.raises(SystemExit):
        engine.load_data()


# ===========================
# Tests for Task 5: PersonContextsWithKSeqs
# ===========================


def test_load_data_with_preprocessed_q5(create_temp_file):
    """ Test loading data from a preprocessed JSON file. """
    preprocessed_content = json.dumps({
        "Question 1": {
            "Processed Sentences": [["harry", "caught", "snitch"], ["snitch", "flew", "away"]],
            "Processed Names": [[["harry"], [["harry potter"]]]]
        }
    })
    preprocessed_path = create_temp_file(preprocessed_content)

    engine = PersonContextsWithKSeqs(None, None,
                                     None, max_k=3, preprocessed_path=preprocessed_path)
    engine.load_data()

    assert engine.sentences == [["harry", "caught", "snitch"], ["snitch", "flew", "away"]]
    assert engine.names == [[["harry"], [["harry potter"]]]]


def test_load_data_missing_paths_q5():
    """Test if missing input paths cause the program to exit."""
    engine = PersonContextsWithKSeqs(None, None, None, max_k=3)

    with pytest.raises(SystemExit):
        engine.load_data()


def test_load_data_invalid_json(create_temp_file):
    """Test if an invalid JSON file causes the program to exit."""
    invalid_json_path = create_temp_file("{invalid json}")  # Invalid JSON content

    engine = PersonContextsWithKSeqs("sentences.csv", "remove_words.csv",
                                     "people.csv", max_k=3, preprocessed_path=invalid_json_path)

    with pytest.raises(SystemExit):
        engine.load_data()


def test_generate_k_seqs_standard():
    """Test generating K-sequences from a standard sentence."""
    engine = PersonContextsWithKSeqs(None, None, None, max_k=3)
    sentence = ["harry", "caught", "snitch"]

    k_seqs = engine._generate_k_seqs(sentence)

    expected_k_seqs = [
        ["harry"], ["caught"], ["snitch"],  # 1-seq
        ["harry", "caught"], ["caught", "snitch"],  # 2-seq
        ["harry", "caught", "snitch"]  # 3-seq
    ]

    assert sorted(k_seqs) == sorted(expected_k_seqs)


def test_get_name_contexts(create_temp_file):
    """ Test extracting name contexts and generating K-seqs for each person. """
    sentences_content = "Harry caught the Snitch!\nHermione saw Harry in the library."
    people_content = "harry,harry potter\nhermione,hermione granger"
    remove_words_content = "the\n"
    # Create temporary files for input data.
    sentences_path = create_temp_file(sentences_content, add_header=True, header_name="sentence")
    people_path = create_temp_file(people_content, add_header=True, header_name="Name,Other Names")
    remove_words_path = create_temp_file(remove_words_content, add_header=True, header_name="words")

    # Initialize the engine and load data.
    engine = PersonContextsWithKSeqs(sentences_path, remove_words_path, people_path, max_k=3)
    engine.load_data()
    name_contexts = engine._get_name_contexts()

    expected_output = [
        ["harry", [
            ["caught"], ["caught", "snitch"], ["harry"],
            ["harry", "caught"], ["harry", "caught", "snitch"],
            ["harry", "in"], ["harry", "in", "library"],
            ["hermione"], ["hermione", "saw"], ["hermione", "saw", "harry"],
            ["in"], ["in", "library"], ["library"], ["saw"],
            ["saw", "harry"], ["saw", "harry", "in"], ["snitch"]
        ]],
        ["hermione", [
            ["harry"], ["harry", "in"], ["harry", "in", "library"],
            ["hermione"], ["hermione", "saw"], ["hermione", "saw", "harry"],
            ["in"], ["in", "library"], ["library"], ["saw"],
            ["saw", "harry"], ["saw", "harry", "in"]
        ]]
    ]

    assert name_contexts == expected_output


def test_generate_k_seqs_edge_cases():
    """TTest _generate_k_seqs behavior, specifically for cases where len(sentence) < k."""
    engine = PersonContextsWithKSeqs(None, None, None, max_k=3)

    # Case with a very short sentence.
    short_sentence = ["word"]
    result = engine._generate_k_seqs(short_sentence)
    expected_output = [["word"]]  # max_k = 3 but only one word exists

    assert result == expected_output, f"Expected {expected_output}, but got {result}"

    # Case with a slightly longer sentence.
    longer_sentence = ["a", "b"]
    result = engine._generate_k_seqs(longer_sentence)
    expected_output = [["a"], ["b"], ["a", "b"]]  # Max possible is 2-seq

    assert result == expected_output, f"Expected {expected_output}, but got {result}"


def test_generate_output_q5(capsys):
    """Test that generate_output correctly formats and prints the expected JSON structure."""
    engine = PersonContextsWithKSeqs(None, None, None, max_k=3)

    # Assume that the result of _get_name_contexts returns something like this:
    engine._get_name_contexts = lambda: [["harry", [["caught"], ["snitch"]]], ["hermione", [["saw"]]]]

    engine.generate_output(question_number=5)

    captured = capsys.readouterr()
    output = json.loads(captured.out)

    expected_output = {
        "Question 5": {
            "Person Contexts and K-Seqs": [["harry", [["caught"], ["snitch"]]], ["hermione", [["saw"]]]]
        }
    }

    assert output == expected_output, f"Expected {expected_output}, but got {output}"

# ===========================
# Tests for Task 6: GraphAnalyzer
# ===========================


@pytest.fixture
def create_temp_file_q6():
    """Creates a temporary file with given content and returns its path."""

    def _create_temp_file(content, add_header=False, header_name=""):
        temp = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='')
        if add_header:
            temp.write(f"{header_name}\n")
        temp.write(content)
        temp.close()
        return temp.name

    return _create_temp_file


def test_graph_analyzer_basic(create_temp_file):
    """Tests if GraphAnalyzer correctly finds direct connections in a small dataset."""
    sentences_content = "Harry met Ron.\nRon saw Hermione.\nHermione talked to Harry."
    people_content = "Name,Other Names\nHarry Potter,Harry\nRon Weasley,Ron\nHermione Granger,Hermione"
    remove_words_content = "the\n"
    # Create temporary files with sentences, people, and words to remove.
    sentences_path = create_temp_file(sentences_content, add_header=True, header_name="sentence")
    people_path = create_temp_file(people_content, add_header=True, header_name="Name,Other Names")
    remove_words_path = create_temp_file(remove_words_content, add_header=True, header_name="words")

    analyzer = GraphAnalyzer(sentences_path, people_path, remove_words_path, window_size=2, threshold=1)
    analyzer.load_and_process_data()

    # Generate overlapping sentence windows and count connections between people.
    windows = analyzer.generate_windows()
    connection_counts = analyzer.count_connections(windows)
    filtered_connections = analyzer.filter_connections(connection_counts)

    assert sorted(filtered_connections) == sorted([
        [["harry", "potter"], ["ron", "weasley"]],
        [["hermione", "granger"], ["ron", "weasley"]],
        [["harry", "potter"], ["hermione", "granger"]]
    ])


def test_graph_analyzer_with_preprocessed(create_temp_file, capsys):
    """Tests if GraphAnalyzer correctly processes preprocessed input data."""

    # Create a temporary preprocessed JSON file with sentences and names.
    preprocessed_content = json.dumps({
        "Question 1": {
            "Processed Sentences": [
                ["harry", "met", "ron"],
                ["ron", "saw", "hermione"],
                ["hermione", "talked", "to", "harry"]
            ],
            "Processed Names": [
                [["harry", "potter"], [["harry"]]],
                [["ron", "weasley"], [["ron"]]],
                [["hermione", "granger"], [["hermione"]]]
            ]
        }
    })
    preprocessed_path = create_temp_file(preprocessed_content)

    # Initialize GraphAnalyzer without file paths (using preprocessed data instead).
    analyzer = GraphAnalyzer(
        sentences_path=None,
        names_path=None,
        remove_words_path=None,
        window_size=2,
        threshold=1
    )
    # Load preprocessed data manually from the temporary file.
    with open(preprocessed_path, "r") as f:
        preprocessed_data = json.load(f)

    analyzer.processed_sentences = preprocessed_data["Question 1"]["Processed Sentences"]
    analyzer.processed_names = preprocessed_data["Question 1"]["Processed Names"]

    # Process the data and check if the extracted connections match expectations.
    windows = analyzer.generate_windows()
    connection_counts = analyzer.count_connections(windows)
    filtered_connections = analyzer.filter_connections(connection_counts)

    assert sorted(filtered_connections) == sorted([
        [["harry", "potter"], ["ron", "weasley"]],
        [["hermione", "granger"], ["ron", "weasley"]],
        [["harry", "potter"], ["hermione", "granger"]]
    ])


def test_graph_analyzer_empty_input(create_temp_file):
    """Tests if 'invalid input' triggers SystemExit when input files are empty."""
    empty_sentences_path = create_temp_file("", add_header=True, header_name="sentence")
    empty_people_path = create_temp_file("", add_header=True, header_name="Name,Other Names")
    empty_remove_words_path = create_temp_file("", add_header=True, header_name="words")

    analyzer = GraphAnalyzer(empty_sentences_path, empty_people_path, empty_remove_words_path, window_size=2,
                             threshold=1)

    with pytest.raises(SystemExit):
        analyzer.load_and_process_data()


def test_graph_analyzer_no_connections(create_temp_file):
    """Tests if GraphAnalyzer correctly returns an empty list when no connections meet the threshold."""
    sentences_content = "Harry met Dumbledore.\nSnape saw Draco."
    people_content = "Name,Other Names\nHarry Potter,Harry\nDumbledore,Albus\nSnape,Severus\nDraco Malfoy,Draco"
    remove_words_content = "the\n"

    sentences_path = create_temp_file(sentences_content, add_header=True, header_name="sentence")
    people_path = create_temp_file(people_content, add_header=True, header_name="Name,Other Names")
    remove_words_path = create_temp_file(remove_words_content, add_header=True, header_name="words")

    analyzer = GraphAnalyzer(sentences_path, people_path, remove_words_path, window_size=2, threshold=2)
    analyzer.load_and_process_data()
    windows = analyzer.generate_windows()
    connection_counts = analyzer.count_connections(windows)
    filtered_connections = analyzer.filter_connections(connection_counts)

    assert filtered_connections == []


def test_graph_analyzer_invalid_json(create_temp_file, capsys):
    """Tests if GraphAnalyzer correctly handles an invalid JSON format in a preprocessed file."""
    invalid_json_path = create_temp_file("{invalid json}")  # Invalid JSON format

    with pytest.raises(json.JSONDecodeError):
        with open(invalid_json_path, "r") as f:
            json.load(f)


def test_generate_output_q6(capsys):
    """Tests if generate_output correctly prints the expected JSON format."""
    connections = [
        [["harry", "potter"], ["ron", "weasley"]],
        [["hermione", "granger"], ["ron", "weasley"]]
    ]
    expected_output = json.dumps({
        "Question 6": {
            "Pair Matches": connections
        }
    }, indent=4)
    GraphAnalyzer.generate_output(connections)  # Call the function to capture its output

    captured = capsys.readouterr()
    assert captured.out.strip() == expected_output

# ===========================
# Tests for Task 7: IndirectConnections
# ===========================


@pytest.fixture
def create_temp_file_q7():
    """Creates a temporary file with given content."""
    def _create_temp_file(content):
        temp = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='')
        temp.write(content)
        temp.close()
        return temp.name
    return _create_temp_file


def test_convert_to_graph():
    """Tests if convert_to_graph correctly builds a connection graph."""
    connections = [
        [["harry", "potter"], ["hermione", "granger"]],
        [["hermione", "granger"], ["ron", "weasley"]],
        [["draco", "malfoy"], ["harry", "potter"]]
    ]
    graph = IndirectConnections.convert_to_graph(connections)

    assert graph["harry potter"] == ["hermione granger", "draco malfoy"]
    assert graph["hermione granger"] == ["harry potter", "ron weasley"]
    assert graph["ron weasley"] == ["hermione granger"]
    assert graph["draco malfoy"] == ["harry potter"]


def test_is_connected():
    """Tests if is_connected correctly identifies direct and indirect connections."""
    graph = {
        "harry potter": ["hermione granger", "draco malfoy"],
        "hermione granger": ["harry potter", "ron weasley"],
        "ron weasley": ["hermione granger"],
        "draco malfoy": ["harry potter"]
    }

    analyzer = IndirectConnections(graph, None, maximal_distance=2)

    assert analyzer.is_connected("harry potter", "hermione granger") is True
    assert analyzer.is_connected("harry potter", "ron weasley") is True
    assert analyzer.is_connected("harry potter", "draco malfoy") is True
    assert analyzer.is_connected("ron weasley", "draco malfoy") is False


def test_is_connected_with_distance():
    """Tests if maximal distance correctly limits connections."""
    graph = {
        "harry potter": ["hermione granger"],
        "hermione granger": ["ron weasley"],
        "ron weasley": ["draco malfoy"]
    }
    analyzer = IndirectConnections(graph, None, maximal_distance=2)

    assert analyzer.is_connected("harry potter", "ron weasley") is True  # Distance 2
    assert analyzer.is_connected("harry potter", "draco malfoy") is False  # Distance 3, exceeds limit


def test_load_data(create_temp_file):
    """Tests if load_data correctly reads connection pairs from a JSON file."""
    connections_content = json.dumps({
        "keys": [
            ["harry potter", "hermione granger"],
            ["ron weasley", "draco malfoy"]
        ]
    })
    connections_path = create_temp_file(connections_content)

    analyzer = IndirectConnections({}, connections_path, maximal_distance=3)
    analyzer.load_data()

    assert analyzer.person_pairs == [
        ["harry potter", "hermione granger"],
        ["ron weasley", "draco malfoy"]
    ]   # Ensure the loaded data matches the expected connection pairs.


def test_invalid_connections_file(create_temp_file):
    """Tests if an invalid JSON file raises an error."""
    invalid_content = "{invalid json}"
    invalid_path = create_temp_file(invalid_content)

    analyzer = IndirectConnections({}, invalid_path, maximal_distance=3)

    with pytest.raises(json.JSONDecodeError):
        analyzer.load_data()


def test_check_connections(create_temp_file):
    """Tests if check_connections correctly determines connected pairs."""
    graph = {
        "harry potter": ["hermione granger"],
        "hermione granger": ["harry potter", "ron weasley"],
        "ron weasley": ["hermione granger"]
    }
    connections_content = json.dumps({
        "keys": [
            ["harry potter", "ron weasley"],  # Indirectly connected
            ["harry potter", "draco malfoy"]  # Not connected
        ]
    })
    connections_path = create_temp_file(connections_content)

    analyzer = IndirectConnections(graph, connections_path, maximal_distance=2)
    analyzer.load_data()
    results = analyzer.check_connections()

    expected_results = {
        "Question 7": {
            "Pair Matches": [
                ["harry potter", "ron weasley", True],
                ["harry potter", "draco malfoy", False]
            ]
        }
    }

    # Sort names within each pair to ensure order does not affect comparison.
    def sort_pair(pair):
        return [sorted(pair[:2]), pair[2]]

    sorted_actual = sorted([sort_pair(pair) for pair in results["Question 7"]["Pair Matches"]])
    sorted_expected = sorted([sort_pair(pair) for pair in expected_results["Question 7"]["Pair Matches"]])

    # Ensure the function correctly identifies direct and indirect connections.
    assert sorted_actual == sorted_expected


def test_load_data_invalid_format(create_temp_file):
    """Tests if 'invalid input' triggers SystemExit for an invalid connections file format."""
    invalid_connections_content = json.dumps({
        "keys": [
            ["harry potter"],  # Invalid – only one name instead of two.
            ["hermione granger", "ron weasley"],
            "not a list at all"  # Invalid – string instead of a list.
        ]
    })
    connections_path = create_temp_file(invalid_connections_content)
    analyzer = IndirectConnections(graph={}, connections_path=connections_path, maximal_distance=2)
    with pytest.raises(SystemExit):
        analyzer.load_data()


def test_check_connections_invalid_pair_format():
    """Tests if 'invalid input' triggers SystemExit for an invalid pair format."""
    analyzer = IndirectConnections(graph={}, connections_path=None, maximal_distance=2)
    analyzer.person_pairs = [
        ["harry potter"],  # Invalid – only one name instead of two.
        ["hermione granger", "ron weasley"],
        ["draco malfoy", "luna lovegood", "extra name"]  # Invalid – three names instead of two.
    ]

    with pytest.raises(SystemExit):
        analyzer.check_connections()


# ===========================
# Tests for Task 8: IndirectConnections
# ===========================


@pytest.fixture
def create_temp_file_q8():
    """Creates a temporary file with given content."""
    def _create_temp_file(content):
        temp = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8', newline='')
        temp.write(content)
        temp.close()
        return temp.name
    return _create_temp_file


def test_is_connected_fixed_length():
    """Tests if is_connected correctly identifies paths of exactly fixed length."""
    graph = {
        "harry potter": ["hermione granger"],
        "hermione granger": ["ron weasley"],
        "ron weasley": ["draco malfoy"]
    }
    analyzer = IndirectConnections(graph, None, maximal_distance=5, fixed_length=2, is_task_8=True)

    assert analyzer.is_connected("harry potter", "ron weasley") is True  # Distance exactly 2
    assert analyzer.is_connected("harry potter", "draco malfoy") is False  # Distance 3, exceeds fixed_length
    assert analyzer.is_connected("harry potter", "hermione granger") is False  # Distance 1, not exactly 2


def test_load_data_q8(create_temp_file_q8):
    """Tests if load_data correctly reads connection pairs from a JSON file for Task 8."""
    connections_content = json.dumps({
        "keys": [
            ["harry potter", "hermione granger"],
            ["ron weasley", "draco malfoy"]
        ]
    })
    connections_path = create_temp_file_q8(connections_content)
    analyzer = IndirectConnections({}, connections_path, maximal_distance=5, fixed_length=2, is_task_8=True)
    analyzer.load_data()

    assert analyzer.person_pairs == [
        ["harry potter", "hermione granger"],
        ["ron weasley", "draco malfoy"]
    ]  # Ensure the loaded data matches expected pairs.


def test_check_connections_fixed_length(create_temp_file_q8):
    """Tests if check_connections correctly determines paths of exactly fixed length."""
    graph = {
        "harry potter": ["hermione granger"],
        "hermione granger": ["ron weasley"],
        "ron weasley": ["draco malfoy"]
    }
    connections_content = json.dumps({
        "keys": [
            ["harry potter", "ron weasley"],  # Path exists with exact length 2
            ["harry potter", "draco malfoy"]  # Path exists but length is 3
        ]
    })
    connections_path = create_temp_file_q8(connections_content)

    analyzer = IndirectConnections(graph, connections_path, maximal_distance=5, fixed_length=2, is_task_8=True)
    analyzer.load_data()
    results = analyzer.check_connections()

    expected_results = {
        "Question 8": {
            "Pair Matches": [
                ["harry potter", "ron weasley", True],
                ["harry potter", "draco malfoy", False]
            ]
        }
    }
    assert sorted([sorted(pair[:2]) + [pair[2]] for pair in results["Question 8"]["Pair Matches"]]) == \
           sorted([sorted(pair[:2]) + [pair[2]] for pair in expected_results["Question 8"]["Pair Matches"]])


def test_check_connections_invalid_fixed_length(create_temp_file_q8):
    """Tests if check_connections correctly handles an invalid fixed length case."""
    graph = {
        "harry potter": ["hermione granger"],
        "hermione granger": ["ron weasley"]
    }
    connections_content = json.dumps({
        "keys": [
            ["harry potter", "ron weasley"],  # Path exists but length is not exactly 3
        ]
    })
    connections_path = create_temp_file_q8(connections_content)
    analyzer = IndirectConnections(graph, connections_path, maximal_distance=5, fixed_length=3, is_task_8=True)
    analyzer.load_data()
    results = analyzer.check_connections()
    expected_results = {
        "Question 8": {
            "Pair Matches": [
                ["harry potter", "ron weasley", False]  # Should return False because the exact length isn't met
            ]
        }
    }
    assert results == expected_results
