import json
import re
import sys
from text_tasks import TextProcessor
from typing import List, Dict, Union, Tuple, Optional, Set


class BasicSearchEngine:
    """Task 4: Implements a basic search engine for finding and indexing K-sequences in sentences."""

    def __init__(self, sentences_path: str, remove_words_path: str,
                 kseq_path: str, preprocessed_path: Optional[str] = None) -> None:
        """Initializes the BasicSearchEngine with paths to input files."""
        self.sentences_path = sentences_path
        self.remove_words_path = remove_words_path
        self.kseq_path = kseq_path
        self.preprocessed_path = preprocessed_path
        self.processor = None
        self.kseqs = []
        self.preprocessed_data = None

    def load_data(self):
        """Loads and processes input data from files."""
        try:
            # If a preprocessed file is provided, load it instead of processing raw data.
            if self.preprocessed_path:
                with open(self.preprocessed_path, 'r') as f:
                    self.preprocessed_data = json.load(f)
            else:
                # Validate that required file paths are provided
                if not self.sentences_path or not self.remove_words_path:
                    sys.exit("invalid input")  # Invalid input case

                # Names file is not needed for this task, so None is intentionally passed
                self.processor = TextProcessor(self.sentences_path, self.remove_words_path, None)
                self.processor.load_data()
            if not self.kseq_path:
                sys.exit("invalid input")  # Invalid input case

            with open(self.kseq_path, 'r') as f:
                raw_kseqs = json.load(f)
                # Check if the JSON file contains a valid "keys" list.
                if isinstance(raw_kseqs, dict) and "keys" in raw_kseqs:
                    self.kseqs = {}
                    for seq in raw_kseqs["keys"]:
                        if seq and any(word.strip() for word in seq):
                            # Normalize words (remove punctuation, convert to lowercase)
                            kseq = " ".join(
                                re.sub(r"[^\w\s]", "", word.strip().lower()) for word in seq if word.strip()).strip()
                            if kseq not in self.kseqs:
                                self.kseqs[kseq] = []
                else:
                    sys.exit("invalid input")  # Invalid JSON format case

        except (FileNotFoundError, json.JSONDecodeError, ValueError, PermissionError):
            sys.exit("invalid input")

    def build_index(self) -> Dict[str, List[List[str]]]:
        """Builds an index for efficient K-seq searching."""
        index = {}
        # Use preprocessed sentences if available; otherwise, process sentences from scratch.
        if self.preprocessed_data:
            cleaned_sentences = self.preprocessed_data["Question 1"]["Processed Sentences"]
        else:
            cleaned_sentences = self.processor.process_sentences()
        for sentence in cleaned_sentences:
            sentence_str = " ".join(sentence)  # Convert sentence list to a string for easy substring search.
            for kseq in self.kseqs:
                if kseq in sentence_str:
                    if kseq not in index:
                        index[kseq] = []
                    index[kseq].append(sentence)
                    # Check each word in the main name as a separate occurrence

        # Sort sentences alphabetically for each K-seq key, as required by the task instructions.
        for kseq in index:
            index[kseq] = sorted(index[kseq], key=lambda s: " ".join(s))

        return index

    def generate_output(self, question_number: int) -> None:
        """Generate and print the output in JSON format for the given task."""
        index = self.build_index()
        sorted_kseqs = sorted(index.items(), key=lambda x: x[0])
        output = {
            f"Question {question_number}": {
                "K-Seq Matches": [
                    [kseq, sentences] for kseq, sentences in sorted_kseqs
                ]
            }
        }
        print(json.dumps(output, indent=4))


class PersonContextsWithKSeqs:
    """Task 5: Analyzes sentence contexts with names and generates K-sequences."""

    def __init__(self, sentences_path: str, remove_words_path: str, names_path: str,
                 max_k: int, preprocessed_path: Optional[str] = None) -> None:
        """Initializes the class with paths and parameters."""
        self.sentences_path = sentences_path
        self.remove_words_path = remove_words_path
        self.names_path = names_path
        self.max_k = max_k
        self.preprocessed_path = preprocessed_path
        self.processor = None
        self.preprocessed_data = None
        self.sentences = []
        self.names = []

    def load_data(self):
        """Loads and processes input data from files or preprocessed JSON."""
        try:
            # Ensure valid input paths if preprocessed data is not provided.
            if not self.preprocessed_path and (
                    not self.sentences_path or not self.remove_words_path or not self.names_path):
                sys.exit("invalid input")  # Prevents attempting to read from a None path

            if self.preprocessed_path:
                # Load preprocessed data from JSON file.
                with open(self.preprocessed_path, 'r') as f:
                    self.preprocessed_data = json.load(f)
                self.sentences = self.preprocessed_data["Question 1"]["Processed Sentences"]
                self.names = self.preprocessed_data["Question 1"]["Processed Names"]
            else:
                # Process raw data using the TextProcessor class.
                self.processor = TextProcessor(self.sentences_path, self.remove_words_path, self.names_path)
                self.processor.load_data()
                self.sentences = self.processor.process_sentences()
                self.names = self.processor.process_names()

        except (FileNotFoundError, json.JSONDecodeError, ValueError, PermissionError):
            sys.exit("invalid input")

    def _generate_k_seqs(self, sentence: List[str]) -> List[List[str]]:
        """Generates K-seqs for a single sentence."""
        k_seqs = []
        for k in range(1, self.max_k + 1):
            if len(sentence) < k:
                continue
            for i in range(len(sentence) - k + 1):
                k_seq = sentence[i:i + k]
                k_seqs.append(k_seq)
        return k_seqs

    def _get_name_contexts(self) -> List[List[Union[str, List[List[str]]]]]:
        """Finds sentences where people are mentioned and generates K-seqs for those sentences."""
        name_contexts = []

        for name_record in self.names:
            # Extract the main name and possible nicknames.
            main_name = " ".join(name_record[0])
            nicknames = [" ".join(nickname) for nickname in name_record[1]]
            all_names = [main_name] + nicknames + main_name.split()  # Includes full name and variations.
            relevant_sentences = []
            for sentence in self.sentences:
                sentence_str = " ".join(sentence)
                # Check if any variation of the name appears in the sentence.
                for name in all_names:
                    if name in sentence_str:
                        relevant_sentences.append(sentence)
                        break  # Avoid adding the same sentence multiple times.
            k_seqs_set = set()
            for sentence in relevant_sentences:
                sentence_k_seqs = self._generate_k_seqs(sentence)
                for k_seq in sentence_k_seqs:
                    k_seqs_set.add(tuple(k_seq))  # Ensure unique sequences.
            sorted_k_seqs = sorted([list(k_seq) for k_seq in k_seqs_set])
            if sorted_k_seqs:
                name_contexts.append([main_name, sorted_k_seqs])

        # Sort the final list of name contexts alphabetically by name.
        name_contexts.sort(key=lambda x: x[0])
        return name_contexts

    def generate_output(self, question_number: int) -> None:
        """Generate and print the output in JSON format for the given task."""
        name_contexts = self._get_name_contexts()
        output = {
            f"Question {question_number}": {
                "Person Contexts and K-Seqs": name_contexts
            }
        }
        print(json.dumps(output, indent=4))


class GraphAnalyzer:
    """Task 6: Builds and analyzes a graph of connections between names based on sentence proximity."""

    def __init__(self, sentences_path, names_path, remove_words_path, window_size, threshold):
        """Initializes with paths, window size, and threshold for connections."""
        self.sentences_path = sentences_path
        self.names_path = names_path
        self.remove_words_path = remove_words_path
        self.window_size = window_size
        self.threshold = threshold
        self.text_processor = TextProcessor(sentences_path, remove_words_path, names_path)
        self.processed_sentences = []
        self.processed_names = []

    def load_and_process_data(self) -> None:
        """Loads and processes sentences and names from the provided files."""
        self.text_processor.load_data()
        self.processed_sentences = self.text_processor.process_sentences()
        self.processed_names = self.text_processor.process_names()

    def generate_windows(self) -> List[List[List[str]]]:
        """Generates overlapping sentence windows based on the specified window size."""
        windows = []
        for i in range(len(self.processed_sentences) - self.window_size + 1):
            window = self.processed_sentences[i:i + self.window_size]
            windows.append(window)
        return windows

    def count_connections(self, windows: List[List[List[str]]]) -> Dict[Tuple[str, str], int]:
        """Counts how many times pairs of names appear together in the sentence windows."""
        connection_counts = {}
        for window in windows:
            people_in_window = set()
            for sentence in window:
                for name_record in self.processed_names:
                    main_name = " ".join(name_record[0])
                    nicknames = [" ".join(nickname) for nickname in name_record[1]]
                    # Include all name variations (full name, nicknames, individual words).
                    all_names = [main_name] + nicknames + main_name.split()
                    if any(name in sentence for name in all_names):
                        people_in_window.add(main_name)

            # Sort names to ensure pairs are always in the same order
            sorted_people = sorted(people_in_window)
            for i in range(len(sorted_people)):
                for j in range(i + 1, len(sorted_people)):
                    person1 = sorted_people[i]
                    person2 = sorted_people[j]
                    if (person1, person2) not in connection_counts:
                        connection_counts[(person1, person2)] = 0
                    connection_counts[(person1, person2)] += 1  # Increase count if they appear together.
        return connection_counts

    def filter_connections(self, connection_counts: Dict[Tuple[str, str], int]) -> List[List[List[str]]]:
        """Filters pairs of names with connections meeting the threshold."""
        filtered_connections = []
        for (person1, person2), count in connection_counts.items():
            # Only keep pairs that appear together at least 'threshold' times.
            if count >= self.threshold:
                pair = sorted([person1.split(), person2.split()])
                filtered_connections.append(pair)
        filtered_connections.sort()
        return filtered_connections

    @staticmethod
    def generate_output(connections) -> None:
        """Generate and print the output in JSON format for the given task."""
        output = {
            "Question 6": {
                "Pair Matches": connections
            }
        }
        print(json.dumps(output, indent=4))


class IndirectConnections:
    """Task 7 & Task 8: Determines if two names are indirectly connected in a graph through any path.
    - Task 7: Checks if two people are connected within a given maximum distance.
    - Task 8: Checks if two people are connected through an exact number of steps.
    """

    def __init__(self, graph: Dict[str, List[str]], connections_path: str, maximal_distance: Optional[int],
                 fixed_length: Optional[int] = None, is_task_8: bool = False):
        """Initializes the graph, path to connections, and allowed distance or fixed steps."""
        self.graph = graph
        self.connections_path = connections_path
        self.max_distance = maximal_distance
        self.fixed_length = fixed_length  # Added for Task 8
        self.is_task_8 = is_task_8  # Flag to differentiate between task 7 and task 8
        self.person_pairs = []

    def load_data(self) -> None:
        """Loads name pairs from a file and checks if the format is valid."""
        with open(self.connections_path, "r") as f:
            data = json.load(f)
            self.person_pairs = data.get("keys", [])
        if not all(isinstance(pair, list) and len(pair) == 2 for pair in self.person_pairs):
            sys.exit("invalid input")

    @staticmethod
    def convert_to_graph(connections: List[List[List[str]]]) -> Dict[str, List[str]]:
        """Converts connections (pairs of names) into a graph structure."""
        graph = {}
        for pair in connections:
            person1 = " ".join(pair[0])
            person2 = " ".join(pair[1])
            # Initialize empty lists for each person if they are not already in the graph.
            if person1 not in graph:
                graph[person1] = []
            if person2 not in graph:
                graph[person2] = []

            # Add each person to the other's adjacency list to create connections.
            graph[person1].append(person2)
            graph[person2].append(person1)
        return graph

    def is_connected(self, name1: str, name2: str, visited: Optional[Set[str]] = None,
                     distance: int = 0, fixed_length: Optional[int] = None) -> bool:
        """Checks if two names are connected within the allowed distance in the graph."""
        if visited is None:
            visited = set()

        # If distance exceeds the allowed limit, return False
        if self.max_distance is not None and distance > self.max_distance:
            return False

        # If we've reached the target person
        if name1 == name2:
            # If we are in task 8, we need to check if the path length is exactly fixed_length
            if self.is_task_8 and distance == self.fixed_length:
                return True
            # Otherwise, for task 7, return True immediately
            elif not self.is_task_8:
                return True
            else:
                return False  # If distance doesn't match fixed_length in task 8, return False

        visited.add(name1)

        # Explore neighbors recursively
        for neighbor in self.graph.get(name1, []):
            if neighbor not in visited:
                if self.is_connected(neighbor, name2, visited, distance + 1, fixed_length):
                    return True  # If any recursive call returns True, propagate it up

        return False  # If no valid path is found, return False

    def check_connections(self) -> Dict[str, Dict[str, List[List[str]]]]:
        """Checks all pairs of names for connection and prints the results in sorted JSON format."""
        results = []
        for pair in self.person_pairs:
            if len(pair) != 2:
                sys.exit("invalid input")  # Ensure the input format is valid before processing.

            name1, name2 = sorted(pair)
            connected = self.is_connected(name1, name2, fixed_length=self.fixed_length)
            results.append([name1, name2, connected])

        results_sorted = sorted(results, key=lambda x: (x[0], x[1]))
        # Determine question number based on whether fixed_length is provided
        question_number = "Question 8" if self.is_task_8 else "Question 7"
        return {question_number: {"Pair Matches": results_sorted}}
