import pandas as pd
import re
import json
from typing import List, Dict, Union, Optional
import sys


class TextProcessor:
    """Task 1: A class for processing sentences and names from files."""

    def __init__(self, sentences_path: str, remove_words_path: str, names_path: Optional[str] = None) -> None:
        """Initializes the class with file paths."""
        self.sentences_path: str = sentences_path
        self.remove_words_path: str = remove_words_path
        self.names_path: str = names_path
        self.sentences: List[str] = []
        self.remove_words: List[str] = []
        self.names: List[Dict[str, Union[str, float]]] = []

    def load_data(self) -> None:
        """Loads data from input files."""
        try:
            sentences_df = pd.read_csv(self.sentences_path, header=0)
            if 'sentence' not in sentences_df.columns or sentences_df.empty:
                sys.exit("invalid input")
            self.sentences = sentences_df['sentence'].dropna().tolist()
            # Load and validate the remove words file
            remove_words_df = pd.read_csv(self.remove_words_path, header=0)
            if 'words' not in remove_words_df.columns or remove_words_df.empty:
                sys.exit("invalid input")
            self.remove_words = remove_words_df['words'].dropna().tolist()

            if self.names_path is not None:
                names_df = pd.read_csv(self.names_path, header=0)
                if 'Name' not in names_df.columns or 'Other Names' not in names_df.columns or names_df.empty:
                    sys.exit("invalid input")
                self.names = names_df.dropna(subset=['Name']).to_dict(orient='records')

        except (FileNotFoundError, json.JSONDecodeError, ValueError, PermissionError):
            # Print a general error message if any of the specified errors occur
            sys.exit("invalid input")

    def clean_sentence(self, sentence: str) -> List[str]:
        """ Cleans a sentence by removing punctuation, unwanted words, and extra spaces."""
        sentence = sentence.lower()
        sentence = re.sub(r'-', ' ', sentence)
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        words = sentence.split()
        words = [word for word in words if word not in self.remove_words]
        return words

    def process_sentences(self) -> List[List[str]]:
        """Processes all sentences and returns cleaned results."""
        processed_sentences = []
        for sentence in self.sentences:
            words = self.clean_sentence(sentence)
            if words:
                processed_sentences.append(words)
        return processed_sentences

    def process_names(self) -> List[List[Union[List[str], List[List[str]]]]]:
        """Cleans and organizes names and nicknames."""
        processed_names = []
        seen_names = set()
        seen_nicknames = set()
        for name_record in self.names:
            name = name_record['Name'].lower().strip()
            if not name:
                continue  # Skip empty names

            name_words = [word for word in self.clean_sentence(name) if word not in self.remove_words]
            if tuple(name_words) in seen_names:
                continue  # Skip duplicate names
            seen_names.add(tuple(name_words))

            # Process nicknames, ensuring they are unique
            other_names = [n.strip() for n in name_record['Other Names'].split(',') if n.strip()] if pd.notna(
                name_record['Other Names']) else []
            other_names = [
                self.clean_sentence(n) for n in other_names
                if tuple(self.clean_sentence(n)) not in seen_nicknames and n.strip()
            ]
            seen_nicknames.update([tuple(n) for n in other_names])
            processed_names.append([name_words, other_names])

        return processed_names

    @staticmethod
    def generate_output(question_number: int, processed_sentences: List[List[str]],
                        processed_names: List[List[Union[str, List[str]]]]) -> None:
        """Generate and print the output in JSON format for the given task."""
        output = {
            f"Question {question_number}": {
                "Processed Sentences": processed_sentences,
                "Processed Names": processed_names
            }
        }
        print(json.dumps(output, indent=4))


class KSeqAnalyzer:
    """Task 2: Analyze k-length sequences from sentences"""

    def __init__(self, sentences: List[List[str]], max_k: int):
        """Initialize the analyzer with sentences and maximum sequence length (k)."""
        self.sentences = sentences
        self.max_k = max_k

    def _generate_k_seq_counts(self, k: int) -> Dict[str, int]:
        """Generate counts of all k-length sequences in the sentences."""
        k_seq_counts = {}
        for sentence in self.sentences:
            if len(sentence) < k:
                continue

            # Loop through the sentence to extract all k-length subsequences
            for i in range(len(sentence) - k + 1):
                k_seq = " ".join(sentence[i:i + k])
                k_seq_counts[k_seq] = k_seq_counts.get(k_seq, 0) + 1
        return k_seq_counts

    def analyze_sequences(self) -> List[List[Union[str, List[List[Union[str, int]]]]]]:
        """Analyze and collect k-length sequences for all k from 1 to max_k."""
        # If max_k is larger than the longest sentence, return an empty list since no valid k-seq can be formed.
        if all(len(sentence) < self.max_k for sentence in self.sentences):
            return []
        results = []
        for k in range(1, self.max_k + 1):
            k_seq_counts = self._generate_k_seq_counts(k)

            # Sort sequences lexicographically for consistent output
            sorted_k_seq_counts = sorted(k_seq_counts.items(), key=lambda x: x[0])
            results.append([f"{k}_seq", [[seq, count] for seq, count in sorted_k_seq_counts]])
        return results

    def generate_output(self, question_number: int) -> None:
        """Generate and print the output in JSON format for the given task."""
        sequences = self.analyze_sequences()
        output = {
            f"Question {question_number}": {
                f"{self.max_k}-Seq Counts": sequences
            }
        }
        print(json.dumps(output, indent=4))


class PersonMentionCounter:
    """Task 3: Count mentions of names and nicknames in sentences"""

    def __init__(self, processed_sentences: List[List[str]], processed_names: List[List[List[str]]]):
        """Initializes the class with preprocessed sentences and names."""
        self.sentences = processed_sentences
        self.names = processed_names

    def _prepare_names(self) -> List[Dict[str, List[str]]]:
        """Prepares names and nicknames into a dictionary format."""
        prepared_names = []
        for name_record in self.names:
            main_name = " ".join(name_record[0])  # Convert main name list to string
            nicknames = [" ".join(nickname) for nickname in name_record[1]]  # Convert nicknames to strings
            prepared_names.append({"Name": main_name, "Nicknames": nicknames})
        return prepared_names

    def count_mentions(self) -> Dict[str, int]:
        """Counts mentions of main names and nicknames in the sentences."""
        prepared_names = self._prepare_names()
        name_counts = {}

        for name_record in prepared_names:
            main_name = name_record["Name"]
            nicknames = name_record["Nicknames"]
            main_name_str = " ".join(main_name) if isinstance(main_name, list) else main_name

            # Split the full name into individual words
            main_name_parts = main_name_str.split()
            count = 0
            for sentence in self.sentences:
                sentence_str = " ".join(sentence)

                # Count occurrences of individual name parts (each word separately)
                for part in main_name_parts:
                    count += sentence_str.count(part)

                # Count occurrences of nicknames (as a whole)
                for nickname in nicknames:
                    count += sentence_str.count(nickname)

            if count > 0:
                name_counts[main_name_str] = count
        return name_counts

    def generate_output(self, question_number: int) -> None:
        """Generate and print the output in JSON format for the given task"""
        name_counts = self.count_mentions()
        sorted_name_counts = sorted(name_counts.items(), key=lambda x: x[0])
        output = {
            f"Question {question_number}": {
                "Name Mentions": [[name, count] for name, count in sorted_name_counts]
            }
        }
        print(json.dumps(output, indent=4))
