#!/usr/bin/env python3
import argparse
import json
import sys
from text_tasks import TextProcessor, KSeqAnalyzer, PersonMentionCounter
from basic_search_engine import BasicSearchEngine, PersonContextsWithKSeqs, GraphAnalyzer, IndirectConnections
from advanced_text_analysis import SentenceGrouping


def readargs(args=None):
    parser = argparse.ArgumentParser(
        prog='Text Analyzer project',
    )
    # General arguments
    parser.add_argument('-t', '--task',
                        help="task number",
                        required=True,
                        type=int
                        )
    parser.add_argument('-s', '--sentences',
                        help="Sentence file path",
                        )
    parser.add_argument('-n', '--names',
                        help="Names file path",
                        )
    parser.add_argument('-r', '--removewords',
                        help="Words to remove file path",
                        )
    parser.add_argument('-p', '--preprocessed',
                        action='append',
                        help="json with preprocessed data",
                        )
    # Task specific arguments
    parser.add_argument('--maxk',
                        type=int,
                        help="Max k",
                        )
    parser.add_argument('--fixed_length',
                        type=int,
                        help="fixed length to find",
                        )
    parser.add_argument('--windowsize',
                        type=int,
                        help="Window size",
                        )
    parser.add_argument('--pairs',
                        help="json file with list of pairs",
                        )
    parser.add_argument('--threshold',
                        type=int,
                        help="graph connection threshold",
                        )
    parser.add_argument('--maximal_distance',
                        type=int,
                        help="maximal distance between nodes in graph",
                        )

    parser.add_argument('--qsek_query_path',
                        help="json file with query path",
                        )
    return parser.parse_args(args)


def main():
    args = readargs()

    if args.task == 1:
        try:
            processor = TextProcessor(
                sentences_path=args.sentences,
                remove_words_path=args.removewords,
                names_path=args.names
            )
            processor.load_data()
            processed_sentences = processor.process_sentences()
            processed_names = processor.process_names()
            processor.generate_output(
                question_number=args.task,
                processed_sentences=processed_sentences,
                processed_names=processed_names
            )
        except RuntimeError:
            sys.exit("invalid input")

    elif args.task == 2:
        if args.preprocessed:
            # Load preprocessed data from JSON file
            with open(args.preprocessed[0], "r") as f:
                preprocessed_data = json.load(f)
            sentences = preprocessed_data["Question 1"]["Processed Sentences"]
        else:
            # Process sentences using Task 1 logic
            processor = TextProcessor(
                sentences_path=args.sentences,
                remove_words_path=args.removewords,
                names_path=args.names
            )
            processor.load_data()
            sentences = processor.process_sentences()

        # Ensure max_k is at least 1, otherwise input is invalid
        if not args.maxk or args.maxk < 1:
            sys.exit("invalid input")
        analyzer = KSeqAnalyzer(sentences, args.maxk)
        analyzer.generate_output(question_number=args.task)

    elif args.task == 3:
        if args.preprocessed:
            with open(args.preprocessed[0], "r") as f:
                preprocessed_data = json.load(f)
            sentences = preprocessed_data["Question 1"]["Processed Sentences"]
            names = preprocessed_data["Question 1"]["Processed Names"]
        else:
            # Process sentences and names using Task 1 logic
            processor = TextProcessor(
                sentences_path=args.sentences,
                remove_words_path=args.removewords,
                names_path=args.names
            )
            processor.load_data()
            sentences = processor.process_sentences()
            names = processor.process_names()
        analyzer = PersonMentionCounter(sentences, names)
        analyzer.generate_output(question_number=args.task)

    elif args.task == 4:
        try:
            preprocessed_path = args.preprocessed[0] if args.preprocessed else None
            engine = BasicSearchEngine(
                sentences_path=args.sentences,
                remove_words_path=args.removewords,
                kseq_path=args.qsek_query_path,
                preprocessed_path=preprocessed_path
            )
            engine.load_data()
            engine.generate_output(question_number=args.task)
        except RuntimeError:
            sys.exit("invalid input")

    elif args.task == 5:
        try:
            analyzer = PersonContextsWithKSeqs(
                sentences_path=args.sentences,
                remove_words_path=args.removewords,
                names_path=args.names,
                max_k=args.maxk,
                preprocessed_path=args.preprocessed[0] if args.preprocessed else None
            )
            analyzer.load_data()
            analyzer.generate_output(question_number=args.task)
        except RuntimeError:
            sys.exit("invalid input")

    elif args.task == 6:
        try:
            if args.preprocessed:
                with open(args.preprocessed[0], "r") as f:
                    preprocessed_data = json.load(f)

                processed_sentences = preprocessed_data["Question 1"]["Processed Sentences"]
                processed_names = preprocessed_data["Question 1"]["Processed Names"]

                analyzer = GraphAnalyzer(
                    sentences_path=None,
                    names_path=None,
                    remove_words_path=None,
                    window_size=args.windowsize,
                    threshold=args.threshold
                )

                analyzer.processed_sentences = processed_sentences
                analyzer.processed_names = processed_names

            else:
                analyzer = GraphAnalyzer(
                    sentences_path=args.sentences,
                    names_path=args.names,
                    remove_words_path=args.removewords,
                    window_size=args.windowsize,
                    threshold=args.threshold
                )
                analyzer.load_and_process_data()

            windows = analyzer.generate_windows()
            connection_counts = analyzer.count_connections(windows)
            filtered_connections = analyzer.filter_connections(connection_counts)
            analyzer.generate_output(filtered_connections)

        except RuntimeError:
            sys.exit("invalid input")

    elif args.task == 7:
        try:
            if args.preprocessed:
                with open(args.preprocessed[0], "r") as f:
                    preprocessed_data = json.load(f)
                connections = preprocessed_data["Question 6"]["Pair Matches"]
                graph = IndirectConnections.convert_to_graph(connections)
            else:
                graph_analyzer = GraphAnalyzer(
                    sentences_path=args.sentences,
                    names_path=args.names,
                    remove_words_path=args.removewords,
                    window_size=args.windowsize,
                    threshold=args.threshold
                )
                graph_analyzer.load_and_process_data()
                windows = graph_analyzer.generate_windows()
                connection_counts = graph_analyzer.count_connections(windows)
                connections = graph_analyzer.filter_connections(connection_counts)
                graph = IndirectConnections.convert_to_graph(connections)
            analyzer = IndirectConnections(
                graph=graph,
                connections_path=args.pairs,
                maximal_distance=args.maximal_distance,
            )
            analyzer.load_data()
            indirect_connections = analyzer.check_connections()
            print(json.dumps(indirect_connections, indent=4))

        except RuntimeError:
            sys.exit("invalid input")

    elif args.task == 8:
        try:
            if args.preprocessed:
                with open(args.preprocessed[0], "r") as f:
                    preprocessed_data = json.load(f)
                connections = preprocessed_data["Question 6"]["Pair Matches"]
                graph = IndirectConnections.convert_to_graph(connections)
            else:
                graph_analyzer = GraphAnalyzer(
                    sentences_path=args.sentences,
                    names_path=args.names,
                    remove_words_path=args.removewords,
                    window_size=args.windowsize,
                    threshold=args.threshold
                )
                graph_analyzer.load_and_process_data()
                windows = graph_analyzer.generate_windows()
                connection_counts = graph_analyzer.count_connections(windows)
                connections = graph_analyzer.filter_connections(connection_counts)
                graph = IndirectConnections.convert_to_graph(connections)
            # Initialize IndirectConnections with the fixed length requirement
            analyzer = IndirectConnections(
                graph=graph,
                connections_path=args.pairs,
                maximal_distance=None,  # Task 8 does not use max distance
                fixed_length=args.fixed_length,  # Task 8 requires exact distance
                is_task_8=True  # Enable task 8 mode
            )
            analyzer.load_data()
            result = analyzer.check_connections()
            print(json.dumps(result, indent=4))
        except RuntimeError:
            sys.exit("invalid input")

    elif args.task == 9:
        try:
            if args.preprocessed:
                with open(args.preprocessed[0], "r") as f:
                    preprocessed_data = json.load(f)
                processed_sentences = preprocessed_data["Question 1"]["Processed Sentences"]
            else:
                text_processor = TextProcessor(
                    sentences_path=args.sentences,
                    remove_words_path=args.removewords,
                    names_path=None
                )
                text_processor.load_data()
                processed_sentences = text_processor.process_sentences()

            sentence_grouping = SentenceGrouping(processed_sentences=processed_sentences, threshold=args.threshold)
            groups = sentence_grouping.group_sentences()
            sentence_grouping.generate_output(groups)

        except RuntimeError:
            sys.exit("invalid input")


if __name__ == "__main__":
    main()
