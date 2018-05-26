import argparse
import json
import os
import shutil
import sys

import spacy
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    project_root = os.path.abspath(os.path.realpath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)))))
    parser.add_argument("--data-path", type=str,
                        default=os.path.join(
                            project_root, "data", "train"),
                        help="Path to the Conflict Wikipedia JSON data.")
    parser.add_argument("--save-dir", type=str,
                        help="Path to save cleaned, conflict wikipedia data")
    args = parser.parse_args()

    print("Cleaning Conflict Wikipedia JSON data:")
    try:
        if os.path.exists(args.save_dir):
            # save directory already exists, do we really want to overwrite?
            input("Save directory {} already exists. Press <Enter> "
                  "to clear, overwrite and continue , or "
                  "<Ctrl-c> to abort.".format(args.save_dir))
            shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
    except KeyboardInterrupt:
        print()
        sys.exit(0)

    wiki_json_paths = [os.path.join(args.data_path, wiki_json_path)
                       for wiki_json_path in os.listdir(args.data_path)]
    out_paths = [os.path.join(args.save_dir, wiki_json_path)
                 for wiki_json_path in os.listdir(args.data_path)]

    nlp = spacy.load('en_core_web_sm')
    exclude_sections = ["References", "Bibliography", "See also", "Further reading"]
    print("Saving cleaned, wikipedia JSONs:")
    for input_path, output_path in tqdm(list(zip(wiki_json_paths, out_paths))):
        clean_json(input_path, nlp, exclude_sections, output_path)


def clean_json(file_path, nlp, exclude_sections, out):

    parsed_document = json.load(open(file_path, 'r'))
    if "title" not in parsed_document or "sections" not in parsed_document:
        return None

    sections = [section["text"] for section in parsed_document["sections"]
                if "heading" in section and
                section["heading"] not in exclude_sections]

    grouped_sentences = [nlp(section).sents for section in sections]
    tokenized_sentences = []
    for group in grouped_sentences:
        for sentence in group:
            tokenized_sentence = ' '.join([token.text for token in sentence])
            tokenized_sentence = tokenized_sentence.strip()
            if len(tokenized_sentence):
                tokenized_sentences.append(tokenized_sentence)

    cleaned_example = {
        "title": parsed_document["title"],
        "text": '\n'.join(tokenized_sentences)
    }

    with open(out, 'w') as f:
        json.dump(cleaned_example, f,
                  ensure_ascii=False,
                  indent=4)


def print_progress_in_place(*args):
    print("\r", *args, end="")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
