import subprocess
import time
from datetime import datetime

from classifier.Model import INPUT_LENGTH
from model.Sequence import Sequence


def parse_paf(input_file):
    timestamp = datetime.now().isoformat()

    line_count = file_line_count(input_file)

    start = time.time()
    with open(file=input_file, mode="r", buffering=1) as paf_handle:
        with open(file="./output/output_" + timestamp + ".tsv", mode="w+") as out_handle:
            current_sequence = Sequence()

            for idx, alignment in enumerate(paf_handle):
                if idx % 1000 == 0:
                    print("Parsing line {}. Percent complete = {:.4f}\n".format(idx, idx / line_count))

                fields = alignment.split(sep="\t")

                # Extract interesting fields
                query_id = fields[0]
                query_len = int(fields[1])
                query_hit_start = int(fields[2])
                query_hit_end = int(fields[3])

                # Uninitialized sequence, init with ID & total length
                if current_sequence.query_id == "-1":
                    current_sequence.setup(query_id, query_len)

                if current_sequence.query_id == query_id:
                    # Still the same query sequence, increment existing values
                    current_sequence.append(query_hit_start, query_hit_end)

                    if idx == line_count - 1:
                        save_sequence(current_sequence, out_handle)
                        return
                else:
                    save_sequence(current_sequence, out_handle)

                    # Setup new sequence & append information from current line
                    current_sequence.setup(query_id, query_len)
                    current_sequence.append(query_hit_start, query_hit_end)

    end = time.time()
    print("\nTotal execution time {:.2f}s".format(end - start))


def save_sequence(sequence, out_handle):
    if sequence.query_len < INPUT_LENGTH:
        sequence.pad_sequence()
    else:
        sequence.trim_sequence()

    # Write the current sequence to file
    sequence.print(out_handle)


def file_line_count(input_file):
    out = subprocess.Popen(
        ['wc', '-l', input_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    ).communicate()[0]

    return int(out.decode('utf-8').strip().split(' ')[0])
