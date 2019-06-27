import matplotlib.pyplot as plt

from PAFParser import file_line_count


def stats(input_file):
    with open(file=input_file, mode="r", buffering=1) as paf_handle:

        line_count = file_line_count(input_file)

        lengths = []
        current_id = "-1"
        current_query_len = 0
        for idx, line in enumerate(paf_handle):
            if idx % 1000 == 0:
                print("\nParsing line {}. Percent complete = {:.2f}\n".format(idx, idx / line_count))

            fields = line.split(sep="\t")

            if current_id == "-1":
                # print("Init first value")
                current_id = fields[0]
                current_query_len = int(fields[1])
            elif current_id == fields[0]:
                # print("Skip entry for the same sequence")
                continue
            else:
                # print("Write down sequence length & set new current sequence")
                lengths.append(current_query_len)
                current_id = fields[0]
                current_query_len = int(fields[1])

        average = sum(lengths) // len(lengths)
        max_val = max(lengths)
        min_val = min(lengths)
        print("Max \\ Min \\ Average")
        print("{} \\ {} \\ {}".format(max_val, min_val, average))

        labels = ["(0, 1000)", "(1000, 3000)", "(3000, 5000)", "(5000, 10000)", "(10000, 15000)", "(15000, 20000)",
                  "(20000, 25000)", "(25000, 30000)"]

        bins = [(0, 1000), (1000, 3000), (3000, 5000), (5000, 10000), (10000, 15000), (15000, 20000), (20000, 25000),
                (25000, 30000)]

        bin_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in lengths:
            for idx, bin in enumerate(bins):
                if bin[0] <= i <= bin[1]:
                    bin_count[idx] += 1
                    continue

        print()
        print(bin_count)

        plt.hist(lengths, color='b')
        plt.axhline(y=average, color='r', linestyle='-')
        plt.show()
