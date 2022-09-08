import model

import argparse as arg
import os

if __name__ == '__main__':
    parser = arg.ArgumentParser(description='Training inputs')

    parser.add_argument('--input_dir', dest='input_dir', type=str, required=True,
                        help='relative path to a folder with training data')

    parser.add_argument('--model', dest='model_storage', type=str, required=True,
                        help='relative path to a file, where trained model will be stored')

    parser.add_argument('--gram_length', dest='gram_length', type=int, default=4,
                        help='N of a N-Gram Model model')
    args = parser.parse_args()

    input_dir, model_storage, gram_length = (args.input_dir, args.model_storage, args.gram_length)

    (_, _, filenames) = next(os.walk(input_dir))

    my_model = model.NGramModel()
    my_model.set_gram_length(gram_length)

    for filename in filenames:
        if filename[-3:] == "txt":
            print(filename + " learning started")
            with open(input_dir + r"/" + filename, "r") as file:
                text = file.read()
                my_model.fit(text)
            print(filename + " learned")

    print("storing model")
    my_model.dump(model_storage)