import argparse as arg
import model

if __name__ == '__main__':
    parser = arg.ArgumentParser(description='Training inputs')

    parser.add_argument('--model', dest='model_json', type=str, required=True,
                        help='relative path to a model file')

    parser.add_argument('--length', dest='length', type=int, required=True,
                        help='number of words to be generated')

    parser.add_argument('--prefix', dest='prefix', type=str, default=None,
                        help='the prefix of a generated text')

    args = parser.parse_args()

    model_json, length, prefix = (args.model_json, args.length, args.prefix)

    my_model = model.NGramModel(model_json)

    print(my_model.generate(length, prefix))