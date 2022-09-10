import argparse as arg
import model
import neural_model

if __name__ == '__main__':
    parser = arg.ArgumentParser(description='Training inputs')

    parser.add_argument('--model', dest='model_dst', type=str, required=True,
                        help='relative path to a model file')

    parser.add_argument('--length', dest='length', type=int, required=True,
                        help='number of words to be generated')

    parser.add_argument('--prefix', dest='prefix', type=str, default="",
                        help='the prefix of a generated text')

    parser.add_argument('--model_type', dest='nn_type', type=bool, default=False,
                        help='the generator will use nn model')

    args = parser.parse_args()

    model_dst, length, prefix, nn_type = (args.model_dst, args.length, args.prefix, args.nn_type)

    if nn_type:
        my_model = neural_model.NeuralModel(storage_location=model_dst)
    else:
        my_model = model.NGramModel(model_dst)

    print(my_model.generate(length, prefix))
