import neural_model
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

    parser.add_argument('--model_type', dest='nn_type', type=bool, default=False,
                        help='the generator will use nn model')

    parser.add_argument('--epochs', dest='epochs', type=int, default=100,
                        help='number of epochs the model will be learning')

    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.0003,
                        help='start learning rate of a model')
    args = parser.parse_args()

    input_dir, model_storage, gram_length, nn_type, epochs, learning_rate = (
        args.input_dir, args.model_storage, args.gram_length, args.nn_type, args.epochs, args.learning_rate)

    (_, _, filenames) = next(os.walk(input_dir))

    if nn_type:
        my_model = neural_model.NeuralModel(gram_length)
        text = str()

        for filename in filenames:
            if filename[-4:] == '.txt':
                with open(input_dir + '/' + filename, "r") as file:
                    text += file.read()

        print('preparing training data')
        my_model.fit(text, epochs, learning_rate)

        print('stroing model in ' + model_storage + ".pkl")
        my_model.dump(model_storage)
    else:
        my_model = model.NGramModel()
        my_model.set_gram_length(gram_length)

        for filename in filenames:
            if filename[-4:] == '.txt':
                with open(input_dir + '/' + filename, "r") as file:
                    text = file.read()
                    my_model.fit(text)
                print(filename + " learned")

        print('stroing model in ' + model_storage + ".pkl")
        my_model.dump(model_storage)
