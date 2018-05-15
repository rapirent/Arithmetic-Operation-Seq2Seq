# Sequence to Sequnce

## files & directories
- corpus/*: the original corpus of each experiment
- data/*: the performance record of each experiment
- fig/*: the performance analysis figure of each experiment
- models/*: the original model of each experiment
- subtractor.py: the subtractor which are in hw3 requirements
    - `A - B, A>=B`
- addition-subtractor.py: the adder and subtractor
    - `A - B, A + B`
- multiply.py: the adder, subtrctor, multiplier
    - `A - B, A + B, A * B`
- model_load-subtractor.py: the loader which will load a subtractor model
- model_load-addition-subtractor.py: the loader which will load a adder and subtractor model
- model_load-multiply.py: the loader which will load a adder, subtractor, multiplier model

## requirements

- python3.6
- keras
- tensorflow
- more details are writed as a `requirements.txt` file
## usage

- run an experiments
    - use data_size to specify size of whole dataset
    - use train_szie to specify size of train dataset incluing validation dataset
    - use digits to specify the maximum possible digits
    - use epoch to specify the epoch number (but it will multiply with 100)
    - use activation to specify the activation function
    - use output_name to specify the output file name
```sh
$ python3 ./subtractor.py [--data_size DATA_SIZE] [--train_size TRAIN_SIZE]
                     [--digits DIGITS] [--epoch EPOCH]
                     [--activation ACTIVATION] [--output_name OUTPUT_NAME]
$ python3 ./addition-subtractor.py [--data_size DATA_SIZE] [--train_size TRAIN_SIZE]
                     [--digits DIGITS] [--epoch EPOCH]
                     [--activation ACTIVATION] [--output_name OUTPUT_NAME]
$ python3 ./multiply.py [--data_size DATA_SIZE] [--train_size TRAIN_SIZE]
                     [--digits DIGITS] [--epoch EPOCH]
                     [--activation ACTIVATION] [--output_name OUTPUT_NAME]
```


- load the models created by former experiments & re-do a test
    - use model_name to specify the loaded model name (which will refer to files under models/ and corpus/)
    - use model_name to specify the maximum possible digits of the loaded model (***which should be same as digits of original experiments***)
```sh
$ python3 ./model_load-subtractor.py [--model_name MODEL_NAME]
                                         [--digits DIGITS]
$ python3 ./model_load-addition-subtractor.py [--model_name MODEL_NAME]
                                         [--digits DIGITS]
$ python3 ./model_load-multiply.py [--model_name MODEL_NAME]
                                         [--digits DIGITS]
```

## detail and report

You can refer to jupyter notebook [Model.ipynb](https://nbviewer.jupyter.org/github/rapirent/DSAI-HW3/blob/master/Model.ipynb)
