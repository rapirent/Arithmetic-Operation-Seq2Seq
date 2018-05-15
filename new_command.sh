#!/bin/bash
python3 ./subtractor.py "--epoch=2" "--output_name=digit2" "--data_size=75000" "--train_size=45000" "--digits=2"
python3 ./subtractor.py "--epoch=2" "--output_name=digit4" "--data_size=75000" "--train_size=45000" "--digits=4"

python3 ./subtractor.py "--epoch=2" "--output_name=digit4-train36000" "--data_size=66000" "--train_size=36000" "--digits=4"
python3 ./subtractor.py "--epoch=2" "--output_name=digit4-train45000" "--data_size=75000" "--train_size=45000" "--digits=4"


