#!/bin/bash

curl https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip -o processed_celeba_small.zip

unzip processed_celeba_small.zip

rm processed_celeba_small.zip
