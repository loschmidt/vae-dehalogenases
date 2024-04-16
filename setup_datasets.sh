#!/bin/bash

## untar datasets
tar -zxvf datasets.tar.gz

## copy datasets.tar.gz to datasets directory for pbs scripts
mv datasets.tar.gz ./datasets/.