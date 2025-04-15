#!/bin/bash

# Install dependencies if they're not already installed
pip install -r requirements.txt

# Install Schedule-Free from source
pip install git+https://github.com/facebookresearch/schedule_free.git
