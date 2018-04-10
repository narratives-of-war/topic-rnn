#!/bin/bash
set -e

echo 'List files from cached directories'
if [ -d $HOME/download ]; then
    echo 'download:'
    ls $HOME/download
fi
if [ -d $HOME/.cache/pip ]; then
    echo 'pip:'
    ls $HOME/.cache/pip
fi

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
source deactivate

# Add the miniconda bin directory to $PATH
export PATH=/home/travis/miniconda3/bin:$PATH
echo $PATH

# Use the miniconda installer for setup of conda itself
pushd .
cd
mkdir -p download
cd download
if [[ ! -f /home/travis/miniconda3/bin/activate ]]
then
    if [[ ! -f miniconda.sh ]]
    then
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
             -O miniconda.sh
    fi
    chmod +x miniconda.sh && ./miniconda.sh -b -f
    conda update --yes conda
    echo "Creating environment to run tests in."
    conda create -q -n testenv --yes python="$PYTHON_VERSION"
fi

if [[ "$SKIP_TESTS" != "true" ]]; then
    if [[ ! -f /home/travis/download/glove.6B.50d.txt ]]
    then
        if [[ ! -f /home/travis/download/glove.6B.zip ]]
        then
            wget http://nlp.stanford.edu/data/glove.6B.zip
        fi
        unzip glove.6B.zip
        mkdir ~/build/nelson-liu/CSE447_RNN/glove
        mv /home/travis/download/glove.6B.50d.txt ~/build/nelson-liu/CSE447_RNN/glove
    fi
fi
cd ..
popd

# Activate the python environment we created.
echo "Activating Environment"
source activate testenv

# Install requirements via pip in our conda environment
echo "Installing requirements"
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en

# Install pytorch only if we are running tests
if [[ "$SKIP_TESTS" != "true" ]]; then
    echo "Installing PyTorch"
    conda install -q --yes pytorch torchvision -c pytorch
fi
