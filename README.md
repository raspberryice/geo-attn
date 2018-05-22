# GeoAttn

## Dependencies
Apart from the packages listed in `requirements.txt`, please install `pytorch` according to your version of Python3 and platform.

## Preprocessing
1. Add the top level path to PYTHONPATH.
(export PYTHONPATH=`pwd`)
2. Set up the path to `ark-tweet-nlp-0.3.2` in  `__init__.py` of `tokenizers`
3. Use tokenizer to preprocess dataset. An example dataset can be seen under `data/working/train` and `data/working/test`.
4. (Optional) Train glove embeddings on the training set of tweets. 
We observed that word embeddings trained on our own dataset > random initialization > pretrained general word embeddings.
## Training
1. Download the pretrained GloVe embeddings (200d twitter)
2. Set the configurations in  `poiqa/config.py`(network settings) and `poiqa/train.py`(runtime settings and files).
3. Run `train.py` Important options are `--mode` and `--network`.

Mode can be selected between `train` and `test`. 
If training, you can also provide a `train_ratio` to use part of the training as validation and use early stopping 
according to the performance on the validation set. When testing, the distance predictions will be saved as a numpy file.

You should provide the name of the model in `--network`. Available models include `mem-attn` (GeoAttn), `bow-mdn` (the MDN-Shared baseline),
`attn-mdn` (GeoAttn without memory module), `mem` (GeoAttn without attention layer), `regression` (the AttnReg baseline).

## Acknowledgements
The code structure is altered from  `facebookresearch/DrQA`.

