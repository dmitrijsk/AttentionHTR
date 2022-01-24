# AttentionHTR

PyTorch implementation of an end-to-end Handwritten Text Recognition (HTR) system based on attention encoder-decoder networks. Scene Text Recognition (STR) benchmark model [1], trained on synthetic scene text images, is used to perform transfer learning from the STR domain to HTR. Different fine-tuning approaches are investigated using the multi-writer datasets: Imgur5K [2] and IAM [3]. 

For more details, refer to our paper at arXiv: (link to appear here)


## Getting started

* Download our pre-trained models from [here](https://drive.google.com/drive/folders/1h6edewgRUTJPzI81Mn0eSsqItnk9RMeO?usp=sharing). Details [below](#our-pre-trained-models).
* Use the models for predictions or fine-tuning on additional datases using the [official PyTorch implementation](https://github.com/clovaai/deep-text-recognition-benchmark) of the STR benchmark [1], or [our fork](https://github.com/dmitrijsk/deep-text-recognition-benchmark) containing an early stopping. Details [below](#predictions-and-fine-tuning).
* Links to datasets: [Imgur5K](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset), [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).


## Our pre-trained models

Download our pre-trained models from [here](https://drive.google.com/drive/folders/1h6edewgRUTJPzI81Mn0eSsqItnk9RMeO?usp=sharing). The names of the .pth files are explained in a table below. There are 6 models in total, 3 for each character set, corresponding to the dataset they perform best on.


| Character set    | Imgur5K                 | IAM                 | Both datasets                 |
| :---             |     :---:               |          :---:       |          :---:             |
| Case-insensitive | AttentionHTR-Imgur5K.pth | AttentionHTR-IAM.pth | AttentionHTR-General.pth |
| Case-sensitive   | AttentionHTR-Imgur5K-sensitive.pth | AttentionHTR-IAM-sensitive.pth | AttentionHTR-General-sensitive.pth |

Print the character sets using the Python `string` module: `string.printable[:36]` for the case-insensitive and `string.printable[:-6]` for the case-sensitive character set.

## Use the models for predictions or fine-tuning

### Partitions

Prepare the train, validation (for fine-tuning) and test (for testing and for predicting on unseen data) partitions with word-level images. For the Imgur5K and the IAM datasets you may use [our scripts](https://github.com/dmitrijsk/AttentionHTR/tree/main/process-datasets).

### LMDB datasets

When using the PyTorch implementation of the STR benchmark model [1], images need to be converted into an LMDB dataset. See [this section](https://github.com/clovaai/deep-text-recognition-benchmark#when-you-need-to-train-on-your-own-dataset-or-non-latin-language-datasets) for details. 

### Predictions and fine-tuning

For fine-tuning and predictions use `train.py` and `text.py`, respectively, as described [here](https://github.com/clovaai/deep-text-recognition-benchmark#training-and-evaluation). If you need an early stopping, you may use [our fork](https://github.com/dmitrijsk/deep-text-recognition-benchmark). The `--patience` parameter sets the number of epochs to wait for the validation loss to decrease below the last minimum. In both cases use the arguments `--Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn` and `--sensitive` for the case-sensitive character set.

## Acknowledgements

* Our implementation is based on [Clova AI's deep text recognition benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
* The authors would like to thank Facebook Research for [the Imgur5K dataset](https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset).
* The computations were performed through resources provided by the Swedish National Infrastructure for Computing (SNIC) at Chalmers Centre for Computational Science and Engineering (C3SE). 

## References

[1]: Baek, J., Kim, G., Lee, J., Park, S., Han, D., Yun, S., ... & Lee, H. (2019). What is wrong with scene text recognition model comparisons? dataset and model analysis. In *Proceedings of the IEEE/CVF International Conference on Computer Vision* (pp. 4715-4723). https://arxiv.org/abs/1904.01906
[2]: Krishnan, P., Kovvuri, R., Pang, G., Vassilev, B., & Hassner, T. (2021). TextStyleBrush: Transfer of Text Aesthetics from a Single Example. *arXiv preprint* arXiv:2106.08385. https://arxiv.org/abs/2106.08385
[3]: Marti, U. V., & Bunke, H. (2002). The IAM-database: an English sentence database for offline handwriting recognition. *International Journal on Document Analysis and Recognition*, 5(1), 39-46. https://doi.org/10.1007/s100320200071

## Citation

(Citation to appear here)

## Contact

Dmitrijs Kass (dmitrijs.kass@it.uu.se)
Ekta Vats (ekta.vats@abm.uu.se)
