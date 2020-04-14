# COVID-19

Kaggle COVID-19 Challenge: https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

<img src="https://www.stlucianewsonline.com/wp-content/uploads/2020/01/CORONAVIRUS.jpg"/>

This is a joint work by [@Hazoom](https://github.com/Hazoom),  [@sarahJune1](https://github.com/sarahJune1) and [@KevinBenass](https://github.com/KevinBenass]).
### Setup Instructions
Install `pipenv` with the following command:

```
$ pip install pipenv
```

Open pipenv environment in a new shell:

```
$ pipenv shell
```

Add the project to PYTHONPATH:

```
$ export PYTHONPATH=$PYTHONPATH:/path/to/covid19/src
```

Install dependencies:

```
$ pipenv sync
```

Install `FastText` dependency from the instructions [here](https://github.com/facebookresearch/fastText#building-fasttext-for-python).