# Tensorflow Keras Boilerplate

Tensorflow Keras Boilerplate using Tensorflow GPU

## Requirement

- Tensorflow GPU (2.9.1)
- pydot (1.4.2)
- Graphviz (2.42.2)

```shell
pip install pydot
sudo apt install graphviz
```

## Train

```
python Train.py
```

![train_result.png](./example/train_result.png)

## Test

> Need `Train` step

```shell
python Test.py
```

![test_result.png](./example/test_result.png)

![output_model_result.png](./example/output_model_result.png)

## Predict

> Need `Train`, `Test` step

```shell
python Predict.py
```

![predict_result.png](./example/predict_result.png)

## Tensorboard

> Need `Train` step

```shell
tensorboard --logdir=logs --host=0.0.0.0 --port=6006
```

![tensorboard_result.png](./example/tensorboard_result.png)
