# Car Brand Classifier

The following repository contains the source code for training and deploying a machine learning for classifying car brands using Azure ML service.

## To prepare the data

Follow the steps in teh `prep_data.md` file.

## To train model

Run the following command using Azure ML CLI 2.0

```
az ml job create -f job-single-node.yml
```

## To deploy the model

Run the following commands using Azure ML CLI 2.0

```
az ml online-endpoint create -f endpoint.yml
az ml online-deployment create -f deployment.yml
```