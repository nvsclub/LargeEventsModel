# Large Events Model

This repository contains the code for the paper "Large Events Model: A Foundation Events Model for Soccer" and "Estimating Player Performance in Different Contexts Using Large Events Models".

## How to use*
*Note: This code was developed for WyScout v2 data, specifically the freely available datasets. It currently does not support any other data provider.*
1. Load your data to the data/wyscout/json/ folder.
2. Run the converter to CSV (0001), followed by the Calculate Features (0011).
3. *Optional* Train your own models with notebooks 0111-0113.
4. Use the examples from the remaining notebooks to implement your own applications. The xP+ example provided in the first paper is shown in notebook 0132. Notebooks 021* provide the framework to fine-tune specific contexts using LEMs.
