# Predictive Flood Modeling with Multimodal Transformer-based Architecture

This repository contains a proof-of-concept (PoC) for applying transformer-based architectures to predictive flood modeling using multimodal data sources. The goal of this project is to demonstrate the potential of leveraging state-of-the-art techniques, such as unrolling 2D data and combining it with time series weather data, to effectively incorporate both 2D and 1D multivariate modalities into a single time-forecast network.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Accurate and timely prediction of flood events is crucial for effective disaster response and resource allocation. This project explores the application of transformer-based architectures, inspired by the success of DETR (DEtection TRansformer) in computer vision, to predictive flood modeling. By unrolling 2D data and combining it with time series weather data, we aim to demonstrate the potential of transformers in effectively incorporating multimodal data sources for improved flood prediction.

As everyones aware of LLMs more orr less - here you can think of the Question being the past data, the Answer being the future logits (for now just used to creat a binary prediction; think of it as if propting an LLM with: Input: Question[512hrs], and getting Output[96hrs]: Plan[24hrs] (this is where the prediction currrently happens by taking binary cross entropy of the Plan logits with a binary class of Flood/No flood of the last 24hrs)) and Answer[24hrs] (this is currently the "ground truth" of where the flood either happens or not. To fit with the pre-trainend dimensions of an Open Source fine-tune of this model, I had to leave the last 48 hours unattended.

It might make a lot more sense to train the model regressively on a contigious stream of data of one (or many) flood prone regions, then have two predictive regimes:

### Predictive/Training Regime 1

Regressive: Feed it data as I preprocessed it in real time, and allow it to spit out 24 hours or so of flood warning forcast by taking the embedding and applying the standard "tokenisation" back to the (a) next time step in the input data [in LLM speak: predict the next word] (b) less-standrad tokenization to predict the probability of a flood, incorporated by adding an extra embedding layer with some ReLU and softmax 


### Predictive/Training Regime 1

Autoregressive: Use the pre-trained "extra embedding layer with some ReLU and softmax" layer as one path way for the model output, with the other one being autoregression -> this way you'd get prediction of how the "world model" evolves and as a sideeffect; if you get a flood or not. Can be useful to check if the model is applicable to a new region, without having to train/fine-tune it. Akeen to instruction fine-tuning in LLMs, the DPO (see the original paper for details: https://arxiv.org/pdf/2305.18290) can come in at the final embedded feature level, thus avoiding unlearning what's been previously learned (better at generalizing) and requiring only non-flood data to predict flood data (potentially - this would need to  be rigorously proven) in a new region.


## Future Work and some Methodology
The proposed approach leverages the power of transformer-based architectures to handle multimodal data sources effectively. The key aspects of the methodology include:

0. (Forgot to add this, sorry) Modality wise drop out could help this model generalise to, for example, regions with some of the modalities being more sparse.

1. Unrolling 2D data: Inspired by the DETR architecture, we unroll 2D data sources, such as satellite imagery and digital elevation models (DEM), into a sequence of tokens that can be processed by the transformer model.

2. Combining with time series data: Time series weather data, such as precipitation and temperature, is incorporated into the model to capture the temporal dynamics and improve flood prediction accuracy.

3. Transformer-based architecture: The unrolled 2D data and time series data are fed into a transformer-based architecture, which learns to capture the complex relationships and dependencies between the multimodal data sources for effective flood prediction.

4. Transfer learning and efficient fine-tuning: The transformer-based model can benefit from pre-training on large-scale datasets and efficient fine-tuning techniques, such as DPO (Dynamic Projection Operator) and QLoRA (Quantization-aware Low-Rank Adaptation), to enable transferability and adaptability to different regions and data sources.

5. Autoregressive regime: The model can be extended to an autoregressive regime, where the model's predictions are fed back as input for the next time step, enabling real-time monitoring and early response to flood events.

6. "Curriculum learning in time": Increase the time delta between each token, let the model performance recover, then increase it some more. WOuld be interesting to see how far into the future the model can predict, and how sparse the data can be (given enough training and parameters)

## Repository Structure
The repository is structured as follows:
- `data/`: Contains the data collection and loading scripts.
  - `data_collector.py`: Script for collecting and preprocessing data from various sources.
  - `dataloader.py`: Script for loading and batching the preprocessed data.
- `model/`: Includes the source code for the transformer-based flood prediction model.
  - `modeltrainer.py`: Script for training and evaluating the flood prediction model.
  - `modeltrainer-visual.py`: Script for training the model with visual training dashboard.
- `flood-predict-idea.ipynb`: Jupyter notebook showcasing the initial idea and experimentation.
- `img.png`: Image file used in the notebook.
- `README.md`: Provides an overview of the project and instructions for running the code.
- `REPORT.md`: Detailed report on the methodology, results, and future enhancements.

## Installation
To set up the project environment, follow these steps:
1. Clone the repository: `git clone https://github.com/your-username/predictive-flood-modeling.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Set up the necessary data sources and update the configuration files in the `config/` directory.

## Usage
To train and evaluate the flood prediction model, run through the corresponding steps of the notebooks

TO visualise the evolution of validation results use (thought it might be cool if this is scaled up and/or made autoregressive [to see the model performance evolve over time using  data from new region(s)] we pull data on the fly - to see how the model evolves with more data - at some point you will get catastrophic forgetting though, so swithcing to something more clever like DPO might be a good idea):
```
python model/modeltrainer-visual.py --config config/model_config.yaml
```

For more detailed usage instructions and examples, refer to the documentation in the `docs/` directory.

## Future Enhancements
This PoC demonstrates the potential of applying transformer-based architectures to predictive flood modeling using multimodal data sources. However, there are several areas for future enhancements and exploration:

1. Extensive testing and debugging: As this PoC was developed within a limited timeframe of 4-6 hours, further testing and debugging are necessary to ensure the robustness and reliability of the model.

2. Vary the input and output context length of the model to see where the sweet spot is, as well as the time resolution of each "token", and spatial resolution of the region being observed.

2. Incorporation of additional data sources: Exploring the integration of other relevant data sources, such as soil moisture and land cover data, to further improve the model's performance.

3. Optimization of model architecture: Investigating and optimizing the transformer-based architecture to better capture the complex relationships between the multimodal data sources and improve flood prediction accuracy.

4. Real-time monitoring and early response: Extending the model to an autoregressive regime to enable real-time monitoring and early response to flood events.

5. Transfer learning and domain adaptation: Leveraging pre-trained models and efficient fine-tuning techniques to enable transferability and adaptability to different regions and data sources.

## Contributing
Contributions to this project are welcome. If you would like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository, explaining your changes and their benefits.

## License
This project is licensed under the MIT license. For more information, see the not yet existant `LICENSE` file.
