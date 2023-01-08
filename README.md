## Sentimental Analysis: LSTM, Bi-LSTM, Pyramid, and BERT models

<p align='justify'>
<strong>Sentimental Analysis</strong> is one of the most exciting topics in machine learning and natural language processing. Based on this analysis, you can find the feeling of a sentence. This task is a classification project, and there are many sentences that the model should classify as <strong>'Good,' 'Bad,' </strong>and <strong>'Natural.'</strong> For instance, these models can classify a sentence like <strong>'The weather is great.'</strong> as a <strong>good</strong> feeling, <strong>'This story is terrible.'</strong> as a <strong>bad</strong> one, and <strong>'This is a book.'</strong> as a <strong>natural</strong> one. There are many models to analyze the feeling of a sentence, but here we implement four models based on the pictures below.
</p>

<br />
<p align="center">
<img src="https://github.com/HosseinPAI/Sentimental-Analysis/blob/master/pics/LSTM.png" alt="Simple LSTM" width="300" height='200'/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/HosseinPAI/Sentimental-Analysis/blob/master/pics/Bi_LSTM.png" alt="Bidirectional LSTM" width="300" height='200'/>
</p>
<p align="center">
<strong>Simple LSTM </strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>Bidirectional LSTM</strong>
</p>

<br />
<p align="center">
<img src="https://github.com/HosseinPAI/Sentimental-Analysis/blob/master/pics/Pyramid.png" alt="Pyramid Model" width="300" height='200'/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://github.com/HosseinPAI/Sentimental-Analysis/blob/master/pics/bert.png" alt="Bert Model" width="300" height='200'/>
</p>
<p align="center">
<strong>Pyramid Model</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>Bert Model</strong>
</p>

<br />
<p align="justify">
This project is an implementation of various models to analyze the feeling of sentences and intends to compare the power and differences of these models in this category.
</p>
<p align="justify">
You should follow the below steps to run this project and show results. The dataset size, the number of sentences, and the depth of models affect the time to prepare data and train models, so it takes much time to show the results.  
</p>

### How to run this project:
1. Download the [Sentiment140 dataset](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) and [Glove_42B_300d](http://nlp.stanford.edu/data/glove.42B.300d.zip) as a word embedding and put **zip files** into the **original_data** folder.

2. This dataset consists of more than **1 million** sentences from **Twitter** with many weird symbols that we need to clean this data and prepare them for the model, so run the below command to prepare and clean data. It takes a couple of hours. The **mode** parameter shows the destination model.

   * For the **Bert model**, run the below command:
      ```
      python preparing_data.py --mode='Bert' --glove42B_path ./original_data/glove.42B.300d.zip --original_dataset_path ./original_data/trainingandtestdata.zip
      ```
   * For **other models**, run the below commands:
      ```
      python preparing_data.py --mode='LSTM' --glove42B_path ./original_data/glove.42B.300d.zip --original_dataset_path ./original_data/trainingandtestdata.zip
      ```
      
3. After preparing data, one of the below commands can use to train your model and show results. These commands are based on your selected final model. The results are stored in the **results** folder.
   * For **LSTM** model:
      ```
      python main.py --mode='LSTM'
      ```
   * For **Bidirectional LSTM** model:
      ```
      python main.py --mode='LSTM_Bid'
      ```
   * For **Pyramid** model:
      ```
      python main.py --mode='Pyramid'
      ```
   * For **Bert** model:
      ```
      python main.py --mode='Bert'
      ```
<br />
<p align="justify">
You can access all parameters in the <strong>main.py</strong> file and change them. All necessary libraries are written in the <strong>requirements.txt</strong> file.  
</p>
