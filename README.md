# People segmenter

This repository aims to provide a simple tutorial to train a model and deploy a web application that solve the people segmentation task. Given an image with a person in input, it should return a PNG image where all the background is made transparent.

# What is needed

## Dataset

The dataset is made by 290 images taken from the following github repository: https://github.com/VikramShenoy97/Human-Segmentation-Dataset

## Model

There are lots of models that are suitable for solving the image segmentation task. I used the U-Net model since it fits some memory requirements I had for deploying the web application for free. The model architecture and the training process is taken from the following github repository: https://github.com/usuyama/pytorch-unet

## Colab

To perform the training of the model, I used the notebook placed in the `training` folder. You can easily reproduce the same results running it on Google Colab, which provides free GPUs that allows you to speed up the training process.

## Back-end and front-end

I used FastAPI for developing the API which serves the model. Since I have no idea how to develop the front-end, I used the Swagger docs automatically generated by FastAPI as user interface.

## Heroku and Dropbox

I used Heroku for serving the web app for free. I just needed to create an account, to connect this Github repository and to enable the auto deploy at each commit in the main branch. The files related to Heroku are "Procfile" (which defines the command to launch the web app), "runtime.txt" (which defines the runtime to use) and "requirements.txt" (which defines the required python packages).
<br><br>
The weights of the model are saved in Dropbox using the training notebook. If you want to use it, you just need to fill "YOUR_ACCESS_TOKEN" with your dropbox access token in the last cell. The Heroku app downloads the model from Dropbox when it is launched. <br>
If you want to try the app locally, you need to add a file named ".env" at the same level of the other Heroku related files. It should contain a single line as follows: `token=your_access_token`. <br>
For use it on the Heroku server, you first need to add the token to the app configurations with the CLI as follows: `heroku config:set token=your_access_token -a app_name`.

# My app

You can use my app at the following url: https://people-segmenter.herokuapp.com/docs <br>
Click on the green bar at the beginning of the page, then click on the "Try it out" button on the right.
Use the "Browse..." button to select the image you want to segment and finally click on the "Execute" blue button.
The output PNG image will be displayed below. <br><br>
It could take some time to load the initial page at the first time since Heroku turns down the web app after 30 minutes of inactivity. Just wait for the page to load. <br>
The segmentation model should take some seconds to generate the output image.
