## Prepare the data for the OCI Vision neural network

Basically a final user can only run the script and generate from the original images other images optimized and augmented ready to be trained on the OCI services.

```console
python preprocessing.py
```

### Coping data on OCI Object Storage bucket

Before importing the processed files we need to copy them on an OCI Object Storage, space used by the next steps.

Create a bucket and copy all processed folders inside:

```console
oci os object bulk-upload \
 --namespace-name YOURNAMESPACENAME \
 --bucket-name YOURBUCKETNAME \
 --src-dir processed/ --content-type image/jpeg
```

Now you have the same processed content in YOURBUCKETNAME ready to be imported!

Log-in on the OCI web console in your tenant and go to "Analticis & AI" --> "Machine Learning" --> "Data Labeling" --> "Dataset" --> "Import dataset"

### Create the model

Finally, we are ready to build the Oci Vision model and use it!

Create a new Project on the OCI Vision page and select the Data Label data.

"Analticis & AI" --> "Machine Learning" --> "Vision" --> "Project"

Go into the project and create a Model choose "Image classification" as the type and "grains" as a dataset (from the OCI Data Label)
 
Confirm all and continue to start the model build.

In this phase OCI build the model in automated way, the build time depends on the complexity of data, maximum 24 hours.

We don't need to set up any server or buy a GPU to train the model or split data and apply algorithms, all will done automatically by OCI Vision.

We have a very good F1 score for this model, now we can test it!

### Try the model

I have downloaded new random images related to the trained object and tested them:

We have a matching label for every image with over 90% confidence, a very good result! 

Now my OCI Vision can distinguish cereals!!
