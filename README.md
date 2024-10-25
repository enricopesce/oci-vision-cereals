## Prepare the data for the OCI Vision neural network

These images are not sufficient to train a model, I experienced that a neural network to work well requires more images filtered and augmented.

In my little iteration, I prepared a script to preprocess and augment the original images using some basic techniques:

- Resize and crop
- Apply random rotation
- Apply random brightness
- Apply random horizontal flip

other techniques are well documented [in this paper](https://arxiv.org/pdf/2301.02830).

You can find the Python script here: [https://github.com/enricopesce/oci-vision-cereals/blob/main/preprocessing.py](https://github.com/enricopesce/oci-vision-cereals/blob/main/preprocessing.py)

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

Now we are ready to use OCI Artificial Intelligence service!

### Labeling data with OCI Data Label

To build a model, before, we need to classify the images, in other words, we need to assign for every image a tag corrispondig the content: wheat, corn or sorghum.

Yes is it a boring phase but fortunately the script, executed before, generated a JSONL file with all metadata to import on OCI Vision for you!!!

The new processed folder contains all processed images classified by a folder and the metadata file in JSON line format supported by OCI Vision with all data needed.

```console
.
├── README.md
├── original
│   ├── corn
│   ├── sorghum
│   └── wheat
├── preprocessing.py
└── processed
 ├── corn
 ├── metadata.jsonl
 ├── sorghum
 └── wheat
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
