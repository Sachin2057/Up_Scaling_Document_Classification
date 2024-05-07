## Data pipeline

- Acquire a image
    - Download images in bulk.
- Organize and manage dataset
    - Check the images and remove images taht confirm to specification
    -  The raw images are organized in  the follwing folder structure
        ```
         RAW
            ├───Citizenship
            ├───Licens
            ├───Others
            └───Passport
        ```
- Augment and create train-test split
    - generate_dataset.py augements and create train-test split from above Folder structure
    - The scripts performs the following action
        - Augment the initial dataset
        - Split the dataset
        - Add additional augmenation for valid dataset
        - Move the images and orginal images to dataset folder.
        ```
        DATASET
            ├───Train
            └───Valid
        ```
