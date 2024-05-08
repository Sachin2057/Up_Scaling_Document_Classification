down
Copy code
## Data Pipeline 🚀

- Acquire an image
    - Download images in bulk.
- Organize and manage dataset 📂
    - Check the images and remove images that conform to specifications.
    - The raw images are organized in the following folder structure:
        ```
         RAW
            ├───Citizenship 🏛️
            ├───License 🚗
            ├───Others 🌐
            └───Passport 🛂
        ```
- Augment and create train-test split 🔄
    - `generate_dataset.py` augments and creates a train-test split from the above folder structure.
    - The script performs the following actions:
        - Augment the initial dataset
        - Split the dataset
        - Add additional augmentation for the validation dataset
        - Move the images and original images to the dataset folder.
        ```
        DATASET
            ├───Train 🚂
            └───Valid ✅
        ```