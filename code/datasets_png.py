def original_datasets_png(datasets,corruptions):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    import tensorflow as tf
    import tensorflow_datasets as tfds
    for dataset in datasets:
        for corruption in corruptions:
            if corruption == "fog_severity_2":
                original_datasets = np.load("original_datasets/severity_exp/{}/{}.npy".format(dataset, corruption))
            elif corruption == "fog_severity_1":
                original_datasets = np.load("original_datasets/severity_exp/{}/{}.npy".format(dataset, corruption))
            elif corruption == "fog_severity_3":
                original_datasets = np.load("original_datasets/severity_exp/{}/{}.npy".format(dataset, corruption))
            elif corruption == "fog_severity_4":
                original_datasets = np.load("original_datasets/severity_exp/{}/{}.npy".format(dataset, corruption))
            elif corruption == "fog_severity_5":
                original_datasets = np.load("original_datasets/severity_exp/{}/{}.npy".format(dataset, corruption))
            else:
                original_datasets = np.load("original_datasets/{}/{}.npy".format(dataset, corruption))

            original_datasets = original_datasets[np.random.choice(len(original_datasets)
                                                                   , size=5000, replace=False)]

            resized_image_array = np.zeros((5000, 64, 64, 3))
            for i in range(5000):
                image = Image.fromarray(original_datasets[i])
                resized_image = image.resize((64, 64))
                resized_image_array[i] = np.array(resized_image)
            original_datasets = resized_image_array
            original_datasets = original_datasets.astype(np.uint8)
            print("{}-{}:({},{}),{},{}".format(dataset, corruption,np.min(original_datasets)
                                            , np.max(original_datasets),original_datasets.shape, type(original_datasets[0][0][0][0])))

            img_dir = "original_imgs/{}/{}".format(dataset, corruption)
            os.makedirs(img_dir, exist_ok=True)
            
            arr = original_datasets[0:6]
            fig, axs = plt.subplots(1, 6, figsize=(12, 12))
            for i, ax in enumerate(axs.flat):
                ax.imshow(arr[i], interpolation="nearest")
                ax.axis('off')
            plt.show()

            for i in range(len(original_datasets)):
                # create an Image object from the array
                img = Image.fromarray(original_datasets[i])

                # save the image as PNG
                img.save("{}/{}_{}_{}.png".format(img_dir, dataset, corruption,i))
            print("saved path is {}/".format(img_dir))
                
def generated_datasets_png(models,datasets,corruptions):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from PIL import Image
    for model in models:
        for dataset in datasets:
            for corruption in corruptions:
                print("=================================")
                generated_datasets = np.load("generated_datasets/{}/{}/{}_gen.npy".format(model, dataset, corruption))
                generated_datasets = (generated_datasets*127.5) + 127.5
                generated_datasets = generated_datasets.astype(np.uint8)
                print("{}-{}-{}:({},{})".format(model,dataset, corruption,np.min(generated_datasets), np.max(generated_datasets)))

                random_arr = np.random.choice(len(generated_datasets), size=6, replace=False)
                arr = generated_datasets[random_arr]
                fig, axs = plt.subplots(1, 6, figsize=(12, 12))
                for i, ax in enumerate(axs.flat):
                    ax.imshow(arr[i], interpolation="nearest")
                    ax.axis('off')
                plt.show()

                img_dir = "generated_imgs/{}/{}/{}".format(model,dataset,corruption)
                os.makedirs(img_dir, exist_ok=True)

                # save each image from the array
                for i in range(len(generated_datasets)):
                    # create an Image object from the array
                    img = Image.fromarray(generated_datasets[i])

                    # save the image as PNG
                    img.save("{}/{}_{}_{}.png".format(img_dir, dataset, corruption,i))
                print("saved path is {}/".format(img_dir))
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")