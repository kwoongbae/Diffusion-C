import numpy as np

def final_train(model, datasets, corruptions):
    from model.dcgan import dcgan_inference
    from model.wgan import wgan_inference
    from model.ddpm.ddpm import inference
    from fid import calculate_fid
    
    for dataset in datasets:
        for corruption in corruptions:
            # load original datasets
            if dataset == 'mnist':
                original_dataset = np.load('original_datasets/{}/{}/train_images.npy'.format(dataset,corruption,))
            else:
                original_dataset = np.load('original_datasets/{}/{}.npy'.format(dataset,corruption,))
            
            random_indices = np.random.choice(original_dataset.shape[0], size=5000, replace=False)
            selected_datasets = original_dataset[random_indices]
            # load generated datasets
            if model == "dcgan":
                generated_datasets = dcgan_inference(dataset, corruption, 60,5000)
            elif model == "wgan":
                generated_datasets = dcgan_inference(dataset, corruption, 60,5000)
            elif model == "ddpm":
                generated_datasets = inference(dataset, corruption, 30)

            model_dir = "./generated_datasets/{}".format(model)
            np.save("{}/gen{}_{}.npy".format(model_dir, dataset, corruption), generated_datasets)

            print("[{}]-[{}] : [{}]".format(dataset, corruption, calculate_fid(generated_datasets, selected_datasets)))
            print()