def generating_method(model, datasets, corruptions, num_imgs):
    import numpy as np
    import os
    from model.ddpm.ddpm import inference
    from model.dcgan import dcgan_inference
    from model.wgan import wgan_inference
    from model.ddim import ddim_inference
    EPOCH = num_imgs // 500

    # Start generating
    for dataset in datasets:
        for corruption in corruptions:
            generated_dataset = []
            print("{}-{}".format(dataset,corruption))
            for i in range(EPOCH):
                print("==================================")
                print("{}-{}-{}번째".format(dataset, corruption, i+1))
                print("==================================")
                if model == "ddpm":
                    generated_dataset.append(inference(dataset, corruption, 500))
                    # pickle로 저장하기
                elif model == "wgan":
                    generated_dataset.append(wgan_inference(dataset, corruption,60, 500))
                elif model == "dcgan":
                    generated_dataset.append(dcgan_inference(dataset, corruption,60, 500))
                elif model == "ddim":
                    generated_dataset.append(ddim_inference(dataset, corruption, 500))
                else:
                    print("Wrong model name")
                    
            generated_datasets = generated_dataset[0]
            for k in range(1,len(generated_dataset)):
                generated_datasets = np.concatenate((generated_datasets,generated_dataset[k]), axis = 0)
                print(generated_datasets.shape)
#             path = "./generated_datasets/{}".format(model)
            img_dir = "./generated_datasets/{}/{}".format(model,dataset)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            np.save("{}/{}_gen.npy".format(img_dir, corruption), generated_datasets)