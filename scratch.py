device = torch_stuff["device"]
    model = torch_stuff["model"].to(device)
    data_transform = torch_stuff["data_transform"]

    VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR = data_dirs["VAL_MRI_DIR"], data_dirs["VAL_SEG_DIR"], data_dirs["DICE_DIR"]
    BATCH_SIZE = hyperparams["batch_size"]
    val_data = FinetuneDiceDataset(VAL_MRI_DIR, VAL_SEG_DIR, DICE_DIR, img_transform=data_transform)
    val_dataloader = get_dataloader(val_data, BATCH_SIZE)

    #model.eval()

    background, images = None, None
    use_for_background = True
    slice = -1

    for itr, data in enumerate(val_dataloader):
        imgs, segs = data["img"].to(device), data["seg"].to(device)
        if ((itr+1)*BATCH_SIZE) > num_background:
            use_for_background = False
            slice = num_background - (itr)*BATCH_SIZE

        if slice != 0: 
            segs_slice = segs[:slice, ...]
            img_tensor = imgs[:slice, ...]
        elif slice == -1: img_tensor = imgs
        else: pass

        img_tensor = torch.cat((img_tensor, segs_slice), dim=1) 

        if background is not None: background = torch.cat((background, img_tensor), dim=0)
        else: background = img_tensor

        if not(use_for_background): 
            images = imgs[slice:slice+num_test, ...]
            segs = segs[slice:slice+num_test, ...]
            images = torch.cat((images, segs), dim=1)
            break

        if (itr >= 1): del img_tensor
        del imgs, segs, segs_slice
    
    print(f"Image shape: {images.size()}")
    print(f"Background shape: {background.size()}")
    e = shap.DeepExplainer(model, background)
    shap_values, _ = e.shap_values(images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(images.numpy(), 1, -1), 1, 2)
    shap.image_plot(shap_numpy, -test_numpy)