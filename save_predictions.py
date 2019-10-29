import glob

def show_inference(model, image_path, save_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
  # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    img = Image.fromarray(image_np)
    img.save(save_path)
    
failed_list = []
test_path = glob.glob("./models/research/object_detection/ExDark/*/*.jpg")
test_path.extend(glob.glob("./models/research/object_detection/ExDark/*/*.png"))
for idx, img_path in enumerate(test_path):
    savefolder = '.'+os.path.dirname(img_path)[34:]
    savepath = savefolder+'/'+os.path.basename(img_path)
    print(savepath)
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    try:
        show_inference(masking_model, img_path, savepath)
    except:
        failed_list.append(img_path)
    print(idx, len(test_path))
