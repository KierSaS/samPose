from SamPose_model_api import *
from utils.draw_utils import draw_bbox_3d, draw_axis
def load_intrinsic_matrix(file_path):
    # 读取文件内容
    with open(file_path, 'r') as file:
        lines = file.readlines()

    matrix = np.array([list(map(float, line.split())) for line in lines])

    return matrix

if __name__  == "__main__":
    ckpt, model_type = get_model_info("h")
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    DEVICE = "cuda"
    sam.to(device=DEVICE)

    MASK_GEN = SamAutomaticMaskGenerator(sam)


    logger.info(f"load SAM model from {ckpt}")

    crop_tool = CropImage()

    dinov2_model = load_dinov2_model()
    dinov2_model.to("cuda:0")


    #####################
    prompt_filename = os.path.join("/home/wubin/code/SamPose/data/demo/T-LESS/color/000205.png")  #########################################
    targe_filename = os.path.join("/home/wubin/code/SamPose/data/demo/T-LESS/color_full/000199.png")  ##############################

    file0_path = '/home/wubin/code/SamPose/data/demo/T-LESS/color/205.txt'#############################
    # 从文件中加载内参矩阵
    K0 = load_intrinsic_matrix(file0_path)

    file1_path = '/home/wubin/code/SamPose/data/demo/T-LESS/color_full/199.txt'  ###########################
    K1 = load_intrinsic_matrix(file1_path)

####这个包围盒也要改
    x, y , z = 49.75, 49.75, 24.2
    _3d_bbox = np.array([
        [-x, -y , -z],
        [-x, -y , z],
        [-x, y , z],
        [-x, y , -z],
        [x, -y , -z],
        [x, -y , z],
        [x, y , z],
        [x, y , -z],
    ])#定义一个三维边界框_3d_bbox以绘图

    print(f"File path: {prompt_filename}")
    prompt_image = cv2.imread(prompt_filename)
    if prompt_image is None:
        print("Failed to load image!")
    # prompt_image = cv2.imread(prompt_filename)
    prompt_image_copy = prompt_image.copy()

    ref_torch_image = set_torch_image(prompt_image, center_crop=True)
    ref_fea = get_cls_token_torch(dinov2_model, ref_torch_image)

    target_image = cv2.imread(targe_filename)
    image_h,image_w,_ = target_image.shape

    t1 = time.time()
    masks = MASK_GEN.generate(target_image)
    t2 = time.time()


    similarity_score, top_images  = np.array([0,0,0],np.float32) , [[],[],[]]
    t3 = time.time()

    compact_percent = 0.3


    for _, mask in enumerate(masks):
        object_mask = np.expand_dims(mask["segmentation"], -1)
        x0, y0, w, h = mask["bbox"]

        x1, y1 = x0+w,y0+h
        x0 -= int(w * compact_percent)
        y0 -= int(h * compact_percent)
        x1 += int(w * compact_percent)
        y1 += int(h * compact_percent)


        box = np.array([x0, y0, x1, y1])
        resize_shape = np.array([y1 - y0, x1 - x0])
        K_crop, K_crop_homo = get_K_crop_resize(box, K1, resize_shape)
        image_crop, _ = get_image_crop_resize(target_image, box, resize_shape)

        # object_mask,_ = get_image_crop_resize(object_mask, box, resize_shape)
        box_new = np.array([0, 0, x1 - x0, y1 - y0])
        resize_shape = np.array([256, 256])
        K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
        image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)

        crop_tensor = set_torch_image(image_crop, center_crop=True)
        with torch.no_grad():
            fea = get_cls_token_torch(dinov2_model, crop_tensor)
            #将裁剪后的图像转换为PyTorch张量，并使用DINO v2提取其特征。

        score = F.cosine_similarity(ref_fea, fea, dim=1, eps=1e-8)


        if  (score.item() > similarity_score).any():
            mask["crop_image"] = image_crop
            mask["K"] = K_crop
            mask["bbox"] = box

            min_idx = np.argmin(similarity_score)

            similarity_score[min_idx] = score.item()
            top_images[min_idx] = mask.copy()


    prompt_image =  cv2.cvtColor(prompt_image, cv2.COLOR_BGR2GRAY)
    prompt_image = torch.from_numpy(prompt_image).float()[None] / 255.
    prompt_image = prompt_image.unsqueeze(0).cuda()



    matching_score =  [ [0] for _ in range(len(top_images)) ]

    for top_idx in range(len(top_images)):

        img1 =  cv2.cvtColor(top_images[top_idx]["crop_image"], cv2.COLOR_BGR2GRAY)

        img1 = torch.from_numpy(img1).float()[None] / 255.

        img1 = img1.unsqueeze(0).cuda()

        batch = {'image0': prompt_image, 'image1': img1}


        with torch.no_grad():


            matcher(batch)

            mkpts0 = batch['mkpts0_f'].cpu().numpy()

            mkpts1 = batch['mkpts1_f'].cpu().numpy()

            confidences = batch["mconf"].cpu().numpy()



        conf_mask = np.where(confidences > 0.9)

        matching_score[top_idx] = conf_mask[0].shape[0]

        top_images[top_idx]["mkpts0"] = mkpts0

        top_images[top_idx]["mkpts1"] = mkpts1

        top_images[top_idx]["mconf"] = confidences

    max_match_idx = np.argmax(matching_score)

    pre_bbox  = top_images[max_match_idx]["bbox"]
    mkpts0 = top_images[max_match_idx]["mkpts0"]
    mkpts1 = top_images[max_match_idx]["mkpts1"]
    pre_K = top_images[max_match_idx]["K"]
    
    crop_image = cv2.resize(top_images[np.argmax(matching_score)]["crop_image"],(256,256))        
    que_image = cv2.resize(prompt_image_copy,(256,256))

    segment_mask = (255*top_images[np.argmax(matching_score)]["segmentation"]).astype(np.uint8)
    stack_result_image = np.hstack((que_image, crop_image)) 
    cv2.imwrite("query_result.png", stack_result_image)

    R, t, inliers = estimate_pose(mkpts0, mkpts1 , K0 , pre_K , 0.5, 0.99)
    prompt_pose = np.loadtxt(os.path.join("/home/wubin/code/SamPose/data/demo/T-LESS/pose","205.txt")) ###########################
    target_pose = np.loadtxt(os.path.join("/home/wubin/code/SamPose/data/demo/T-LESS/pose","199.txt"))  ##########################


    predict_pose = np.zeros((3,4)).astype(np.float32)
    predict_pose[:3,:3] =  np.matmul(R, prompt_pose[:3,:3])
    our_predict_pose = predict_pose[:3,:3].copy()
    predict_pose[:3,3] = target_pose[:3,3]


    pre_bbox_pts_3d, _ = project_points(_3d_bbox, predict_pose[:3,:4] , K1)
    pre_bbox_pts_3d = pre_bbox_pts_3d.astype(np.int32)

    our_bbox_img = draw_bbox_3d(target_image, pre_bbox_pts_3d,(255,0,0))
    our_bbox_img = draw_axis(our_bbox_img,predict_pose[:3,:3], predict_pose[:3,3],K1)

    true_bbox_pts_3d, _ = project_points(_3d_bbox, target_pose[:3, :4], K1)
    true_bbox_pts_3d = true_bbox_pts_3d.astype(np.int32)

    our_bbox_img = draw_bbox_3d(our_bbox_img, true_bbox_pts_3d, (0, 255, 0))
    directory = "/home/wubin/code/POPE/data/demo/T-LESS/pose"
    save_path = os.path.join(directory, "3D_BBox-205-199.png")
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(save_path, our_bbox_img)