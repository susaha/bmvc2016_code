function  computeCVOF(image_folder, savepath)
img_list = dir(image_folder);
num_imgs = size(img_list,1);
num_imgs = num_imgs - 2;
for i = 1:num_imgs
    output_flow_img = sprintf('%s/%05d.jpg', savepath, i);
    fprintf('output flow image: %s\n', output_flow_img);
    if ~exist(output_flow_img, 'file')
        if i < num_imgs % if it is not the last frame
            fprintf('Computing CVOF (Classicial Variation Optical Flow) (Thomas Brox ECCV 2004)...\n');
            th = tic();
            img_file_1 = sprintf('%s/%05d.jpg', image_folder, i);
            fprintf('input image1: %s\n', img_file_1);
            img_file_2 = sprintf('%s/%05d.jpg', image_folder, i+1);
            fprintf('input image2: %s\n', img_file_2);
            im1 = imread(img_file_1);
            im2 = imread(img_file_2);
            flow_img = compute_flow(im1, im2);
        else % if it is the last frame then pick the flow image for previous frame
            prev_flow_img = sprintf('%s/%05d.jpg', savepath, i-1);
            flow_img = imread(prev_flow_img);
        end
        imwrite(flow_img, output_flow_img);
    else
        fprintf('output flow image file already exisit : %s\n', output_flow_img);
    end
end



