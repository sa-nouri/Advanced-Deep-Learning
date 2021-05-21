from fid_score.fid_score import FidScore

directory_real_img = '/real_img/'
directory_fake_img = '/real_img/'

device = tf.device('cuda') # 1: for own device
fid = FidScore([directory_real_img, directory_fake_img], device, 256)
score = fid.calculate_fid_score()
print(f'The FID Score = {score}')