import argparse
from HistEq import run_histeq
from Morph import run_morph, run_morph_multi_kernel_size
from Sharpen import run_sharpen

def histeq(img_dir, save_dir, img_name, img_type):
    if img_type == 'gray':
        run_histeq(img_dir, save_dir, img_name, type='gray')
    else:
        run_histeq(img_dir, save_dir, img_name, type='color')

def morph(img_dir, save_dir, img_name, img_type):
    if img_type == 'gray':
        run_morph_multi_kernel_size(img_dir, save_dir, img_name, type='gray', kernel_type = 1)
    else:
        run_morph_multi_kernel_size(img_dir, save_dir, img_name, type='color', color_type='rgb', kernel_type = 1)

def sharpen(img_dir, save_dir, img_name, img_type):
    if img_type == 'gray':
        run_sharpen(img_dir, save_dir, img_name, type='gray')
    else:
        run_sharpen(img_dir, save_dir, img_name, type='color')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", type=str, default='histeq', help="Task to run.", choices=['histeq', 'morph', 'sharpen']
    )
    parser.add_argument(
        "--img_name", type=str, default='histeq1', help="Img name to process, if set to all, run all img in the task dataset dir."
    )
    parser.add_argument(
        "--img_type", type=str, default='gray', help="Img type to process.", choices=['gray', 'color']
    )
    args = parser.parse_args()

    if args.task == 'histeq':
        task = 'HistEq'
        img_dir = 'dataset/%s/'%task
        save_dir = 'results/%s/'%task
        img_names = {'histeq1.jpg':'gray','histeq2.jpg':'gray','histeq3.jpg':'gray','histeq4.jpg':'gray','histeqColor.jpg':'color',
                    'gray_1.png':'gray','gray_2.png':'gray','gray_3.png':'gray','gray_4.png':'gray','gray_5.png':'gray',
                    'gray_6.png':'gray','gray_7.png':'gray','gray_8.png':'gray','gray_9.png':'gray','gray_10.png':'gray',
                    'color_1.png':'color','color_2.png':'color','color_3.png':'color','color_4.png':'color','color_5.png':'color',
                    'color_6.png':'color','color_7.png':'color','color_8.png':'color','color_9.png':'color','color_10.png':'color',
                    }
        if args.img_name == 'all':
            for img_name, img_type in img_names.items():
                histeq(img_dir, save_dir, img_name, img_type)
        else:
            assert args.img_name in img_names.keys()
            histeq(img_dir, save_dir, args.img_name, args.img_type)
    elif args.task == 'morph':
        task = 'Morph'
        img_dir = 'dataset/%s/'%task
        save_dir = 'results/%s/'%task
        img_names = {'word_bw.bmp':'gray',
                    'finger_bin_1.jpg':'gray','finger_bin_2.jpg':'gray','finger_bin_3.jpg':'gray',
                    'handwrite_bin_1.jpg':'gray','handwrite_bin_2.jpg':'gray','handwrite_bin_3.jpg':'gray',
                    'finger_gray_1.jpg':'gray','finger_gray_2.jpg':'gray','finger_gray_3.jpg':'gray',
                    'handwrite_gray_1.jpg':'gray','handwrite_gray_2.jpg':'gray','handwrite_gray_3.jpg':'gray',
                    'star_gray_1.jpg':'gray','star_gray_2.jpg':'gray',
                    'dct5_gray.jpg':'gray','gray_1.png':'gray','gray_2.png':'gray',
                    'finger_color_1.jpg':'color','finger_color_2.jpg':'color','finger_color_3.jpg':'color',
                    'handwrite_color_1.jpg':'color','handwrite_color_2.jpg':'color','handwrite_color_3.jpg':'color',
                    'star_color_1.jpg':'color','star_color_2.jpg':'color',
                    'histeqColor.jpg':'color','color_1.png':'color','color_2.png':'color',
                    }
        if args.img_name == 'all':
            for img_name, img_type in img_names.items():
                morph(img_dir, save_dir, img_name, img_type)
        else:
            assert args.img_name in img_names.keys()
            morph(img_dir, save_dir, args.img_name, args.img_type)
    elif args.task == 'sharpen':
        task = 'Sharpen'
        img_dir = 'dataset/%s/'%task
        save_dir = 'results/%s/'%task
        img_names = {'moon.tif':'gray','histeqColor.jpg':'color',
                    'gray_1.png':'gray','gray_2.png':'gray','gray_3.png':'gray','gray_4.png':'gray','gray_5.png':'gray',
                    'gray_6.png':'gray','gray_7.png':'gray','gray_8.png':'gray','gray_9.png':'gray','gray_10.png':'gray',
                    'color_1.png':'color','color_2.png':'color','color_3.png':'color','color_4.png':'color','color_5.png':'color',
                    'color_6.png':'color','color_7.png':'color','color_8.png':'color','color_9.png':'color','color_10.png':'color',
                    }
        if args.img_name == 'all':
            for img_name, img_type in img_names.items():
                sharpen(img_dir, save_dir, img_name, img_type)
        else:
            assert args.img_name in img_names.keys()
            sharpen(img_dir, save_dir, args.img_name, args.img_type)