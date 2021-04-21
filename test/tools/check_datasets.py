# %%
import os
if os.path.abspath('.').split('/')[-1] != 'yolov4-tiny':
    os.chdir('../..')
print(os.path.abspath('.'))

# %%
def print_color(s, end='\n', color=[]):
    if color:
        try:
            from colorama import Fore, Back, Style
            color = list(map(eval, color))
        except:
            color = []

    if color:
        print(*color, end='')
    print(s, end=end)
    if color:
        print(Style.RESET_ALL)

# %%
import os
from tqdm import tqdm

from config import cfg

datasets_labels_path = cfg.train_datasets_labels_path
datasets_images_path = cfg.train_datasets_images_path
files_name = [x.split('.')[0] for x in os.listdir(datasets_labels_path) if x.endswith('.txt')]
labels_path = [f'{datasets_labels_path}/{x}.txt' for x in files_name]
images_path = [f'{datasets_images_path}/{x}.jpg' for x in files_name]

not_exist_files = []
for img_path in tqdm(images_path):
    if not os.path.exists(img_path):
        not_exist_files.append(img_path)

print('\n\n\n-------')
for file in not_exist_files:
    print(file)
print('\n-------')
if not_exist_files:
    # print(f'总计不存在的图片数量为: {Fore.RED}{Back.WHITE}{Style.DIM}{len(not_exist_files)}{Style.RESET_ALL}')
    print_color(f'总计不存在的图片数量为: {len(not_exist_files)}', color=['Fore.RED', 'Back.WHITE', 'Style.DIM'])
else:
    # print(f'{Fore.GREEN}数据集完整! (标签对应的图片都存在){Style.RESET_ALL}')
    print_color(f'数据集完整! (标签对应的图片都存在)', color=['Fore.GREEN'])

# # %% test
# def print_color(s, end='\n', color=[]):
#     if color:
#         try:
#             from colorama import Fore, Back, Style
#             color = list(map(eval, color))
#         except:
#             color = []

#     if color:
#         print(*color, end='')
#     print(s, end=end)
#     if color:
#         print(Style.RESET_ALL)

# print_color(f'test', color=['Fore.RED', 'Back.WHITE', 'Style.DIM'])