##  I/O concurrency --> threading 
## CPU concurrency --> multiprocessing


import concurrent.futures
import time

## Demo -Threading

start = time.perf_counter()


def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    return f'Done Sleeping...{seconds}'


with concurrent.futures.ThreadPoolExecutor() as executor:
    secs = [5, 4, 3, 2, 1]
    results = executor.map(do_something, secs)

    # for result in results:
    #     print(result)

# threads = []

# for _ in range(10):
#     t = threading.Thread(target=do_something, args=[1.5])
#     t.start()
#     threads.append(t)

# for thread in threads:
#     thread.join()

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')

# Sleeping 5 second(s)...
# Sleeping 4 second(s)...
# Sleeping 3 second(s)...
# Sleeping 2 second(s)...
# Sleeping 1 second(s)...
# Finished in 5.08 second(s)

## Demo Multiprocessing 

start = time.perf_counter()


def do_something(seconds):
    print(f'Sleeping {seconds} second(s)...')
    time.sleep(seconds)
    return f'Done Sleeping...{seconds}'


with concurrent.futures.ProcessPoolExecutor() as executor:
    secs = [5, 4, 3, 2, 1]
    results = executor.map(do_something, secs)

    # for result in results:
    #     print(result)

finish = time.perf_counter()

print(f'Finished in {round(finish-start, 2)} second(s)')
# Sleeping 5 second(s)...
# Sleeping 4 second(s)...
# Sleeping 3 second(s)...
# Sleeping 2 second(s)...
# Sleeping 1 second(s)...
# Finished in 5.54 second(s)
---------------------------------------------------------------------------------------------------------------------------------

## use 2 different files for threading & multiprocessing

# Download images with Threading 

import requests,os

img_urls = [
    'https://images.unsplash.com/photo-1516117172878-fd2c41f4a759',
    'https://images.unsplash.com/photo-1532009324734-20a7a5813719',
    'https://images.unsplash.com/photo-1524429656589-6633a470097c',
    'https://images.unsplash.com/photo-1530224264768-7ff8c1789d79',
    'https://images.unsplash.com/photo-1564135624576-c5c88640f235',
    'https://images.unsplash.com/photo-1541698444083-023c97d3f4b6',
    'https://images.unsplash.com/photo-1522364723953-452d3431c267',
    'https://images.unsplash.com/photo-1513938709626-033611b8cc03',
    'https://images.unsplash.com/photo-1507143550189-fed454f93097',
    'https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e',
    'https://images.unsplash.com/photo-1504198453319-5ce911bafcde',
    'https://images.unsplash.com/photo-1530122037265-a5f1f91d3b99',
    'https://images.unsplash.com/photo-1516972810927-80185027ca84',
    'https://images.unsplash.com/photo-1550439062-609e1531270e',
    'https://images.unsplash.com/photo-1549692520-acc6669e2f0c'
]

t1 = time.perf_counter()
os.makedirs("Test_img")
base_path = os.path.join(os.getcwd()+"/Test_img/")

def download_image(img_url):
    img_bytes = requests.get(img_url).content
    img_name = img_url.split('/')[3]
    img_name = f'{img_name}.jpg'
    with open(base_path+img_name, 'wb') as img_file:
        img_file.write(img_bytes)
        print(f'{img_name} was downloaded...')


with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(download_image, img_urls)


t2 = time.perf_counter()

print(f'Finished in {t2-t1} seconds')
# photo-1513938709626-033611b8cc03.jpg was downloaded...
# photo-1507143550189-fed454f93097.jpg was downloaded...
# photo-1516117172878-fd2c41f4a759.jpg was downloaded...
# photo-1564135624576-c5c88640f235.jpg was downloaded...
# photo-1541698444083-023c97d3f4b6.jpg was downloaded...
# photo-1530224264768-7ff8c1789d79.jpg was downloaded...
# photo-1522364723953-452d3431c267.jpg was downloaded...
# photo-1516972810927-80185027ca84.jpg was downloaded...
# photo-1532009324734-20a7a5813719.jpg was downloaded...
# photo-1524429656589-6633a470097c.jpg was downloaded...
# photo-1530122037265-a5f1f91d3b99.jpg was downloaded...
# photo-1504198453319-5ce911bafcde.jpg was downloaded...
# photo-1549692520-acc6669e2f0c.jpg was downloaded...
# photo-1550439062-609e1531270e.jpg was downloaded...
# photo-1493976040374-85c8e12f0c0e.jpg was downloaded...
# Finished in 20.413230311000007 seconds


# Processing images - Multiprocessing Images

from PIL import Image, ImageFilter


t1 = time.perf_counter()

size = (1200, 1200)

os.makedirs("Processed_img")
target_path = os.path.join(os.getcwd()+"/Processed_img/")
base_path = os.path.join(os.getcwd()+"/Test_img/")
img_names = []
for file in os.listdir("Test_img"):
    img_names.append(base_path+file)

def process_image(img_name):
    img = Image.open(img_name)

    img = img.filter(ImageFilter.GaussianBlur(15))

    img.thumbnail(size)
    target = target_path+img_name.split("/")[-1]
    img.save(target)
    print(f'{img_name.split("/")[-1]} was processed...')


with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(process_image, img_names)


t2 = time.perf_counter()

print(f'Finished in {t2-t1} seconds')
# photo-1532009324734-20a7a5813719.jpg was processed...
# photo-1549692520-acc6669e2f0c.jpg was processed...
# photo-1530224264768-7ff8c1789d79.jpg was processed...
# photo-1516972810927-80185027ca84.jpg was processed...
# photo-1530122037265-a5f1f91d3b99.jpg was processed...
# photo-1541698444083-023c97d3f4b6.jpg was processed...
# photo-1493976040374-85c8e12f0c0e.jpg was processed...
# photo-1516117172878-fd2c41f4a759.jpg was processed...
# photo-1504198453319-5ce911bafcde.jpg was processed...
# photo-1522364723953-452d3431c267.jpg was processed...
# photo-1550439062-609e1531270e.jpg was processed...
# photo-1524429656589-6633a470097c.jpg was processed...
# photo-1564135624576-c5c88640f235.jpg was processed...
# Finished in 0.5562845160002325 seconds


------------------------------------------------------------------------------------------------------------------------------------
import os
from contextlib import contextmanager


@contextmanager
def change_dir(destination):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)


with change_dir('Sample-Dir-One'):
    print(os.listdir())     # ['work.txt', 'mydoc.txt', 'todo.txt']

with change_dir('Sample-Dir-Two'):
    print(os.listdir())    # ['sample.txt', 'test.txt', 'demo.txt']


class Open_File():

    def __init__(self, destination):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, traceback):
        pass


#### Using contextlib ####

@contextmanager
def open_file(file, mode):
    f = open(file, mode)
    yield f
    f.close()


with open_file('sample.txt', 'w') as f:
    f.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit.')

print(f.closed) # True


#### CD Example ####

cwd = os.getcwd()
os.chdir('Sample-Dir-One')
print(os.listdir()) # ['work.txt', 'mydoc.txt', 'todo.txt']
os.chdir(cwd)
print(os.listdir()) # ['Test2.cpp', 'Sample-Dir-One' ... ]


cwd = os.getcwd()
os.chdir('Sample-Dir-Two')
print(os.listdir()) # ['sample.txt', 'test.txt', 'demo.txt']
os.chdir(cwd)
print(os.listdir()) # ['Test2.cpp', 'Sample-Dir-One' ... ]
