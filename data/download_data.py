import os
import requests 

def get_data(url: str, filename:str): 
    if(url == None): 
        raise ValueError('Empty URL')
    try: 
        response = requests.get(url)
        response.raise_for_status() 

        current_dir = os.getcwd() 
        file_save_path = os.path.join(current_dir, filename)

        with open(file_save_path, 'wb') as file: 
            file.write(response.content)
        
        print('\nDataset Downloaded Successfully!\n')
    except requests.exceptions.RequestException as message: 
        print(f'\nError:\n{message}')

if __name__ == '__main__': 
    url = 'https://storage.googleapis.com/gresearch/dakshina/dakshina_dataset_v1.0.tar'
    filename = 'google_dakshina_dataset'
    get_data(url, filename)