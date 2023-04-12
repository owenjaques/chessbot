# functions to split the data into smaller chunks for training
import numpy as np
import sys
sys.path.append("..")
import time

# change import and board initialization to match other versions
from enc_dec_chess import *


def split_data(data, chunk_size):
    # split a dataset into chunks of size chunk_size
    # chunk_size is an integer
    # returns a list of numpy arrays of 1-hot vectors

    # initialize chunks
    chunks = []

    # get number of chunks
    num_chunks = int(np.ceil(len(data) / chunk_size))

    # for each chunk
    for i in range(num_chunks):
        # get the start and end indices
        start = i * chunk_size
        end = (i + 1) * chunk_size

        # append the chunk to the list of chunks
        chunks.append(data[start:end])

    return chunks


# pre-process the data for a given chunk
def pre_process_chunk(chunk, dim=2):
    # pre-process a chunk of a dataset of chess games to be used for training

    # initialize board_image
    board_image = np.zeros((len(chunk), 64*dim + 64*64), dtype=np.int8)

    # for game in games
    for i, game in enumerate(chunk):
        # encode game
        board_image[i] = encode_game_fen(game[0], game[1].split()[0], dim)

    return board_image


# save the data as .npz to a file for training later
def save_data(data, file_name):
    # save a dataset to a file
    # data is a numpy array of 1-hot vectors of length 64*2 + 64*64
    # file_name is a string

    # save the data to a file
    np.savez_compressed(file_name, data=data)


# split the data into chunks and save them to files
def split_and_save_data(data, chunk_size, file_name, save_dir, dim=2):
    # split a dataset into chunks of size chunk_size and save them to files
    # chunk_size is an integer
    # file_name is a string

    # split the data into chunks
    chunks = split_data(data, chunk_size)

    start_time = time.time()
    # for each chunk
    for i, chunk in enumerate(chunks):
        
        # pre-process the chunk
        chunk = pre_process_chunk(chunk, dim)

        # save the chunk to a file in the data folder
        save_data(chunk, save_dir + file_name + str(i) + '.npz')
        print('Saved chunk ' + str(i) + ' to file.')

        # print the time it took to save the chunk
        print('Time to save chunk ' + str(i) + ': ' + str(time.time() - start_time) + ' seconds.')
        start_time = time.time()


def split_op(load_file, save_dir, file_name, chunk_size=131072, dim=2):
    data = pd.read_csv(load_file, header=None)
    data = data[[1,2]]
    
    # convert data to numpy array and split at a power of 2
    data = data.to_numpy()[:len(data) - len(data) % chunk_size]

    # shuffle the data
    np.random.shuffle(data)

    # split the data into chunks and save them to files
    split_and_save_data(data, chunk_size, file_name, save_dir, dim)



# main function
def main():

    split_op('data_raw/lichess_db_puzzle_all_moves.csv', 'version_64x12/data_puzzle_8/', 'data_', chunk_size=65536, dim=12)


    return


# run main function
if __name__ == '__main__':
    main()


