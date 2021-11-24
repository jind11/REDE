from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import sys
import numpy as np

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

file_path = sys.argv[1]
save_path = sys.argv[2]
model = sys.argv[3]

#Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    #Create sentences
    lines = [line.strip().split('\t') for line in open(file_path).readlines()]
    sentences = [line[-1] for line in lines]

    #Define the model
    model = SentenceTransformer(model)

    #Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    #Compute the embeddings using the multi-process pool
    emb = model.encode_multi_process(sentences, pool, batch_size=8)
    print("Embeddings computed. Shape:", emb.shape)
    # save emd
    # emb = normalize(emb, norm='l2', axis=1)
    np.save(save_path, emb)

    #Optional: Stop the proccesses in the pool
    # model.stop_multi_process_pool(pool)