model=distilbert-base-nli-stsb-mean-tokens
dataset=kmdm
epoch=40

mkdir -p sentence_embeddings

python sentence_embedder.py data/${dataset}/OODtrain.tsv sentence_embeddings/${dataset}_OODtrain_${model}_${dataset}_mlm_epoch${epoch}_mean_embeddings.npy sentence_embedders/sbert.net_models_${model}-${dataset}-mlm
python sentence_embedder.py data/${dataset}/OODRemovedtrain.tsv sentence_embeddings/${dataset}_train_${model}_${dataset}_mlm_epoch${epoch}_mean_embeddings.npy ${model}-${dataset}-mlm
python sentence_embedder.py data/${dataset}/eval.tsv sentence_embeddings/${dataset}_val_${model}_${dataset}_mlm_epoch${epoch}_mean_embeddings.npy ${model}-${dataset}-mlm
python sentence_embedder.py data/${dataset}/test.tsv sentence_embeddings/${dataset}_test_${model}_${dataset}_mlm_epoch${epoch}_mean_embeddings.npy ${model}-${dataset}-mlm
python sentence_embedder.py data/${dataset}/test_tripadvisor.tsv sentence_embeddings/${dataset}_test_tripadvisor_${model}_${dataset}_mlm_epoch${epoch}_mean_embeddings.npy ${model}-${dataset}-mlm
python sentence_embedder.py data/${dataset}/subjective-questions.tsv sentence_embeddings/${dataset}_subjective_questions_${model}_${dataset}_mlm_epoch${epoch}_mean_embeddings.npy ${model}-${dataset}-mlm
