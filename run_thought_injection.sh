# model_id=meta-llama/Llama-3.1-8B-Instruct
# for layer in 15 20 23; do
#     for magnitude in 2.0 4.0; do
#         python3 thought_injection.py $model_id $layer $magnitude
#     done
# done

# model_id=Qwen/Qwen2.5-7B-Instruct
# for layer in 13 18 22; do
#     for magnitude in 2.0 4.0; do
#         python3 thought_injection.py $model_id $layer $magnitude
#     done
# done

model_id=google/gemma-2-9b-it
for layer in 20 27 34; do
    for magnitude in 2.0 4.0; do
        python3 thought_injection.py $model_id $layer $magnitude
    done
done

model_id=tiiuae/Falcon3-7B-Instruct
for layer in 13 18 22; do
    for magnitude in 2.0 4.0; do
        python3 thought_injection.py $model_id $layer $magnitude
    done
done