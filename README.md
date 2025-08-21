sudo apt update  

sudo apt install libssl-dev

nvcc -o argon2_miner main.cu argon2.cu -lssl -lcrypto

./argon2_miner '{"id":123,"time":1712345678}' 32
