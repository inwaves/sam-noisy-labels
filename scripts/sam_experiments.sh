# CIFAR-10
python3 train.py --dataset "cifar10" --optimiser-choice "SAM" --label_type="aggre" --initial_rho=0.5 --adaptive=False --rho_scheduler="constant" --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SGD" --label_type="worse" --noise=0.8 --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SGD" --label_type="blue" --noise=0.8 --epochs=200
python3 train.py --dataset "cifar10" --optimiser-choice "SGD" --label_type="blue" --noise=1.0 --epochs=200

# CIFAR-100
python3 train.py --dataset "cifar100" --optimiser-choice "SGD" --label_type="clean" --epochs=200
python3 train.py --dataset "cifar100" --optimiser-choice "SGD" --label_type="aggre" --epochs=200 # 18%
python3 train.py --dataset "cifar100" --optimiser-choice "SGD" --label_type="worse" --epochs=200 # 40%
python3 train.py --dataset "cifar100" --optimiser-choice "SGD" --label_type="blue" --noise=0.2 --epochs=200
python3 train.py --dataset "cifar100" --optimiser-choice "SGD" --label_type="blue" --noise=0.8 --epochs=200
python3 train.py --dataset "cifar100" --optimiser-choice "SGD" --label_type="blue" --noise=1.0 --epochs=200