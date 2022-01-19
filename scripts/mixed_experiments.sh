conda activate base
python train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=0.1 --adaptive=False --label_type="worse" --epochs=200
python train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=0.1 --adaptive=False --label_type="blue" --noise=0.4 --epochs=200
python train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=0.1 --rho_scheduler="stepdecay" --adaptive=False --label_type="worse"  --epochs=200
python train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=0.1 --rho_scheduler="stepdecay" --adaptive=False --label_type="blue" --noise=0.4 --epochs=200

python train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=2.0 --adaptive=True --label_type="worse" --epochs=200
python train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=2.0 --adaptive=True --label_type="blue" --noise=0.4 --epochs=200
python train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=2.0 --rho_scheduler="stepdecay" --adaptive=True --label_type="worse"  --epochs=200
python train.py --dataset "cifar10" --optimiser-choice "SAM" --initial_rho=2.0 --rho_scheduler="stepdecay" --adaptive=True --label_type="blue" --noise=0.4 --epochs=200