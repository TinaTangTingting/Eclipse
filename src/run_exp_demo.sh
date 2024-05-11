python run_exp_demo.py --rank 20  --num_seeds 5 --command train --outdir ../results_cora --todos_dir ../todos_train_cora --best_config_file best_config.pkl --only_create_todos
python run_exp_demo.py --rank 20  --num_seeds 5 --command train --outdir ../results_cora --todos_dir ../todos_train_cora --best_config_file best_config.pkl
python parser_ash.py --results_dir ../results_cora
python run_exp_demo.py --rank 20  --num_seeds 5 --command attack --outdir ../results_cora --todos_dir ../todos_attack_cora --best_config_file best_config.pkl --only_create_todos
python run_exp_demo.py --rank 20  --num_seeds 5 --command attack --outdir ../results_cora --todos_dir ../todos_attack_cora --best_config_file best_config.pkl
python parser_ash_trans_attack.py --results_dir ./eval_cora